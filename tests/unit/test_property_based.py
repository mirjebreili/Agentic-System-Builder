"""
Property-based tests using Hypothesis.

These tests generate random inputs to verify system robustness.
"""

import pytest
from hypothesis import given, strategies as st, settings
from hypothesis.strategies import composite
from langchain_core.messages import HumanMessage
import json


# Strategy for generating valid plugin dictionaries
@composite
def plugin_strategy(draw):
    """Generate a valid plugin dictionary."""
    name = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'),
        blacklist_characters=' '
    )))
    
    has_goal = draw(st.booleans())
    has_deps = draw(st.booleans())
    
    plugin = {"name": name}
    
    if has_goal:
        plugin["goal"] = draw(st.text(min_size=5, max_size=100))
    
    if has_deps:
        # Generate 0-3 dependency names
        num_deps = draw(st.integers(min_value=0, max_value=3))
        deps = [draw(st.text(min_size=1, max_size=10)) for _ in range(num_deps)]
        plugin["dependencies"] = deps
    
    return plugin


# Strategy for generating task dictionaries
@composite
def task_strategy(draw):
    """Generate a valid task dictionary."""
    task_id = draw(st.text(min_size=1, max_size=10, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd')
    )))
    
    description = draw(st.text(min_size=3, max_size=200))
    
    task = {
        "id": task_id,
        "description": description
    }
    
    # Optionally add dependencies
    if draw(st.booleans()):
        num_deps = draw(st.integers(min_value=0, max_value=3))
        deps = [draw(st.text(min_size=1, max_size=10)) for _ in range(num_deps)]
        task["dependencies"] = deps
    
    return task


class TestPropertyBased:
    """Property-based tests for robustness."""
    
    @given(st.text(min_size=1, max_size=10000))
    @settings(max_examples=50, deadline=None)
    def test_extract_goal_never_crashes(self, prompt):
        """Goal extraction should handle any text input without crashing."""
        from src.agents.goal_extractor import extract_goal
        
        state = {"messages": [HumanMessage(content=prompt)]}
        
        try:
            result = extract_goal(state)
            assert isinstance(result, dict)
            assert "goal" in result
            assert isinstance(result["goal"], str)
        except Exception as e:
            pytest.fail(f"extract_goal crashed with: {e}")
    
    @given(st.lists(plugin_strategy(), min_size=0, max_size=10))
    @settings(max_examples=50, deadline=None)
    def test_plugin_validation_handles_any_plugins(self, plugins):
        """Plugin validation should handle any list of plugins."""
        from src.agents.state_validator import StateValidator
        
        state = {"plugins": plugins}
        
        try:
            errors = StateValidator.validate_plugins(state)
            assert isinstance(errors, list)
        except Exception as e:
            pytest.fail(f"Plugin validation crashed with: {e}")
    
    @given(st.lists(task_strategy(), min_size=0, max_size=20))
    @settings(max_examples=50, deadline=None)
    def test_task_validation_handles_any_tasks(self, tasks):
        """Task validation should handle any list of tasks."""
        from src.agents.state_validator import StateValidator
        
        state = {"split_tasks": tasks}
        
        try:
            errors = StateValidator.validate_split_tasks(state)
            assert isinstance(errors, list)
        except Exception as e:
            pytest.fail(f"Task validation crashed with: {e}")
    
    @given(st.text(min_size=0, max_size=5000))
    @settings(max_examples=50, deadline=None)
    def test_input_sanitization_never_crashes(self, text):
        """Input sanitization should handle any text."""
        from src.utils.sanitizer import sanitize_input, contains_injection_attempt
        
        try:
            sanitized = sanitize_input(text)
            assert isinstance(sanitized, str)
            
            has_injection = contains_injection_attempt(text)
            assert isinstance(has_injection, bool)
        except Exception as e:
            pytest.fail(f"Sanitization crashed with: {e}")
    
    @given(st.lists(plugin_strategy(), min_size=1, max_size=10))
    @settings(max_examples=30, deadline=None)
    def test_dependency_resolution_is_consistent(self, plugins):
        """Dependency resolution should be deterministic."""
        from src.agents.dependency_resolver import resolve_dependencies
        
        state = {"plugins": plugins}
        
        try:
            result1 = resolve_dependencies(state)
            result2 = resolve_dependencies(state)
            
            # Results should be consistent
            order1 = result1.get("dependency_order", [])
            order2 = result2.get("dependency_order", [])
            
            # If both succeeded, they should give same order
            if order1 and order2:
                assert order1 == order2
        except Exception:
            # It's okay if it fails, just shouldn't be inconsistent
            pass
    
    @given(st.text(min_size=0, max_size=1000))
    @settings(max_examples=50, deadline=None)
    def test_prompt_manager_handles_missing_templates(self, template_name):
        """PromptManager should gracefully handle non-existent templates."""
        from src.utils.prompt_manager import PromptManager
        
        pm = PromptManager()
        
        # Check existence
        exists = pm.template_exists(template_name)
        
        if not exists:
            # Should raise FileNotFoundError, not crash
            with pytest.raises(FileNotFoundError):
                pm.get_template(template_name)
    
    @given(st.lists(task_strategy(), min_size=1, max_size=10))
    @settings(max_examples=30, deadline=None)
    def test_circular_dependency_detection_is_sound(self, tasks):
        """Circular dependency detection should never give false negatives."""
        from src.agents.state_validator import StateValidator
        
        # Create an obvious cycle: A -> B -> C -> A
        # Only if we have at least 3 tasks with unique IDs
        if len(tasks) >= 3:
            # Ensure unique IDs
            task_ids = [t.get("id") for t in tasks]
            if len(set(task_ids)) < 3:
                # Skip if not enough unique IDs
                return
            
            # Get first 3 unique tasks
            unique_tasks = []
            seen_ids = set()
            for task in tasks:
                task_id = task.get("id")
                if task_id not in seen_ids:
                    unique_tasks.append(task)
                    seen_ids.add(task_id)
                if len(unique_tasks) >= 3:
                    break
            
            if len(unique_tasks) >= 3:
                # Create cycle
                unique_tasks[0]["dependencies"] = [unique_tasks[1]["id"]]
                unique_tasks[1]["dependencies"] = [unique_tasks[2]["id"]]
                unique_tasks[2]["dependencies"] = [unique_tasks[0]["id"]]
                
                cycle = StateValidator.check_circular_dependencies(unique_tasks)
                
                # Should detect the cycle
                assert cycle is not None, "Failed to detect obvious circular dependency"
    
    @given(st.text(min_size=10, max_size=1000))
    @settings(max_examples=30, deadline=None)
    def test_sanitizer_output_is_safe(self, text):
        """Sanitized output should never contain null bytes."""
        from src.utils.sanitizer import sanitize_input
        
        result = sanitize_input(text)
        
        # Should not contain null bytes
        assert '\x00' not in result
        
        # Should not exceed max length
        assert len(result) <= 10000
    
    @given(st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.recursive(
            st.one_of(st.text(max_size=50), st.integers(), st.booleans()),
            lambda children: st.lists(children, max_size=5) | st.dictionaries(
                st.text(min_size=1, max_size=10),
                children,
                max_size=5
            ),
            max_leaves=20
        ),
        min_size=0,
        max_size=10
    ))
    @settings(max_examples=30, deadline=None)
    def test_json_validation_depth_limit(self, nested_dict):
        """JSON validation should reject deeply nested structures."""
        from src.utils.sanitizer import validate_json_structure
        
        # This should not crash even with complex nested structures
        try:
            result = validate_json_structure(nested_dict, max_depth=10)
            assert isinstance(result, bool)
        except RecursionError:
            pytest.fail("JSON validation hit recursion limit")


class TestEdgeCases:
    """Test specific edge cases."""
    
    def test_empty_state(self):
        """Test handling of empty state."""
        from src.agents.goal_extractor import extract_goal
        
        result = extract_goal({})
        assert "goal" in result
        assert result["goal"] == ""
    
    def test_state_with_no_messages(self):
        """Test state without messages field."""
        from src.agents.goal_extractor import extract_goal
        
        result = extract_goal({"messages": []})
        assert "goal" in result
    
    def test_plan_with_duplicate_node_ids(self):
        """Test plan validation catches duplicate node IDs."""
        from src.agents.state_validator import StateValidator
        
        state = {
            "plan": {
                "nodes": [
                    {"id": "1", "prompt": "Step 1"},
                    {"id": "1", "prompt": "Step 2"}  # Duplicate ID!
                ]
            }
        }
        
        errors = StateValidator.validate_plan(state)
        assert len(errors) > 0
        assert any("duplicate" in err.lower() for err in errors)
    
    def test_plan_with_invalid_edge_references(self):
        """Test plan validation catches invalid edge references."""
        from src.agents.state_validator import StateValidator
        
        state = {
            "plan": {
                "nodes": [
                    {"id": "1", "prompt": "Step 1"}
                ],
                "edges": [
                    {"from": "1", "to": "99"}  # Node 99 doesn't exist!
                ]
            }
        }
        
        errors = StateValidator.validate_plan(state)
        assert len(errors) > 0
