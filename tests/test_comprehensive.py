"""Comprehensive regression tests for message handling."""

from langchain_core.messages import HumanMessage

from asb.agent.architecture_designer import design_architecture
from asb.agent.planner import plan_tot
from asb.scaffold.coordinator import scaffold_coordinator


def test_no_attribute_errors_in_codebase():
    """Ensure no AttributeError when handling messages throughout codebase."""

    test_state = {
        "messages": [HumanMessage(content="Test message")],
        "goal": "Test goal",
    }

    planner_result = plan_tot(test_state)
    assert not planner_result.get("error")

    arch_result = design_architecture(planner_result)
    assert not arch_result.get("error")


def test_scaffold_with_real_messages():
    """Test scaffold coordinator with real LangChain messages."""

    state = {
        "messages": [HumanMessage(content="Create a test app")],
        "goal": "Create a test app",
        "plan": {
            "goal": "Create a test app",
            "capabilities": [
                {
                    "name": "ui_scaffolding",
                    "description": "Select UI component libraries for rapid prototyping.",
                    "ecosystem_priority": ["npm", "pypi"],
                    "search_terms": ["react", "chakra ui", "material ui"],
                    "complexity": "medium",
                    "required": True,
                }
            ],
            "architecture_approach": "monolithic",
            "primary_language": "javascript",
            "integration_strategy": "Use React scaffolding with supporting Python utilities where needed.",
            "confidence": 0.6,
        },
        "architecture": {
            "nodes": [
                {"id": "plan"},
                {"id": "do"},
                {"id": "finish"},
            ]
        },
    }

    result = scaffold_coordinator(state)
    assert isinstance(result, dict)
