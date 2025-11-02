"""
Plan Validator Node.

This node validates plan completeness and correctness after planning.
"""

from typing import Any, Dict, List, Set
from langchain_core.messages import AIMessage
from src.utils.logger import get_logger, log_node_execution
from src.agents.state_validator import StateValidator

logger = get_logger(__name__)


@log_node_execution("validate_plan")
def validate_plan(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate plan completeness and correctness.
    
    This node checks:
    1. All subtasks from split_task are addressed
    2. No circular dependencies in edges
    3. All referenced tools/plugins exist
    4. Confidence thresholds are met
    5. Plan structure is valid
    
    Args:
        state: Application state with plan
        
    Returns:
        Updated state with validation results and potential replan trigger
    """
    plan = state.get("plan", {})
    split_tasks = state.get("split_tasks", [])
    plugins = state.get("plugins", [])
    
    if not plan:
        logger.error("No plan to validate")
        return {
            "replan": True,
            "review": {
                "action": "revise",
                "feedback": "No plan generated"
            }
        }
    
    validation_errors = []
    
    # 1. Validate plan structure
    structure_errors = StateValidator.validate_plan(state)
    if structure_errors:
        validation_errors.extend(structure_errors)
    
    # 2. Check if all subtasks are addressed
    if split_tasks:
        task_errors = _validate_task_coverage(plan, split_tasks)
        validation_errors.extend(task_errors)
    
    # 3. Check for circular dependencies in plan edges
    cycle_errors = _check_plan_cycles(plan)
    validation_errors.extend(cycle_errors)
    
    # 4. Validate plugin/tool references
    if plugins:
        plugin_errors = _validate_plugin_references(plan, plugins)
        validation_errors.extend(plugin_errors)
    
    # 5. Check confidence threshold
    confidence = plan.get("confidence", 0.0)
    if confidence < 0.5:
        validation_errors.append(f"Low confidence score: {confidence:.2f} (minimum 0.5 recommended)")
    
    # 6. Check for empty or trivial plans
    nodes = plan.get("nodes", [])
    if len(nodes) == 0:
        validation_errors.append("Plan has no nodes")
    elif len(nodes) == 1 and not nodes[0].get("prompt"):
        validation_errors.append("Plan has only one node with no prompt")
    
    # Determine if replan is needed
    if validation_errors:
        logger.warning("plan_validation_failed", error_count=len(validation_errors))
        
        feedback = "Plan validation failed:\n" + "\n".join(f"  • {err}" for err in validation_errors)
        
        # Only trigger replan for critical errors
        critical_errors = [
            "circular dependency",
            "no nodes",
            "missing plugin",
            "task not addressed"
        ]
        
        has_critical_error = any(
            any(ce in err.lower() for ce in critical_errors)
            for err in validation_errors
        )
        
        return {
            "replan": has_critical_error,
            "plan_validation": {
                "valid": False,
                "errors": validation_errors,
                "warnings": [e for e in validation_errors if not has_critical_error]
            },
            "review": {
                "action": "revise" if has_critical_error else "review",
                "feedback": feedback
            }
        }
    
    logger.info("plan_validation_passed", node_count=len(nodes), confidence=confidence)
    
    return {
        "plan_validation": {
            "valid": True,
            "errors": [],
            "warnings": []
        }
    }


def _validate_task_coverage(plan: Dict[str, Any], split_tasks: List[Dict[str, Any]]) -> List[str]:
    """
    Check if all subtasks are addressed in the plan.
    
    Args:
        plan: Execution plan
        split_tasks: List of subtasks
        
    Returns:
        List of validation errors
    """
    errors = []
    nodes = plan.get("nodes", [])
    
    # Extract task IDs mentioned in node prompts
    addressed_tasks = set()
    
    for node in nodes:
        prompt = node.get("prompt", "").lower()
        
        for task in split_tasks:
            task_id = task.get("id", "")
            task_desc = task.get("description", "").lower()
            
            # Check if task is mentioned in prompt
            if str(task_id).lower() in prompt or any(
                word in prompt for word in task_desc.split()[:5]  # Check first 5 words
            ):
                addressed_tasks.add(task_id)
    
    # Find unaddressed tasks
    all_task_ids = {task.get("id") for task in split_tasks}
    unaddressed = all_task_ids - addressed_tasks
    
    if unaddressed:
        errors.append(f"Tasks not addressed in plan: {', '.join(unaddressed)}")
    
    return errors


def _check_plan_cycles(plan: Dict[str, Any]) -> List[str]:
    """
    Check for circular dependencies in plan edges.
    
    Args:
        plan: Execution plan
        
    Returns:
        List of validation errors
    """
    errors = []
    edges = plan.get("edges", [])
    
    if not edges:
        return errors
    
    # Build adjacency list
    graph = {}
    for edge in edges:
        from_node = str(edge.get("from", ""))
        to_node = str(edge.get("to", ""))
        
        if from_node not in graph:
            graph[from_node] = []
        graph[from_node].append(to_node)
    
    # DFS to detect cycles
    visited = set()
    rec_stack = set()
    
    def has_cycle(node: str, path: List[str]) -> bool:
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if has_cycle(neighbor, path):
                    return True
            elif neighbor in rec_stack:
                # Found cycle
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                errors.append(f"Circular dependency in plan: {' -> '.join(cycle)}")
                return True
        
        path.pop()
        rec_stack.remove(node)
        return False
    
    for node in graph:
        if node not in visited:
            has_cycle(node, [])
    
    return errors


def _validate_plugin_references(plan: Dict[str, Any], plugins: List[Dict[str, Any]]) -> List[str]:
    """
    Validate that all referenced plugins/tools exist.
    
    Args:
        plan: Execution plan
        plugins: Available plugins
        
    Returns:
        List of validation errors
    """
    errors = []
    nodes = plan.get("nodes", [])
    
    # Extract plugin names
    plugin_names = {p.get("name", "").lower() for p in plugins}
    plugin_tools = set()
    
    for plugin in plugins:
        tools = plugin.get("tools", [])
        if isinstance(tools, list):
            plugin_tools.update(t.lower() for t in tools)
    
    # Check each node
    for i, node in enumerate(nodes):
        tool = node.get("tool", "")
        agent = node.get("agent", "")
        
        if tool:
            tool_lower = tool.lower()
            # Check if tool exists in plugins
            if tool_lower not in plugin_tools and tool_lower not in plugin_names:
                errors.append(f"Node {i}: References unknown tool '{tool}'")
        
        if agent:
            agent_lower = agent.lower()
            # Check if agent/plugin exists
            if agent_lower not in plugin_names:
                # This might be a built-in agent, so just warn
                logger.debug(f"Node {i}: References agent '{agent}' (might be built-in)")
    
    return errors


def generate_validation_report(state: Dict[str, Any]) -> str:
    """
    Generate a human-readable validation report.
    
    Args:
        state: Application state with plan_validation
        
    Returns:
        Formatted validation report
    """
    validation = state.get("plan_validation", {})
    
    if not validation:
        return "⚠️ No validation results available"
    
    if validation.get("valid"):
        return "✅ Plan validation passed"
    
    lines = ["❌ Plan Validation Failed:"]
    
    errors = validation.get("errors", [])
    if errors:
        lines.append("\nErrors:")
        for error in errors:
            lines.append(f"  • {error}")
    
    warnings = validation.get("warnings", [])
    if warnings:
        lines.append("\nWarnings:")
        for warning in warnings:
            lines.append(f"  ⚠️ {warning}")
    
    return "\n".join(lines)
