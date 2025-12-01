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
    Validate plan completeness and correctness - SIMPLIFIED for Plan B.
    
    This node checks ONLY critical issues:
    1. Plan is not empty
    2. No circular dependencies in edges
    3. All referenced tools/plugins exist (if applicable)
    4. Basic structural integrity
    
    REMOVED (Plan B refactoring):
    - Subtask coverage checks (no more split_task dependency)
    - Confidence threshold checks (moved to format_output)
    
    Args:
        state: Application state with plan
        
    Returns:
        Updated state with validation results and potential replan trigger
    """
    plan = state.get("plan", {})
    plugins = state.get("plugins", [])
    has_system_elements = state.get("has_system_elements", False)
    
    if not plan:
        logger.error("No plan to validate")
        return {
            "plan_validation": {
                "valid": False,
                "errors": ["No plan generated"],
                "needs_replan": True
            },
            "replan": True
        }
    
    validation_errors = []
    warnings = []
    
    # 1. Validate basic plan structure
    nodes = plan.get("nodes", [])
    if len(nodes) == 0:
        validation_errors.append("Plan has no nodes")
    elif len(nodes) == 1 and not nodes[0].get("prompt") and not nodes[0].get("tool"):
        validation_errors.append("Plan has only one node with no prompt or tool")
    
    # 2. Check for circular dependencies in plan edges
    cycle_errors = _check_plan_cycles(plan)
    validation_errors.extend(cycle_errors)
    
    # 3. Validate plugin/tool references (only if system elements were provided)
    if has_system_elements and plugins:
        plugin_errors = _validate_plugin_references(plan, plugins)
        # Downgrade missing plugin errors to warnings in abstract mode
        if plugin_errors:
            warnings.extend(plugin_errors)
    
    # 4. Check for obviously broken plans
    if plan.get("goal", "").strip() == "":
        warnings.append("Plan has empty goal field")
    
    # Determine if replan is needed (only for CRITICAL errors)
    critical_errors = [
        err for err in validation_errors 
        if any(keyword in err.lower() for keyword in ["no nodes", "circular", "cycle"])
    ]
    
    needs_replan = len(critical_errors) > 0
    
    if validation_errors or warnings:
        logger.warning("plan_validation_issues", 
                      error_count=len(validation_errors),
                      warning_count=len(warnings))
    else:
        logger.info("plan_validation_passed", node_count=len(nodes))
    
    validation_result = {
        "valid": len(critical_errors) == 0,
        "errors": validation_errors,
        "warnings": warnings,
        "needs_replan": needs_replan
    }
    
    return {
        "plan_validation": validation_result,
        "replan": needs_replan
    }


def _validate_task_coverage(plan: Dict[str, Any], split_tasks: List[Dict[str, Any]]) -> List[str]:
    """
    DEPRECATED - Kept for compatibility but no longer used in Plan B.
    
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
        unaddressed_str = [str(item) for item in unaddressed]
        errors.append(f"Tasks not addressed in plan: {', '.join(unaddressed_str)}")
    
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
                cycle_str = [str(item) for item in cycle]
                errors.append(f"Circular dependency in plan: {' -> '.join(cycle_str)}")
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
