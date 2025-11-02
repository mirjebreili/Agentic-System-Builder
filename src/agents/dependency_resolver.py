"""
Dependency Resolver Node.

This node resolves dependencies between plugins and checks for conflicts.
"""

from typing import Any, Dict, List, Optional, Set
from src.utils.logger import get_logger, log_node_execution
from src.agents.state_validator import StateValidator

logger = get_logger(__name__)


@log_node_execution("resolve_dependencies")
def resolve_dependencies(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve plugin dependencies and check for conflicts.
    
    This node:
    1. Builds a dependency graph from plugins
    2. Checks for circular dependencies
    3. Topologically sorts plugins by dependency order
    4. Validates plugin compatibility
    
    Args:
        state: Application state with plugins
        
    Returns:
        Updated state with dependency information and sorted plugins
    """
    plugins = state.get("plugins", [])
    
    if not plugins:
        logger.info("No plugins to resolve dependencies for")
        return {
            "plugin_dependencies": {},
            "dependency_order": []
        }
    
    # Build dependency graph
    dependency_graph = _build_dependency_graph(plugins)
    
    logger.info("dependency_graph_built", 
                plugin_count=len(plugins),
                edge_count=sum(len(deps) for deps in dependency_graph.values()))
    
    # Check for circular dependencies
    cycle = StateValidator.check_circular_dependencies(plugins)
    
    if cycle:
        logger.error("circular_dependency_detected", cycle=cycle)
        from langchain_core.messages import AIMessage
        return {
            "messages": [AIMessage(content=f"❌ Circular dependency detected: {' -> '.join(cycle)}")],
            "has_system_elements": False,
            "dependency_errors": [f"Circular dependency: {' -> '.join(cycle)}"]
        }
    
    # Topologically sort plugins
    try:
        sorted_plugins = _topological_sort(dependency_graph, plugins)
        logger.info("plugins_sorted", order=[p.get("name") for p in sorted_plugins])
        
        return {
            "plugins": sorted_plugins,
            "plugin_dependencies": dependency_graph,
            "dependency_order": [p.get("name") for p in sorted_plugins]
        }
        
    except Exception as e:
        logger.error("dependency_resolution_failed", error=str(e))
        return {
            "dependency_errors": [str(e)],
            "plugin_dependencies": dependency_graph
        }


def _build_dependency_graph(plugins: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Build a dependency graph from plugins.
    
    Args:
        plugins: List of plugin dictionaries
        
    Returns:
        Dictionary mapping plugin names to their dependencies
    """
    graph = {}
    
    for plugin in plugins:
        name = plugin.get("name", "")
        dependencies = plugin.get("dependencies", [])
        
        # Ensure dependencies is a list
        if isinstance(dependencies, str):
            dependencies = [dependencies]
        elif not isinstance(dependencies, list):
            dependencies = []
        
        graph[name] = dependencies
    
    return graph


def _topological_sort(
    dependency_graph: Dict[str, List[str]],
    plugins: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Topologically sort plugins based on dependencies using Kahn's algorithm.
    
    Args:
        dependency_graph: Dependency graph
        plugins: List of plugin dictionaries
        
    Returns:
        List of plugins sorted by dependency order
        
    Raises:
        ValueError: If cycle is detected or graph is invalid
    """
    # Create a mapping from name to plugin
    plugin_map = {p.get("name"): p for p in plugins}
    
    # Build reverse graph (who depends on whom)
    in_degree = {name: 0 for name in dependency_graph}
    
    for name, deps in dependency_graph.items():
        for dep in deps:
            if dep in in_degree:
                in_degree[name] += 1
    
    # Find all nodes with in-degree 0 (no dependencies)
    queue = [name for name, degree in in_degree.items() if degree == 0]
    sorted_names = []
    
    while queue:
        # Remove node with no dependencies
        current = queue.pop(0)
        sorted_names.append(current)
        
        # For each plugin that depends on current, reduce in-degree
        for name, deps in dependency_graph.items():
            if current in deps:
                in_degree[name] -= 1
                if in_degree[name] == 0 and name not in sorted_names:
                    queue.append(name)
    
    # Check if all nodes were processed
    if len(sorted_names) != len(dependency_graph):
        raise ValueError("Cycle detected in dependency graph")
    
    # Return plugins in sorted order
    sorted_plugins = []
    for name in sorted_names:
        if name in plugin_map:
            sorted_plugins.append(plugin_map[name])
    
    return sorted_plugins


def check_plugin_compatibility(plugins: List[Dict[str, Any]]) -> List[str]:
    """
    Check for plugin compatibility issues.
    
    Args:
        plugins: List of plugins
        
    Returns:
        List of compatibility warnings/errors
    """
    warnings = []
    
    # Check for duplicate plugin names
    names = [p.get("name") for p in plugins]
    duplicates = [name for name in set(names) if names.count(name) > 1]
    
    if duplicates:
        warnings.append(f"Duplicate plugin names: {', '.join(duplicates)}")
    
    # Check for missing dependencies
    plugin_names = set(names)
    for plugin in plugins:
        name = plugin.get("name")
        deps = plugin.get("dependencies", [])
        
        if isinstance(deps, list):
            for dep in deps:
                if dep not in plugin_names:
                    warnings.append(f"Plugin '{name}' depends on missing plugin '{dep}'")
    
    return warnings


@log_node_execution("validate_plugin_dependencies")
def validate_plugin_dependencies(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate plugin dependencies without modifying the plugin list.
    
    This is useful as a validation step before planning.
    
    Args:
        state: Application state
        
    Returns:
        State with validation results
    """
    plugins = state.get("plugins", [])
    
    if not plugins:
        return {"dependency_validation": {"valid": True, "warnings": []}}
    
    # Check compatibility
    warnings = check_plugin_compatibility(plugins)
    
    # Check for cycles
    cycle = StateValidator.check_circular_dependencies(plugins)
    
    is_valid = not cycle and not any("missing plugin" in w for w in warnings)
    
    validation_result = {
        "valid": is_valid,
        "warnings": warnings,
        "has_cycle": cycle is not None,
        "cycle": cycle if cycle else None
    }
    
    logger.info("dependency_validation_complete", 
                valid=is_valid,
                warning_count=len(warnings))
    
    return {"dependency_validation": validation_result}
