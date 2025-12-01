"""
Execution Sequence Generator Node.

This node translates the abstract PLAN (nodes/edges) into the actual
Delta plugin execution sequence with proper _SETSTREAM and _PARALLEL handling.
"""

from typing import Any, Dict, List
from src.utils.logger import get_logger, log_node_execution

logger = get_logger(__name__)


@log_node_execution("generate_execution_sequence")
def generate_execution_sequence(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate the actual execution sequence from the plan structure.
    
    Translates abstract plan nodes into concrete Delta plugin chain:
    - Identifies which nodes need _SETSTREAM
    - Resolves _PARALLEL blocks and their internal chains
    - Produces a flat or nested execution order
    
    Args:
        state: Application state with plan and system_elements
        
    Returns:
        Updated state with execution_sequence array
    """
    plan = state.get("plan", {})
    system_elements = state.get("system_elements", [])
    
    if not plan or not plan.get("nodes"):
        logger.warning("No plan found to generate execution sequence")
        return {"execution_sequence": []}
    
    nodes = plan.get("nodes", [])
    edges = plan.get("edges", [])
    
    logger.info(f"Generating execution sequence from {len(nodes)} nodes, {len(edges)} edges")
    
    # Build adjacency list
    graph = _build_adjacency_list(nodes, edges)
    
    # Find entry point (node with no incoming edges)
    entry = _find_entry_node(nodes, edges)
    
    if not entry:
        logger.error("No entry point found in plan")
        return {"execution_sequence": []}
    
    # Traverse and build execution sequence
    sequence = _traverse_and_build_sequence(entry, graph, nodes, system_elements)
    
    logger.info(f"Generated execution sequence with {len(sequence)} steps")
    logger.info(f"Execution sequence: {sequence}")
    
    return {
        "execution_sequence": sequence,
        "execution_sequence_text": _format_sequence_text(sequence)
    }


def _build_adjacency_list(nodes: List[Dict], edges: List[Dict]) -> Dict[str, str]:
    """Build adjacency list from edges."""
    graph = {}
    for edge in edges:
        graph[edge["from"]] = edge.get("to")
    return graph


def _find_entry_node(nodes: List[Dict], edges: List[Dict]) -> str:
    """Find the node with no incoming edges (entry point)."""
    incoming = {edge["to"] for edge in edges}
    node_ids = {node["id"] for node in nodes}
    
    entry_nodes = node_ids - incoming
    
    if not entry_nodes:
        return nodes[0]["id"] if nodes else None
    
    return list(entry_nodes)[0]


def _traverse_and_build_sequence(
    node_id: str,
    graph: Dict[str, str],
    nodes: List[Dict],
    system_elements: List[str]
) -> List[str]:
    """
    Traverse the plan graph and build execution sequence.
    
    Args:
        node_id: Starting node ID
        graph: Adjacency list
        nodes: List of all nodes
        system_elements: Available plugins
        
    Returns:
        Ordered list of plugin names representing execution sequence
    """
    sequence = []
    visited = set()
    
    current = node_id
    
    while current and current not in visited:
        visited.add(current)
        
        # Find node details
        node = next((n for n in nodes if n["id"] == current), None)
        if not node:
            break
        
        # Map node ID to actual plugin
        plugin_name = _map_node_to_plugin(node, system_elements)
        
        if plugin_name:
            sequence.append(plugin_name)
        
        # Move to next node
        current = graph.get(current)
    
    return sequence


def _map_node_to_plugin(node: Dict, system_elements: List[str]) -> str:
    """
    Map a plan node to its corresponding Delta plugin.
    
    Logic:
    - initial_key_input → _SETSTREAM
    - build_initial_request → memBasedAtlasRequestStringBuilder  
    - send_initial_read → httpBasedAtlasReadByKey
    - key_extractor → diskBasedAtlasKeyExtractor
    - parallel_fetch → _PARALLEL(...) 
    
    Args:
        node: Plan node dictionary
        system_elements: List of available plugins
        
    Returns:
        Plugin name or special operator
    """
    node_id = node.get("id", "")
    prompt = node.get("prompt", "").lower()
    
    # Mapping rules based on node ID and prompt keywords
    mappings = {
        "initial_key_input": "_SETSTREAM",
        "set_initial_input": "_SETSTREAM",
        "build_initial_request": "@partDeltaPlugin/memBasedAtlasRequestStringBuilder",
        "send_initial_read": "@partDeltaPlugin/httpBasedAtlasReadByKey",
        "key_extractor": "@partDeltaPlugin/diskBasedAtlasKeyExtractor",
        "extract_keys": "@partDeltaPlugin/diskBasedAtlasKeyExtractor",
        "parallel_fetch": "_PARALLEL",
        "parallel_per_key_fetch": "_PARALLEL",
        "parallel_per_key": "_PARALLEL",
        "records_aggregator": None,  # Aggregation is implicit in _PARALLEL output
        "aggregate_all": None,
        "final_aggregator": None
    }
    
    # Direct mapping
    if node_id in mappings:
        plugin = mappings[node_id]
        
        # Special handling for _PARALLEL
        if plugin == "_PARALLEL":
            # Extract inner chain from prompt
            inner_chain = _extract_parallel_chain(prompt, system_elements)
            if inner_chain:
                return f"_PARALLEL({' → '.join(inner_chain)})"
            return "_PARALLEL"
        
        return plugin
    
    # Fallback: try to match keywords in prompt
    if "membasedatlasrequestbuilder" in prompt or "request builder" in prompt:
        return "@partDeltaPlugin/memBasedAtlasRequestStringBuilder"
    
    if "httpbasedatlasreadbykey" in prompt or "readbykey" in prompt:
        return "@partDeltaPlugin/httpBasedAtlasReadByKey"
    
    if "diskbasedatlaskeyextractor" in prompt or "key extractor" in prompt:
        return "@partDeltaPlugin/diskBasedAtlasKeyExtractor"
    
    if "parallel" in prompt:
        inner_chain = _extract_parallel_chain(prompt, system_elements)
        if inner_chain:
            return f"_PARALLEL({' → '.join(inner_chain)})"
        return "_PARALLEL"
    
    logger.warning(f"Could not map node {node_id} to plugin")
    return None


def _extract_parallel_chain(prompt: str, system_elements: List[str]) -> List[str]:
    """
    Extract the inner chain for _PARALLEL from the prompt.
    
    Example prompt:
    "Run _PARALLEL with chain [_SETSTREAM, memBasedAtlasRequestStringBuilder, httpBasedAtlasReadByKey]"
    
    Returns:
    ["_SETSTREAM", "@partDeltaPlugin/memBasedAtlasRequestStringBuilder", "@partDeltaPlugin/httpBasedAtlasReadByKey"]
    """
    # Look for chain keywords
    if "chain" not in prompt.lower():
        return []
    
    # Extract plugins mentioned in prompt
    chain = []
    
    if "_setstream" in prompt.lower():
        chain.append("_SETSTREAM")
    
    if "membasedatlasrequestbuilder" in prompt.lower():
        chain.append("@partDeltaPlugin/memBasedAtlasRequestStringBuilder")
    
    if "httpbasedatlasreadbykey" in prompt.lower():
        chain.append("@partDeltaPlugin/httpBasedAtlasReadByKey")
    
    return chain


def _format_sequence_text(sequence: List[str]) -> str:
    """Format the execution sequence as readable text."""
    if not sequence:
        return "(empty sequence)"
    
    # Join with arrow
    return " → ".join(sequence)
