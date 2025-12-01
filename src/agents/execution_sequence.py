"""
Execution Sequence Generator
Translates PLAN structure into flat execution order showing actual Delta chain sequence.
"""

from typing import Dict, List, Any, Set
from agents.state import AppState
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_execution_sequence(state: AppState) -> Dict[str, Any]:
    """
    Generate flat execution sequence from PLAN structure.
    
    Converts the abstract PLAN (nodes + edges) into a flat execution order like:
    _SETSTREAM → httpBasedAtlasReadByKey → diskBasedAtlasKeyExtractor → _PARALLEL(...)
    
    Args:
        state: State containing the plan structure
        
    Returns:
        Updated state with execution_sequence list
    """
    logger.info("Generating execution sequence from PLAN")
    
    plan = state.get("plan", {})
    if not plan or "nodes" not in plan or "edges" not in plan:
        logger.warning("No valid plan structure found")
        return {"execution_sequence": []}
    
    nodes = plan["nodes"]
    edges = plan["edges"]
    
    # Build adjacency list
    graph = {node["id"]: [] for node in nodes}
    for edge in edges:
        graph[edge["from"]].append(edge["to"])
    
    # Find root nodes (nodes with no incoming edges)
    incoming = {node["id"]: 0 for node in nodes}
    for edge in edges:
        incoming[edge["to"]] += 1
    
    root_nodes = [node_id for node_id, count in incoming.items() if count == 0]
    
    if not root_nodes:
        logger.warning("No root nodes found in plan")
        return {"execution_sequence": []}
    
    # Build node lookup
    node_lookup = {node["id"]: node for node in nodes}
    
    # Generate sequence using topological sort with parallel detection
    sequence = []
    visited = set()
    
    def build_sequence(node_id: str, level: int = 0) -> None:
        """Recursively build execution sequence."""
        if node_id in visited:
            return
        
        visited.add(node_id)
        node = node_lookup[node_id]
        
        # Get node name from 'tool' field (or 'name' or 'id' as fallback)
        node_name = node.get("tool") or node.get("name") or node.get("id", "unknown")
        
        # Add current node
        sequence.append({
            "step": len(sequence) + 1,
            "name": node_name,
            "type": node.get("type", "plugin"),
            "level": level
        })
        
        # Get child nodes
        children = graph.get(node_id, [])
        
        if len(children) == 0:
            # Leaf node
            return
        elif len(children) == 1:
            # Sequential execution
            build_sequence(children[0], level)
        else:
            # Parallel execution
            sequence.append({
                "step": len(sequence) + 1,
                "name": "_PARALLEL",
                "type": "parallel",
                "level": level,
                "branches": len(children)
            })
            for child in children:
                build_sequence(child, level + 1)
    
    # Start from root nodes
    for root in root_nodes:
        build_sequence(root)
    
    # Format as readable sequence
    readable_sequence = format_readable_sequence(sequence)
    
    logger.info(f"Generated execution sequence with {len(sequence)} steps")
    logger.info(f"Readable sequence: {readable_sequence}")
    
    # Create user-facing message with the execution sequence
    from langchain_core.messages import AIMessage
    
    answer_message = AIMessage(
        content=f"""## Execution Sequence

{readable_sequence}

### Detailed Steps:
{chr(10).join([f"{i+1}. **{step['name']}** (type: {step['type']})" for i, step in enumerate(sequence)])}

This sequence represents the order in which your Delta plugins will be executed."""
    )
    
    return {
        "execution_sequence": sequence,
        "execution_sequence_readable": readable_sequence,
        "messages": [answer_message]
    }


def format_readable_sequence(sequence: List[Dict[str, Any]]) -> str:
    """
    Format execution sequence as readable string.
    
    Example output: "_SETSTREAM → httpBasedAtlasReadByKey → _PARALLEL(branch1, branch2)"
    """
    if not sequence:
        return ""
    
    parts = []
    parallel_branches = []
    in_parallel = False
    parallel_level = 0
    
    for item in sequence:
        name = item["name"]
        level = item.get("level", 0)
        
        if name == "_PARALLEL":
            in_parallel = True
            parallel_level = level
            parallel_branches = []
        elif in_parallel and level > parallel_level:
            # This is a branch of the parallel execution
            parallel_branches.append(name)
        else:
            # End of parallel section
            if parallel_branches:
                parts.append(f"_PARALLEL({', '.join(parallel_branches)})")
                parallel_branches = []
                in_parallel = False
            
            if name != "_PARALLEL":
                parts.append(name)
    
    # Handle case where parallel is at the end
    if parallel_branches:
        parts.append(f"_PARALLEL({', '.join(parallel_branches)})")
    
    return " → ".join(parts)


def should_continue_to_execution_sequence(state: AppState) -> str:
    """
    Router to decide if we should generate execution sequence.
    Always return 'generate_sequence' after format_plan_order.
    """
    return "generate_sequence"
