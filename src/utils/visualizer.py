"""
Plan Visualization utilities.

This module provides utilities for generating visual representations of plans,
particularly using Mermaid diagram syntax.
"""

from typing import Any, Dict, List
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_mermaid_diagram(plan: Dict[str, Any]) -> str:
    """
    Generate a Mermaid flowchart diagram from a plan.
    
    Args:
        plan: Execution plan with nodes and edges
        
    Returns:
        Mermaid diagram as string
        
    Example:
        >>> diagram = generate_mermaid_diagram(plan)
        >>> print(diagram)
    """
    nodes = plan.get("nodes", [])
    edges = plan.get("edges", [])
    
    if not nodes:
        return "graph TD\n  Empty[No nodes in plan]"
    
    lines = ["graph TD"]
    
    # Add nodes
    for node in nodes:
        node_id = str(node.get("id", ""))
        tool = node.get("tool", "")
        agent = node.get("agent", "")
        prompt = node.get("prompt", "")
        
        # Determine node label
        if tool:
            label = f"{tool}"
            shape = "([{id}: {label}])"  # Stadium shape for tools
        elif agent:
            label = f"{agent}"
            shape = "[{id}: {label}]"  # Rectangle for agents
        else:
            # Extract short prompt
            label = prompt[:30] + "..." if len(prompt) > 30 else prompt
            label = label.replace('"', "'")  # Escape quotes
            shape = "({id}: {label})"  # Rounded rectangle for prompts
        
        node_def = shape.format(id=node_id, label=label)
        lines.append(f"  {node_id}{node_def}")
    
    # Add edges
    for edge in edges:
        from_node = str(edge.get("from", ""))
        to_node = str(edge.get("to", ""))
        condition = edge.get("condition", "")
        
        if condition:
            # Labeled edge
            condition_label = condition.replace('"', "'")
            lines.append(f"  {from_node} -->|{condition_label}| {to_node}")
        else:
            # Simple edge
            lines.append(f"  {from_node} --> {to_node}")
    
    return "\n".join(lines)


def generate_ascii_tree(plan: Dict[str, Any]) -> str:
    """
    Generate an ASCII tree representation of a plan.
    
    Args:
        plan: Execution plan
        
    Returns:
        ASCII tree as string
    """
    nodes = plan.get("nodes", [])
    edges = plan.get("edges", [])
    
    if not nodes:
        return "Empty plan"
    
    # Build adjacency list
    children = {}
    for edge in edges:
        from_node = str(edge.get("from", ""))
        to_node = str(edge.get("to", ""))
        
        if from_node not in children:
            children[from_node] = []
        children[from_node].append(to_node)
    
    # Find root nodes (nodes with no incoming edges)
    all_nodes = {str(n.get("id")) for n in nodes}
    target_nodes = {str(e.get("to")) for e in edges}
    root_nodes = all_nodes - target_nodes
    
    if not root_nodes:
        # Circular or no edges - just list nodes
        root_nodes = [str(nodes[0].get("id"))]
    
    # Build node info map
    node_map = {str(n.get("id")): n for n in nodes}
    
    lines = []
    
    def format_node_info(node_id: str) -> str:
        """Format node information."""
        node = node_map.get(node_id, {})
        tool = node.get("tool", "")
        agent = node.get("agent", "")
        
        if tool:
            return f"[{node_id}] {tool}"
        elif agent:
            return f"[{node_id}] {agent}"
        else:
            prompt = node.get("prompt", "")
            short_prompt = prompt[:40] + "..." if len(prompt) > 40 else prompt
            return f"[{node_id}] {short_prompt}"
    
    def build_tree(node_id: str, prefix: str = "", is_last: bool = True):
        """Recursively build tree."""
        # Current node
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{format_node_info(node_id)}")
        
        # Children
        child_nodes = children.get(node_id, [])
        
        for i, child in enumerate(child_nodes):
            is_last_child = (i == len(child_nodes) - 1)
            extension = "    " if is_last else "│   "
            build_tree(child, prefix + extension, is_last_child)
    
    # Build tree from each root
    for i, root in enumerate(sorted(root_nodes)):
        if i > 0:
            lines.append("")
        lines.append(f"Root: {format_node_info(root)}")
        
        child_nodes = children.get(root, [])
        for j, child in enumerate(child_nodes):
            is_last_child = (j == len(child_nodes) - 1)
            build_tree(child, "", is_last_child)
    
    return "\n".join(lines)


def generate_plan_summary(plan: Dict[str, Any]) -> str:
    """
    Generate a human-readable plan summary.
    
    Args:
        plan: Execution plan
        
    Returns:
        Summary as string
    """
    nodes = plan.get("nodes", [])
    edges = plan.get("edges", [])
    confidence = plan.get("confidence", 0.0)
    reasoning = plan.get("reasoning", "")
    
    lines = [
        "📋 Plan Summary",
        "=" * 50,
        f"Nodes: {len(nodes)}",
        f"Edges: {len(edges)}",
        f"Confidence: {confidence:.2%}",
    ]
    
    if reasoning:
        lines.append(f"\nReasoning: {reasoning}")
    
    lines.append("\nNodes:")
    for i, node in enumerate(nodes, 1):
        node_id = node.get("id", "")
        tool = node.get("tool", "")
        agent = node.get("agent", "")
        prompt = node.get("prompt", "")
        
        if tool:
            lines.append(f"  {i}. [{node_id}] Tool: {tool}")
        elif agent:
            lines.append(f"  {i}. [{node_id}] Agent: {agent}")
        else:
            short_prompt = prompt[:60] + "..." if len(prompt) > 60 else prompt
            lines.append(f"  {i}. [{node_id}] Prompt: {short_prompt}")
    
    if edges:
        lines.append("\nExecution Flow:")
        for edge in edges:
            from_node = edge.get("from", "")
            to_node = edge.get("to", "")
            condition = edge.get("condition", "")
            
            if condition:
                lines.append(f"  {from_node} --[{condition}]--> {to_node}")
            else:
                lines.append(f"  {from_node} --> {to_node}")
    
    return "\n".join(lines)


def visualize_plan(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate multiple visualizations of the plan.
    
    Args:
        state: Application state with plan
        
    Returns:
        Updated state with visualization data
    """
    plan = state.get("plan", {})
    
    if not plan:
        logger.warning("No plan to visualize")
        return {
            "plan_visualizations": {
                "mermaid": "graph TD\n  Empty[No plan available]",
                "ascii": "No plan available",
                "summary": "No plan available"
            }
        }
    
    try:
        mermaid = generate_mermaid_diagram(plan)
        ascii_tree = generate_ascii_tree(plan)
        summary = generate_plan_summary(plan)
        
        logger.info("plan_visualizations_generated",
                   mermaid_lines=len(mermaid.split('\n')),
                   ascii_lines=len(ascii_tree.split('\n')))
        
        return {
            "plan_visualizations": {
                "mermaid": mermaid,
                "ascii": ascii_tree,
                "summary": summary
            }
        }
        
    except Exception as e:
        logger.error("plan_visualization_failed", error=str(e))
        return {
            "plan_visualizations": {
                "error": str(e)
            }
        }


def format_mermaid_for_markdown(mermaid: str) -> str:
    """
    Format Mermaid diagram for embedding in Markdown.
    
    Args:
        mermaid: Mermaid diagram string
        
    Returns:
        Markdown-formatted string
    """
    return f"```mermaid\n{mermaid}\n```"


def generate_html_visualization(plan: Dict[str, Any]) -> str:
    """
    Generate an HTML page with interactive Mermaid visualization.
    
    Args:
        plan: Execution plan
        
    Returns:
        HTML string
    """
    mermaid = generate_mermaid_diagram(plan)
    summary = generate_plan_summary(plan)
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Plan Visualization</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({{ startOnLoad: true }});</script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
        }}
        .diagram {{
            margin: 20px 0;
            padding: 20px;
            background-color: #fafafa;
            border-radius: 4px;
        }}
        .summary {{
            white-space: pre-wrap;
            font-family: monospace;
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 4px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Execution Plan Visualization</h1>
        
        <div class="diagram">
            <h2>Flowchart</h2>
            <div class="mermaid">
{mermaid}
            </div>
        </div>
        
        <div class="summary">
{summary}
        </div>
    </div>
</body>
</html>
"""
    
    return html
