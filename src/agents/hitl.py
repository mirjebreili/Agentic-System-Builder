from typing import Dict, Any, List
from langchain_core.messages import AIMessage

def hitl_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Presents the generated plan candidates to the user and prepares
    the state for interruption.

    This node formats the candidates into a user-friendly string,
    wraps it in an AIMessage, and adds it to the message history.
    The graph is configured to interrupt immediately after this node.
    """
    candidates = state.get("candidates", [])

    if not candidates:
        message_content = "I could not find any viable plans. Please provide more details or different plugins."
    else:
        message_lines = ["I have generated the following plan candidates. Please review them and respond with 'APPROVE <index>' or 'REVISE <notes>'.\n"]
        for i, cand in enumerate(candidates):
            confidence = cand.get('confidence', 0)
            confidence_str = f"{confidence:.1%}"

            line = (
                f"\n**Candidate #{i}** (Confidence: {confidence_str})\n"
                f"- **Plan:** `{' -> '.join(cand.get('plan', []))}`\n"
                f"- **Rationale:** {cand.get('rationale', 'N/A')}\n"
                f"- **Scores:** {cand.get('scores', {})}"
            )
            message_lines.append(line)
        message_content = "\n".join(message_lines)

    # The output of this node is an AIMessage that will be added to the state.
    # The graph will then interrupt, waiting for the user's HumanMessage.
    return {"messages": [AIMessage(content=message_content)]}