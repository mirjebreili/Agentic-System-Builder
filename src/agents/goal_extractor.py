"""
Goal Extractor Node.

This node extracts and sets the user's goal from the conversation,
addressing the issue where state["goal"] is defined but never populated.
"""

from typing import Any, Dict
from langchain_core.messages import AIMessage, HumanMessage
from src.utils.logger import get_logger, log_node_execution

logger = get_logger(__name__)


def extract_last_message_content(messages) -> str:
    """Extract content from the last human message."""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content
        elif isinstance(msg, dict) and msg.get("type") == "human":
            return msg.get("content", "")
    return ""


@log_node_execution("extract_goal")
def extract_goal(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the user's goal from the conversation.
    
    This node:
    1. Analyzes the user's message(s)
    2. Extracts the core goal/objective
    3. Stores it in state["goal"]
    
    Args:
        state: Application state
        
    Returns:
        Updated state with goal field populated
    """
    messages = state.get("messages", [])
    
    if not messages:
        logger.warning("No messages found in state")
        return {"goal": ""}
    
    # Extract the user's input
    user_input = extract_last_message_content(messages)
    
    if not user_input:
        logger.warning("Could not extract user input from messages")
        return {"goal": ""}
    
    # For simple cases, use the user input directly
    # For complex cases with plugins/system elements, extract just the goal
    
    # Check if this contains system elements (JSON structure)
    if "{" in user_input and "}" in user_input:
        # Try to extract goal from structured input
        try:
            import json
            # Find JSON and non-JSON parts
            lines = user_input.strip().split("\n")
            goal_lines = []
            
            for line in lines:
                # Skip lines that look like JSON
                if line.strip().startswith("{") or line.strip().startswith("}") or \
                   line.strip().startswith("[") or line.strip().startswith("]") or \
                   '"plugins"' in line or '"name"' in line or '"goal"' in line:
                    continue
                goal_lines.append(line)
            
            goal = "\n".join(goal_lines).strip()
            
            if not goal:
                # Fallback: use entire input
                goal = user_input
                
        except Exception as e:
            logger.warning(f"Failed to parse structured input: {e}")
            goal = user_input
    else:
        goal = user_input
    
    # Clean up the goal
    goal = goal.strip()
    
    logger.info("goal_extracted", goal_length=len(goal), has_system_elements=("{" in user_input))
    
    return {
        "goal": goal
    }


def extract_goal_with_llm(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract goal using LLM for more sophisticated analysis.
    
    This is a more advanced version that uses an LLM to extract
    the core goal when the input is complex or ambiguous.
    
    Args:
        state: Application state
        
    Returns:
        Updated state with goal field populated
    """
    messages = state.get("messages", [])
    
    if not messages:
        return {"goal": ""}
    
    user_input = extract_last_message_content(messages)
    
    if not user_input:
        return {"goal": ""}
    
    # Use LLM to extract goal
    from src.llm.client import get_chat_model
    llm = get_chat_model()
    
    prompt = f"""Given the following user input, extract ONLY the core goal or objective.
Ignore any system elements, plugin descriptions, or technical details.
Focus on what the user wants to accomplish.

User Input:
{user_input}

Core Goal (one clear sentence):"""
    
    try:
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])
        goal = response.content.strip()
        
        logger.info("goal_extracted_with_llm", goal_length=len(goal))
        
        return {"goal": goal}
        
    except Exception as e:
        logger.error(f"Failed to extract goal with LLM: {e}")
        # Fallback to simple extraction
        return extract_goal(state)
