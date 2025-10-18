from __future__ import annotations
from typing import Any, Dict
import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from agents.prompts_util import find_prompts_dir
from llm.client import get_chat_model

logger = logging.getLogger(__name__)

PROMPTS_DIR = find_prompts_dir()
SYSTEM_PROMPT = (PROMPTS_DIR / "format_system.jinja").read_text(encoding="utf-8")
USER_TMPL = (PROMPTS_DIR / "format_user.jinja").read_text(encoding="utf-8")


def format_plan_order(state: Dict[str, Any]) -> Dict[str, Any]:
    """Format the plan execution order into a readable text description using an LLM.
    
    This node takes the approved plan and uses an LLM to generate a well-formatted
    text summary that describes the order of execution based on the nodes and edges
    defined in the previous steps. It also includes all alternative plans with their
    confidence scores.
    
    Args:
        state: The current application state containing the plan
        
    Returns:
        Updated state with formatted plan order in messages
    """
    plan = state.get("plan", {})
    
    if not plan:
        logger.warning("No plan found in state to format")
        return state
    
    # Get the LLM
    llm = get_chat_model()
    
    # Get all plan candidates from debug info
    debug_info = state.get("debug", {})
    all_candidates = debug_info.get("plan_candidates", [])
    
    # Prepare the data structure for formatting
    format_data = {
        "selected_plan": plan,
        "all_plans": all_candidates
    }
    
    # Prepare the plan as JSON for the prompt
    plan_json = json.dumps(format_data, indent=2, ensure_ascii=False)
    
    # Create the user prompt by replacing the template variable
    user_prompt = USER_TMPL.replace("{{ plan_json }}", plan_json)
    
    # Create messages
    system_message = SystemMessage(content=SYSTEM_PROMPT)
    user_message = HumanMessage(content=user_prompt)
    
    logger.info(f"Invoking LLM to format plan with {len(plan.get('nodes', []))} nodes and {len(all_candidates)} alternative plans")
    
    # Invoke the LLM
    try:
        response = llm.invoke([system_message, user_message])
        formatted_text = response.content
        
        # Clean up any markdown code fences if the LLM added them
        formatted_text = formatted_text.replace("```", "").strip()
        
        logger.info(f"Successfully formatted plan order ({len(formatted_text)} characters)")
        
    except Exception as e:
        logger.error(f"Error formatting plan with LLM: {e}", exc_info=True)
        # Fallback to a simple format
        formatted_text = _fallback_format(plan, all_candidates)
    
    # Add to messages
    messages = list(state.get("messages", []))
    messages.append({
        "role": "assistant",
        "content": f"Plan execution order:\n\n{formatted_text}"
    })
    
    # Store formatted text in scratch for easy access
    scratch = dict(state.get("scratch") or {})
    scratch["formatted_plan_order"] = formatted_text
    
    return {
        "messages": messages,
        "scratch": scratch
    }


def _fallback_format(plan: Dict[str, Any], all_candidates: list = None) -> str:
    """Fallback formatting if LLM fails - simple numbered list format.
    
    Args:
        plan: The selected plan dictionary
        all_candidates: List of all plan candidates with confidence scores
        
    Returns:
        Simple numbered list with modules and confidence
    """
    output_lines = []
    
    if not all_candidates:
        # If no candidates available, just show the selected plan
        all_candidates = [{
            "confidence": plan.get("confidence", 0.0),
            "plan": plan
        }]
    
    # Sort candidates by confidence (highest first)
    sorted_candidates = sorted(all_candidates, key=lambda x: x.get("confidence", 0.0), reverse=True)
    
    selected_confidence = plan.get("confidence", 0.0)
    
    for plan_idx, candidate in enumerate(sorted_candidates, 1):
        conf = candidate.get("confidence", 0.0)
        candidate_plan = candidate.get("plan", {})
        nodes = candidate_plan.get("nodes", [])
        reasoning = candidate_plan.get("reasoning", "")
        
        # Check if this is the selected plan
        is_selected = abs(conf - selected_confidence) < 0.001  # Float comparison with tolerance
        
        # Plan header
        status = " [SELECTED]" if is_selected else ""
        output_lines.append(f"{plan_idx}.{status}")
        
        # List all modules/tools in order
        for node_idx, node in enumerate(nodes, 1):
            tool_name = node.get("tool") or node.get("id", f"node_{node_idx}")
            node_reasoning = node.get("reasoning", "")
            
            output_lines.append(f"   {plan_idx}.{node_idx} {tool_name}")
            
            # Show per-node reasoning if available
            if node_reasoning:
                output_lines.append(f"   {plan_idx}.{node_idx} Reason: {node_reasoning}")
        
        # Show overall plan reasoning if available
        if reasoning:
            output_lines.append(f"   Reasoning: {reasoning}")
        
        # Show confidence
        output_lines.append(f"   Confidence: {conf:.2f}")
        output_lines.append("")  # Empty line between plans
    
    return "\n".join(output_lines).strip()
