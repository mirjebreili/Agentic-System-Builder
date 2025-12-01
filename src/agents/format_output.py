"""
Format Output Node - Plan B Refactoring

This node merges:
- confidence.py (compute confidence score)
- formatter.py (format plan for output)

Provides a clean, single step for final output generation.
"""

from __future__ import annotations
from typing import Any, Dict
import json
from langchain_core.messages import SystemMessage, HumanMessage
from src.agents.prompts_util import find_prompts_dir
from src.llm.client import get_chat_model
from src.utils.logger import get_logger, log_node_execution

logger = get_logger(__name__)

PROMPTS_DIR = find_prompts_dir()
SYSTEM_PROMPT = (PROMPTS_DIR / "format_system.jinja").read_text(encoding="utf-8")
USER_TMPL = (PROMPTS_DIR / "format_user.jinja").read_text(encoding="utf-8")


def _compute_confidence(plan: Dict[str, Any]) -> float:
    """
    Compute confidence score for the plan.
    
    Combines:
    - LLM's self-assessed confidence (70% weight)
    - Structural complexity score (30% weight)
    
    Args:
        plan: The execution plan
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    # Get LLM's self-assessed confidence
    self_score = float(plan.get("confidence", 0.5) or 0.5)
    
    # Calculate structural score (simpler is better)
    nodes = plan.get("nodes", []) or []
    edges = plan.get("edges", []) or []
    steps = len(nodes)
    
    # Count branching points
    out_counts: Dict[str, int] = {}
    for e in edges:
        src = e.get("from") or e.get("from_")
        if src:
            out_counts[src] = out_counts.get(src, 0) + 1
    branches = sum(1 for count in out_counts.values() if count > 1)
    
    # Start with high score, penalize complexity
    structural = 0.9
    structural -= max(0, steps - 4) * 0.05  # Penalize plans with > 4 steps
    structural -= branches * 0.10  # Penalize branching
    structural = max(0.0, min(1.0, structural))
    
    # Weighted combination (70% LLM, 30% structural)
    confidence = 0.7 * self_score + 0.3 * structural
    confidence = max(0.0, min(1.0, confidence))
    
    logger.info(f"Confidence computed: LLM={self_score:.3f}, structural={structural:.3f}, final={confidence:.3f}")
    
    return confidence


@log_node_execution("format_output")
def format_output(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format the final plan output with confidence scoring.
    
    This node:
    1. Computes final confidence score
    2. Formats the plan into readable text using LLM
    3. Includes all alternative plans sorted by confidence
    4. Adds validation warnings if any
    
    Plan B Refactoring: Merges confidence computation and formatting.
    
    Args:
        state: The current application state containing the plan
        
    Returns:
        Updated state with formatted plan order in messages
    """
    plan = state.get("plan", {})
    
    if not plan:
        logger.warning("No plan found in state to format")
        return state
    
    # Compute final confidence score
    final_confidence = _compute_confidence(plan)
    
    # Update plan with final confidence
    plan["final_confidence"] = final_confidence
    
    # Get validation results
    validation = state.get("plan_validation", {})
    warnings = validation.get("warnings", [])
    
    # Get the LLM
    llm = get_chat_model()
    
    # Get all plan candidates from debug info
    debug_info = state.get("debug", {})
    all_candidates = debug_info.get("plan_candidates", [])
    
    # Sort candidates by confidence (highest to lowest)
    sorted_candidates = sorted(all_candidates, key=lambda x: x.get("confidence", 0.0), reverse=True)
    
    logger.info(f"Formatting plan with {len(sorted_candidates)} alternatives, final confidence: {final_confidence:.3f}")
    
    # Prepare the data structure for formatting
    format_data = {
        "selected_plan": plan,
        "final_confidence": final_confidence,
        "all_plans": sorted_candidates,
        "validation_warnings": warnings
    }
    
    # Prepare the plan as JSON for the prompt
    plan_json = json.dumps(format_data, indent=2, ensure_ascii=False)
    
    # Create the user prompt by replacing the template variable
    user_prompt = USER_TMPL.replace("{{ plan_json }}", plan_json)
    
    # Create messages
    system_message = SystemMessage(content=SYSTEM_PROMPT)
    user_message = HumanMessage(content=user_prompt)
    
    logger.info(f"Invoking LLM to format plan with {len(plan.get('nodes', []))} nodes")
    
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
        formatted_text = _fallback_format(plan, sorted_candidates, final_confidence, warnings)
    
    # Add to messages
    messages = list(state.get("messages", []))
    messages.append({
        "role": "assistant",
        "content": f"Plan execution order:\n\n{formatted_text}"
    })
    
    # Update debug info with final confidence
    debug_update = dict(state.get("debug", {}))
    debug_update["final_confidence"] = {
        "score": final_confidence,
        "llm_confidence": plan.get("confidence", 0.5),
        "structural_score": _compute_confidence.__wrapped__(plan) if hasattr(_compute_confidence, '__wrapped__') else 0.0
    }
    
    return {
        "messages": messages,
        "debug": debug_update
    }


def _fallback_format(
    plan: Dict[str, Any], 
    all_candidates: list = None, 
    confidence: float = 0.0,
    warnings: list = None
) -> str:
    """
    Fallback formatting if LLM fails - simple numbered list format.
    
    Args:
        plan: The selected plan dictionary
        all_candidates: List of all plan candidates with confidence scores
        confidence: Final confidence score
        warnings: List of validation warnings
        
    Returns:
        Simple numbered list with modules and confidence
    """
    nodes = plan.get("nodes", [])
    
    output = f"Selected Plan (Confidence: {confidence:.2%})\n"
    output += "=" * 50 + "\n\n"
    
    if warnings:
        output += "⚠️  Warnings:\n"
        for warning in warnings:
            output += f"   - {warning}\n"
        output += "\n"
    
    output += "Execution Steps:\n"
    for i, node in enumerate(nodes, 1):
        tool = node.get("tool", "N/A")
        prompt = node.get("prompt", "")
        output += f"{i}. {tool}"
        if prompt:
            output += f": {prompt[:60]}{'...' if len(prompt) > 60 else ''}"
        output += "\n"
    
    if all_candidates and len(all_candidates) > 1:
        output += f"\n\nAlternative Plans ({len(all_candidates) - 1}):\n"
        for i, candidate in enumerate(all_candidates[1:], 1):
            cand_conf = candidate.get("confidence", 0.0)
            cand_nodes = candidate.get("node_count", 0)
            output += f"  {i}. Confidence: {cand_conf:.2%}, Steps: {cand_nodes}\n"
    
    return output
