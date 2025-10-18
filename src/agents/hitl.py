from __future__ import annotations
from typing import Any, Dict
import logging
from langgraph.types import interrupt

logger = logging.getLogger(__name__)


def review_plan(state: Dict[str, Any]) -> Dict[str, Any]:
    """Pause for a human-in-the-loop plan review.

    Execution is interrupted and resumes with a payload indicating how to
    proceed. The payload can be supplied as a dictionary, or as a shortcut
    string:

    - ``{"action": "approve", "plan": {...}}`` to accept the plan (optionally
      providing a modified plan)
    - ``{"action": "revise", "feedback": "..."}`` to request changes
    - ``"approve"`` or ``"revise"`` as shorthand strings

    """
    current_plan = state.get("plan", {})
    logger.info(f"REVIEW NODE INPUT: confidence={current_plan.get('confidence', 'N/A')}, nodes={len(current_plan.get('nodes', []))}, first_node={current_plan.get('nodes', [{}])[0].get('tool', 'N/A') if current_plan.get('nodes') else 'N/A'}")
    
    payload = {"plan": current_plan}
    resume = interrupt(payload)
    if isinstance(resume, str):
        resume = {"action": resume}
    action = (resume or {}).get("action")
    if action not in {"approve", "revise"}:
        raise ValueError(f"Unknown action '{action}'. Expected 'approve' or 'revise'.")
    if action == "approve":
        new_plan = (resume or {}).get("plan") or state.get("plan", {})
        state["plan"] = new_plan
        state["review"] = {"action": "approve"}
        state["replan"] = False
    else:  # action == "revise"
        state["review"] = {
            "action": "revise",
            "feedback": (resume or {}).get("feedback", ""),
        }
        state["replan"] = True
    return state
