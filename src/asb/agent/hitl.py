from __future__ import annotations
from typing import Any, Dict
from langgraph.types import interrupt


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
    payload = {"plan": state.get("plan", {})}
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
