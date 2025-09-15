from __future__ import annotations
from typing import Any, Dict
from langgraph.types import interrupt

def review_plan(state: Dict[str, Any]) -> Dict[str, Any]:
    # Pause here; UI/API should send back: {"action":"approve","plan":{...}} or {"action":"revise","feedback":"..."}
    payload = {"plan": state.get("plan", {})}
    resume = interrupt(payload)
    if isinstance(resume, str):
        resume = {"action": resume}
    action = (resume or {}).get("action")
    if action == "approve":
        new_plan = (resume or {}).get("plan") or state.get("plan", {})
        state["plan"] = new_plan
        state["review"] = {"action": "approve"}
        state["replan"] = False
    elif action == "revise":
        state["review"] = {
            "action": "revise",
            "feedback": (resume or {}).get("feedback", ""),
        }
        state["replan"] = True
    else:
        raise ValueError(f"Unknown action: {action}")
    return state
