from __future__ import annotations
from typing import Any, Dict
from pydantic import ValidationError
from langgraph.types import interrupt
from .planner import Plan

def review_plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "type": "plan_review",
        "plan": state.get("plan", {}),
        "confidence": (state.get("plan") or {}).get("confidence", None),
        "message": "Approve plan or submit edits."
    }
    decision = interrupt(payload)

    action = str(decision.get("action", "approve")).lower()
    if action == "edit":
        edited = decision.get("plan")
        try:
            validated = Plan.model_validate(edited).model_dump(by_alias=True)
            state["plan"] = validated
            msgs = list(state.get("messages") or [])
            msgs.append({"role": "assistant", "content": "[hitl] Plan edited by human."})
            state["messages"] = msgs
        except ValidationError as e:
            msgs = list(state.get("messages") or [])
            msgs.append({"role": "assistant", "content": f"[hitl] Invalid edited plan; keeping original. Error: {e}"})
            state["messages"] = msgs
    else:
        msgs = list(state.get("messages") or [])
        msgs.append({"role": "assistant", "content": "[hitl] Plan approved."})
        state["messages"] = msgs

    return state
