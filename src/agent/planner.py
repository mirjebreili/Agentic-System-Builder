from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import SystemMessage, HumanMessage
from llm.client import get_chat_model

class PlanNode(BaseModel):
    id: str
    type: str
    prompt: str | None = None
    tool: str | None = None

class PlanEdge(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    if_: str | None = Field(None, alias="if")

class Plan(BaseModel):
    goal: str
    nodes: list[PlanNode]
    edges: list[PlanEdge]
    confidence: float | None = None

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "plan_system.jinja").read_text(encoding="utf-8")
USER_TMPL = (PROMPTS_DIR / "plan_user.jinja").read_text(encoding="utf-8")

_JSON_BLOCK = re.compile(r"```json\s*(.*?)```", re.S | re.I)

def _render_user_prompt(user_goal: str, constraints: str | None = None) -> str:
    text = USER_TMPL.replace("{{ user_goal }}", user_goal)
    text = text.replace('{{ constraints | default("Keep it simple and low-cost.") }}', constraints or "Keep it simple and low-cost.")
    return text

def _extract_json(text: str) -> str:
    m = _JSON_BLOCK.search(text or "")
    return m.group(1) if m else (text or "")

def plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    llm = get_chat_model()
    user_goal = "Plan a tiny workflow."
    if state.get("messages"):
        user_goal = state["messages"][-1].get("content") or user_goal

    sys = SystemMessage(SYSTEM_PROMPT)
    user = HumanMessage(_render_user_prompt(user_goal))

    resp = llm.invoke([sys, user])
    raw = _extract_json(resp.content)

    try:
        plan_obj = Plan.model_validate_json(raw).model_dump(by_alias=True)
    except ValidationError:
        fix = llm.invoke([SystemMessage("Your previous JSON was invalid. Output ONLY corrected JSON, no prose."), HumanMessage(raw)])
        plan_obj = Plan.model_validate_json(_extract_json(fix.content)).model_dump(by_alias=True)

    out_msgs = list(state.get("messages") or [])
    out_msgs.append({"role": "assistant", "content": "Planned workflow (seed)."})
    new_state: Dict[str, Any] = {"plan": plan_obj, "messages": out_msgs, "flags": {"more_steps": True, "steps_done": False}}
    if "metrics" in state: new_state["metrics"] = state["metrics"]
    if "debug" in state: new_state["debug"] = state["debug"]
    return new_state
