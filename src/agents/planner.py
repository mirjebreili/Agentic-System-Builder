from __future__ import annotations
import json
import logging
import re
import copy
from typing import Any, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, ValidationError
from agents.prompts_util import find_prompts_dir
from llm.client import get_chat_model
from utils.message_utils import extract_last_message_content

class PlanNode(BaseModel):
    id: str
    prompt: str | None = None
    tool: str | None = None
    reasoning: str | None = None

class PlanEdge(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    if_: str | None = Field(None, alias="if")

class Plan(BaseModel):
    goal: str
    nodes: list[PlanNode]
    edges: list[PlanEdge]
    confidence: float | None = None
    reasoning: str | None = None

logger = logging.getLogger(__name__)

PROMPTS_DIR = find_prompts_dir()
SYSTEM_PROMPT = (PROMPTS_DIR / "plan_system.jinja").read_text(encoding="utf-8")
USER_TMPL = (PROMPTS_DIR / "plan_user.jinja").read_text(encoding="utf-8")

def _render_user_prompt(goal: str, constraints: str | None = None) -> str:
    txt = USER_TMPL.replace("{{ user_goal }}", goal)
    txt = txt.replace('{{ constraints | default("Keep it simple and actionable.") }}',
                      constraints or "Keep it simple and actionable.")
    
    # Handle the conditional template logic for Persian/plugin detection
    if "مجموع" in goal or "plugin" in goal.lower() or "price_" in goal:
        # Persian plugin task detected
        txt = txt.replace("{% if \"مجموع\" in user_goal or \"plugin\" in user_goal.lower() or \"price_\" in user_goal %}", "")
        txt = txt.replace("{% else %}", "<!-- ELSE_BLOCK -->")
        txt = txt.replace("{% endif %}", "")
        # Keep the plugin analysis section, remove the general section
        if "<!-- ELSE_BLOCK -->" in txt:
            txt = txt.split("<!-- ELSE_BLOCK -->")[0]
    else:
        # General task
        txt = txt.replace("{% if \"مجموع\" in user_goal or \"plugin\" in user_goal.lower() or \"price_\" in user_goal %}", "<!-- IF_BLOCK -->")
        txt = txt.replace("{% else %}", "")
        txt = txt.replace("{% endif %}", "")
        # Keep the general section, remove the plugin section
        if "<!-- IF_BLOCK -->" in txt:
            txt = txt.split("<!-- IF_BLOCK -->")[1] if "{% else %}" in USER_TMPL else txt
    
    return txt

_JSON_BLOCK = re.compile(r"```json\s*(.*?)```", re.S | re.I)
def _extract_json(text: str) -> str:
    m = _JSON_BLOCK.search(text or "")
    return m.group(1) if m else (text or "")

def plan_tot(state: Dict[str, Any]) -> Dict[str, Any]:
    """ToT: generate K=3 plans, judge, pick best; attach confidence."""
    llm = get_chat_model()
    messages = state.get("messages") or []
    user_goal = extract_last_message_content(messages, "Plan a tiny workflow.")
    K = 3

    sys = SystemMessage(SYSTEM_PROMPT + f"\nReturn {K} ALTERNATIVE JSON plans as a JSON array.")
    user = HumanMessage(_render_user_prompt(user_goal))
    resp = llm.invoke([sys, user]).content

    # Parse / repair
    try:
        cand = json.loads(_extract_json(resp))
    except Exception:
        fix = llm.invoke([SystemMessage("Output ONLY a JSON array of plan objects."), HumanMessage(resp)]).content
        logger.debug("Planner fix raw output: %s", fix)
        try:
            cand = json.loads(_extract_json(fix))
        except json.JSONDecodeError:
            logger.warning("Planner fix output is not valid JSON. Falling back to default plan.", exc_info=True)
            cand = []
    if isinstance(cand, dict) and "alternatives" in cand:
        cand = cand["alternatives"]

    valid: list[dict] = []
    for c in cand if isinstance(cand, list) else []:
        try:
            validated_plan = Plan.model_validate(c).model_dump(by_alias=True)
            valid.append(validated_plan)
            logger.info(
                "Validated plan: confidence=%s, nodes=%s, first_tool=%s",
                validated_plan.get("confidence", "N/A"),
                len(validated_plan.get("nodes", [])),
                validated_plan.get("nodes", [{}])[0].get("tool", "N/A") if validated_plan.get("nodes") else "N/A",
            )
        except ValidationError as exc:
            logger.warning("Plan validation failed: %s", exc)

    if not valid:
        logger.warning("Planner produced no valid plans; falling back to empty plan.")
        empty_plan: dict[str, Any] = {"goal": user_goal, "nodes": [], "edges": [], "confidence": 0.0}
        return {"plan": empty_plan, "messages": state.get("messages", []), "flags": {"more_steps": True, "steps_done": False}}

    # Select the plan with the highest confidence directly
    best = copy.deepcopy(max(valid, key=lambda p: float(p.get("confidence") or 0.0)))
    best_confidence = float(best.get("confidence") or 0.0)
    logger.info(
        "SELECTED BEST PLAN: confidence=%s, nodes=%s, first_tool=%s",
        best_confidence,
        len(best.get("nodes", [])),
        best.get("nodes", [{}])[0].get("tool", "N/A") if best.get("nodes") else "N/A",
    )

    # Store all candidate plans for downstream debugging/inspection
    debug = dict(state.get("debug") or {})
    debug_candidates = []
    for p in valid:
        debug_candidates.append(
            {
                "confidence": float(p.get("confidence") or 0.0),
                "first_tool": p.get("nodes", [{}])[0].get("tool", "N/A") if p.get("nodes") else "N/A",
                "node_count": len(p.get("nodes", [])),
                "plan": copy.deepcopy(p),
            }
        )
    debug["plan_candidates"] = debug_candidates
    state_debug_messages = debug

    msgs = list(state.get("messages") or [])
    msgs.append({"role": "assistant", "content": f"Selected ToT plan (score={best_confidence:.2f})."})

    try:
        from agents.executor import update_node_implementations

        update_node_implementations(best)
    except Exception:
        logger.debug("Unable to update node implementations for plan.", exc_info=True)

    logger.debug("Planner debug - selected plan: %s", best)

    return {
        "plan": copy.deepcopy(best),
        "messages": msgs,
        "flags": {"more_steps": True, "steps_done": False},
        "debug": state_debug_messages,
    }
