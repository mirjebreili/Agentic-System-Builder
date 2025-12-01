from __future__ import annotations
import json
from typing import Any, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from llm.client import get_chat_model
from config.settings import get_settings
from .planner import Plan, _extract_json, SYSTEM_PROMPT, _render_user_prompt

def tot_planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Always run ToT for planning: generate K alternative plans and select one via a quick judge.
    Then store the best plan in state['plan'].
    """
    cfg = get_settings()
    k = max(2, int(getattr(cfg, "tot_branches", 5)))
    llm = get_chat_model()

    user_text = (state.get("messages") or [{}])[-1].get("content", "Plan a tiny workflow.")
    alt_system = SYSTEM_PROMPT + f"\nProduce EXACTLY {k} ALTERNATIVE JSON plans that all follow the same schema, but differ in step wording or prompts. Return a JSON array of objects."
    sys = SystemMessage(alt_system)
    user = HumanMessage(_render_user_prompt(user_text, "Provide diverse, concrete plans."))

    resp = llm.invoke([sys, user])
    raw = _extract_json(resp.content)

    try:
        candidates = json.loads(raw)
    except Exception:
        fix = llm.invoke([SystemMessage("Output ONLY a JSON array of plan objects, no prose."), HumanMessage(raw)])
        candidates = json.loads(_extract_json(fix.content))

    if isinstance(candidates, dict) and "alternatives" in candidates:
        candidates = candidates["alternatives"]

    valid_plans: list[dict] = []
    for c in candidates if isinstance(candidates, list) else []:
        try:
            valid_plans.append(Plan.model_validate(c).model_dump(by_alias=True))
        except Exception:
            continue

    if not valid_plans:
        msgs = list(state.get("messages") or [])
        msgs.append({"role": "assistant", "content": "[tot] No valid candidates; using seed plan."})
        state["messages"] = msgs
        return state

    judge = get_chat_model()
    scored: list[tuple[float, str, dict]] = []
    for p in valid_plans:
        judge_prompt = ("Score this plan in [0,1] using: Clarity, Coverage, Simplicity. Return ONLY JSON: {\"score\":0..1, \"reason\":\"...\"}")
        critique = judge.invoke([SystemMessage(judge_prompt), HumanMessage(json.dumps({"user_goal": user_text, "plan": p}, ensure_ascii=False))]).content
        try:
            js = json.loads(_extract_json(critique))
            score = float(js.get("score", 0.0)); reason = str(js.get("reason", ""))
        except Exception:
            score, reason = 0.5, "fallback"
        scored.append((score, reason, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_reason, best_plan = scored[0]
    best_plan["confidence"] = float(min(1.0, max(0.0, best_score)))

    msgs = list(state.get("messages") or [])
    msgs.append({"role": "assistant", "content": f"[tot] Selected plan (score={best_score:.2f}) — {best_reason}"})
    state["plan"] = best_plan
    state["messages"] = msgs
    dbg = dict(state.get("debug") or {})
    dbg["tot"] = {"candidates": len(valid_plans), "selected_score": round(best_score, 3)}
    state["debug"] = dbg
    return state
