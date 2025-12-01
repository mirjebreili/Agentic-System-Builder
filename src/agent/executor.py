from __future__ import annotations
from typing import Any, Dict
from langchain_core.messages import HumanMessage
from llm.client import get_chat_model
from config.settings import get_settings

_DONE_TOKENS = ("DONE", "COMPLETED", "FINISHED", "ALL STEPS DONE")

def _is_done(text: str) -> bool:
    t = (text or "").upper()
    return any(tok in t for tok in _DONE_TOKENS)

def _normalize(s: str) -> str:
    return " ".join((s or "").strip().split()).lower()

def _reflexion_hint(last_output: str) -> str:
    return (
        "Reflect briefly (<=3 lines): what likely went wrong and ONE concrete next action. "
        "Start the final line with 'NEXT:'\n\n"
        f"LAST OUTPUT:\n{last_output}"
    )

def execute_plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    cfg = get_settings()
    llm = get_chat_model()

    plan = dict(state.get("plan") or {})
    messages = list(state.get("messages") or [])
    flags = dict(state.get("flags") or {"more_steps": True, "steps_done": False})
    metrics = dict(state.get("metrics") or {"fail_streak": 0, "prior_successes": 0, "prior_attempts": 0})

    nodes = {n["id"]: n for n in plan.get("nodes", [])}

    # 1) PLAN
    if "plan" in nodes:
        prompt = nodes["plan"].get("prompt") or "Split the task into 2–5 concrete steps."
        plan_out = llm.invoke([HumanMessage(prompt)]).content
        messages.append({"role": "assistant", "content": f"[plan]\n{plan_out}"})

    # 2) DO loop
    iters = 0
    max_iters = int(getattr(cfg, "exec_max_iters", 6))
    enable_reflexion = bool(getattr(cfg, "reflexion", True))
    reflexion_max = int(getattr(cfg, "reflexion_max_retries", 2))

    last_do = None
    while not flags.get("steps_done", False) and iters < max_iters:
        do_prompt = (nodes.get("do") or {}).get("prompt") or "Execute the next step. When everything is finished, write ONLY the word DONE."
        do_out = llm.invoke([HumanMessage(do_prompt)]).content
        messages.append({"role": "assistant", "content": f"[do]\n{do_out}"})
        iters += 1

        if _is_done(do_out):
            flags["steps_done"] = True; flags["more_steps"] = False; metrics["fail_streak"] = 0
            break

        if last_do is not None and _normalize(do_out) == _normalize(last_do):
            metrics["fail_streak"] = int(metrics.get("fail_streak", 0)) + 1
            if enable_reflexion and metrics["fail_streak"] <= reflexion_max:
                hint = _reflexion_hint(last_do)
                ref_out = llm.invoke([HumanMessage(hint)]).content
                messages.append({"role": "assistant", "content": f"[reflexion]\n{ref_out}"})
        else:
            metrics["fail_streak"] = 0

        last_do = do_out

    # 3) FINISH
    finish_prompt = (nodes.get("finish") or {}).get("prompt") or "Summarize the results briefly in 3 bullet points."
    finish_out = llm.invoke([HumanMessage(finish_prompt)]).content
    messages.append({"role": "assistant", "content": f"[finish]\n{finish_out}"})

    metrics["prior_attempts"] = int(metrics.get("prior_attempts", 0)) + 1
    if flags.get("steps_done"): metrics["prior_successes"] = int(metrics.get("prior_successes", 0)) + 1

    return {"messages": messages, "flags": flags, "metrics": metrics}
