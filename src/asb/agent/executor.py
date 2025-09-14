from __future__ import annotations
from typing import Any, Dict
from langchain_core.messages import HumanMessage
from asb.llm.client import get_chat_model
from asb_cfg.settings_v2 import get_settings

_DONE_TOKENS = ("DONE","COMPLETED","FINISHED","ALL STEPS DONE")

def _is_done(t: str) -> bool:
    u = (t or "").upper()
    return any(tok in u for tok in _DONE_TOKENS)

def _normalize(s: str) -> str:
    return " ".join((s or "").strip().split()).lower()

def _reflexion_hint(last_output: str) -> str:
    return ("Reflect in <=3 lines: what likely went wrong and ONE concrete next action.\n"
            "Start final line with 'NEXT:'\n\nLAST OUTPUT:\n" + last_output)

def execute_deep(state: Dict[str, Any]) -> Dict[str, Any]:
    cfg = get_settings()
    llm = get_chat_model()

    plan = dict(state.get("plan") or {})
    messages = list(state.get("messages") or [])
    flags = dict(state.get("flags") or {"more_steps": True, "steps_done": False})
    metrics = dict(state.get("metrics") or {"fail_streak": 0, "prior_attempts": 0, "prior_successes": 0})

    nodes = {n["id"]: n for n in plan.get("nodes", [])}

    # PLAN
    plan_out = llm.invoke([HumanMessage((nodes.get("plan") or {}).get("prompt") or "Split into steps.")]).content
    messages.append({"role":"assistant","content":f"[plan]\n{plan_out}"})

    # DO loop
    iters, max_iters = 0, int(cfg.exec_max_iters)
    enable_reflexion = bool(cfg.reflexion); reflexion_max = int(cfg.reflexion_max_retries)
    last_do, reflexed = None, 0
    trace: list[str] = [f"[plan] {plan_out}"]
    while not flags.get("steps_done") and iters < max_iters:
        do_prompt = (nodes.get("do") or {}).get("prompt") or "Do next step. When done, write ONLY DONE."
        out = llm.invoke([HumanMessage(do_prompt)]).content
        messages.append({"role":"assistant","content":f"[do]\n{out}"})
        trace.append(f"[do] {out}")
        iters += 1

        if _is_done(out):
            flags["steps_done"] = True; flags["more_steps"] = False; metrics["fail_streak"] = 0
            break

        if last_do is not None and _normalize(out) == _normalize(last_do):
            metrics["fail_streak"] = int(metrics.get("fail_streak", 0)) + 1
            if enable_reflexion and reflexed < reflexion_max:
                ref = _reflexion_hint(last_do or "")
                ref_out = llm.invoke([HumanMessage(ref)]).content
                messages.append({"role":"assistant","content":f"[reflexion]\n{ref_out}"})
                trace.append(f"[reflexion] {ref_out}")
                reflexed += 1
        else:
            metrics["fail_streak"] = 0

        last_do = out

    # FINISH
    fin_prompt = (nodes.get("finish") or {}).get("prompt") or "Summarize briefly."
    fin_out = llm.invoke([HumanMessage(fin_prompt)]).content
    messages.append({"role":"assistant","content":f"[finish]\n{fin_out}"})
    trace.append(f"[finish] {fin_out}")

    artifacts = {
        "plan_run_trace": trace,
        "sample_outputs": [fin_out][:1],
        "acceptance_tests": [{"name":"has summary","kind":"regex","expected":"(?i)summary|done"}],
    }
    passed = bool(flags.get("steps_done", False))

    metrics["prior_attempts"] = int(metrics.get("prior_attempts", 0)) + 1
    if passed:
        metrics["prior_successes"] = int(metrics.get("prior_successes", 0)) + 1

    return {"messages": messages, "flags": flags, "metrics": metrics, "artifacts": artifacts, "passed": passed}
