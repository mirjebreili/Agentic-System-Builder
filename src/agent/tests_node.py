from __future__ import annotations
from typing import Any, Dict
from .planner import plan_tot, Plan
from .executor import execute_deep

def test_agents(state: Dict[str, Any]) -> Dict[str, Any]:
    # Planner dry test
    pstate = {"messages":[{"role":"user","content":"Summarize 3 expenses by category."}]}
    pstate = plan_tot(pstate)
    ok_planner, reason_p = True, ""
    try:
        Plan.model_validate(pstate["plan"])
        nodes = pstate["plan"]["nodes"];
        edges = pstate["plan"]["edges"]
        ids = [n["id"] for n in nodes]
        assert ids == ["plan","do","finish"]
        assert any(e["from"]=="do" and e["to"]=="do" for e in edges)
        assert any(e["from"]=="do" and e["to"]=="finish" for e in edges)
        assert float(pstate["plan"].get("confidence",0.0)) >= 0.3
    except Exception as e:
        ok_planner, reason_p = False, f"planner test failed: {e}"

    # Executor dry test
    ok_exec, reason_e, steps_used = True, "", 0
    try:
        demo = {"plan": {"goal":"demo",
                         "nodes":[
                           {"id":"plan","type":"llm","prompt":"List two steps then DONE."},
                           {"id":"do","type":"llm","prompt":"Say STEP 1, then STEP 2, then DONE."},
                           {"id":"finish","type":"llm","prompt":"Summarize in one line."}],
                         "edges":[
                           {"from":"plan","to":"do"},
                           {"from":"do","to":"do","if":"more_steps"},
                           {"from":"do","to":"finish","if":"steps_done"}]},
                "messages":[],"flags":{"more_steps":True,"steps_done":False}}
        est = execute_deep(demo)
        steps_used = sum(1 for m in est["messages"] if m.get("content","").startswith("[do]"))
        assert any(m.get("content","").startswith("[finish]") for m in est["messages"])
    except Exception as e:
        ok_exec, reason_e = False, f"executor test failed: {e}"

    overall = ok_planner and ok_exec
    out = {"planner_ok": ok_planner, "planner_reason": reason_p,
           "executor_ok": ok_exec, "executor_reason": reason_e,
           "steps_used": steps_used, "overall_ok": overall}

    # Only trigger replan ONCE. If we've failed before, don't set the flag again.
    already_failed = state.get("tests", {}).get("overall_ok") is False
    if not overall and not already_failed:
        state["replan"] = True

    state["tests"] = out
    return state
