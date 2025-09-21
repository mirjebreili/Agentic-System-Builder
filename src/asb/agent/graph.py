from __future__ import annotations
from typing import Dict, Any
import os
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
import config.settings as s
from config.settings import SETTINGS_UID
from asb.agent.state import AppState
from asb.agent.planner import plan_tot
from asb.agent.confidence import compute_plan_confidence
from asb.agent.hitl import review_plan
from asb.agent.tests_node import test_agents
from asb.agent.executor import execute_deep
from asb.agent.scaffold import scaffold_project
from asb.agent.code_validator import code_validator_node
from asb.agent.code_fixer import code_fixer_node
from asb.agent.sandbox import comprehensive_sandbox_test as sandbox_smoke
from asb.agent.report import report

print("### USING settings_v2 FROM:", s.__file__)
print("### SETTINGS UID:", SETTINGS_UID)


def running_on_langgraph_api() -> bool:
    """Return ``True`` when executing within a LangGraph-managed runtime."""

    langgraph_env = os.environ.get("LANGGRAPH_ENV", "").lower()
    if langgraph_env == "cloud":
        return True
    return bool(os.environ.get("LANGGRAPH_API_URL"))


def route_after_review(state: Dict[str, Any]) -> str:
    return "plan_tot" if state.get("replan") else "test_agents"

def route_after_tests(state: Dict[str, Any]) -> str:
    return "plan_tot" if state.get("replan") else "execute_deep"

def route_after_validation(state: Dict[str, Any]) -> str:
    return state.get("next_action", "complete")


def route_after_fixer(state: Dict[str, Any]) -> str:
    fix_attempts = state.get("fix_attempts", 0)

    if fix_attempts >= 3:
        print(f"🛑 CIRCUIT BREAKER: {fix_attempts} attempts reached - FORCING COMPLETION")
        return "force_complete"

    next_action = state.get("next_action", "validate_again")
    return next_action


def _make_graph(path: str | None = os.environ.get("ASB_SQLITE_DB_PATH")):
    g = StateGraph(AppState)
    g.add_node("plan_tot", plan_tot)
    g.add_node("confidence", compute_plan_confidence)
    g.add_node("review_plan", review_plan)  # HITL interrupt; node re-executes after resume
    g.add_node("test_agents", test_agents)
    g.add_node("execute_deep", execute_deep)
    g.add_node("scaffold_project", scaffold_project)
    g.add_node("code_validator", code_validator_node)
    g.add_node("code_fixer", code_fixer_node)
    g.add_node("sandbox_smoke", sandbox_smoke)
    g.add_node("report", report)

    g.add_edge(START, "plan_tot")
    g.add_edge("plan_tot", "confidence")
    g.add_edge("confidence", "review_plan")
    g.add_conditional_edges("review_plan", route_after_review, {"plan_tot":"plan_tot","test_agents":"test_agents"})
    g.add_conditional_edges("test_agents", route_after_tests, {"plan_tot":"plan_tot","execute_deep":"execute_deep"})
    g.add_edge("execute_deep", "scaffold_project")
    g.add_edge("scaffold_project", "code_validator")
    g.add_conditional_edges(
        "code_validator",
        route_after_validation,
        {
            "complete": "sandbox_smoke",
            "fix_code": "code_fixer",
            "force_complete": "sandbox_smoke",
        },
    )
    g.add_conditional_edges(
        "code_fixer",
        route_after_fixer,
        {
            "validate_again": "code_validator",
            "manual_review": "report",
            "force_complete": "sandbox_smoke",
        },
    )
    g.add_edge("sandbox_smoke", "report")
    g.add_edge("report", END)

    if running_on_langgraph_api():
        return g.compile(checkpointer=None)

    dev_server = os.environ.get("ASB_DEV_SERVER")
    if path and not dev_server:
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        memory = SqliteSaver(conn)
        return g.compile(checkpointer=memory)

    return g.compile()

graph = _make_graph()
