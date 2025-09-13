from __future__ import annotations
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from src.agent.state import AppState
from src.agent.planner import plan_tot
from src.agent.confidence import compute_plan_confidence
from src.agent.hitl import review_plan
from src.agent.tests_node import test_agents
from src.agent.executor import execute_deep
from src.agent.scaffold import scaffold_project
from src.agent.sandbox import sandbox_smoke
from src.agent.report import report

def route_after_review(state: Dict[str, Any]) -> str:
    return "plan_tot" if state.get("replan") else "test_agents"

def route_after_tests(state: Dict[str, Any]) -> str:
    return "plan_tot" if state.get("replan") else "execute_deep"

def _make_graph():
    g = StateGraph(AppState)
    g.add_node("plan_tot", plan_tot)
    g.add_node("confidence", compute_plan_confidence)
    g.add_node("review_plan", review_plan)     # HITL interrupt
    g.add_node("test_agents", test_agents)
    g.add_node("execute_deep", execute_deep)
    g.add_node("scaffold_project", scaffold_project)
    g.add_node("sandbox_smoke", sandbox_smoke)
    g.add_node("report", report)

    g.add_edge(START, "plan_tot")
    g.add_edge("plan_tot", "confidence")
    g.add_edge("confidence", "review_plan")
    g.add_conditional_edges("review_plan", route_after_review, {"plan_tot":"plan_tot","test_agents":"test_agents"})
    g.add_conditional_edges("test_agents", route_after_tests, {"plan_tot":"plan_tot","execute_deep":"execute_deep"})
    g.add_edge("execute_deep", "scaffold_project")
    g.add_edge("scaffold_project", "sandbox_smoke")
    g.add_edge("sandbox_smoke", "report")
    g.add_edge("report", END)

    return g.compile()

graph = _make_graph()
