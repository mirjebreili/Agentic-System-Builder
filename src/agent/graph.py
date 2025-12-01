from __future__ import annotations
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AppState
from .planner import plan_node
from .tot import tot_planner_node
from .confidence import compute_plan_confidence
from .review import review_plan_node
from .executor import execute_plan_node

def _make_graph():
    g = StateGraph(AppState)
    g.add_node("plan", plan_node)
    g.add_node("tot_planner", tot_planner_node)
    g.add_node("confidence", compute_plan_confidence)
    g.add_node("review_plan", review_plan_node)
    g.add_node("execute", execute_plan_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "tot_planner")
    g.add_edge("tot_planner", "confidence")
    g.add_edge("confidence", "review_plan")
    g.add_edge("review_plan", "execute")
    g.add_edge("execute", END)

    return g.compile(checkpointer=MemorySaver())

graph = _make_graph()
