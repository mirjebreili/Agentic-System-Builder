from __future__ import annotations
from typing import Dict, Any
import os
from langgraph.graph import StateGraph, START, END
from agents.state import AppState
from agents.planner import plan_tot
from agents.confidence import compute_plan_confidence
from agents.hitl import review_plan
from agents.formatter import format_plan_order

from langfuse.langchain import CallbackHandler

lf_handler = CallbackHandler()  # LangChain/LangGraph-compatible callback

def running_on_langgraph_api() -> bool:
    """Return ``True`` when executing within a LangGraph-managed runtime."""

    langgraph_env = os.environ.get("LANGGRAPH_ENV", "").lower()
    if langgraph_env == "cloud":
        return True
    return bool(os.environ.get("LANGGRAPH_API_URL"))


def route_after_review(state: Dict[str, Any]) -> str:
    """Route after review: either format_plan_order if approved or back to plan_tot if replan needed."""
    return "plan_tot" if state.get("replan") else "format_plan_order"


def _make_graph():
    """Create the LangGraph state graph."""
    g = StateGraph(AppState)
    g.add_node("plan_tot", plan_tot)
    g.add_node("confidence", compute_plan_confidence)
    g.add_node("review_plan", review_plan)  # HITL interrupt; node re-executes after resume
    g.add_node("format_plan_order", format_plan_order)  # Format final plan execution order

    g.set_entry_point("plan_tot")
    g.add_edge("plan_tot", "confidence")
    g.add_edge("confidence", "review_plan")
    g.add_conditional_edges(
        "review_plan",
        route_after_review,
        {"plan_tot": "plan_tot", "format_plan_order": "format_plan_order"},
    )
    g.add_edge("format_plan_order", END)
    
    # For langgraph dev/API, persistence is handled automatically
    # We just need to specify which nodes should interrupt for HITL
    return g.compile(interrupt_before=["review_plan"])

import os
print("Environment variables:")
for key in ['LLM_BASE_URL', 'LLM_MODEL', 'TEMPERATURE', 'LLM_API_KEY', 'LANGFUSE_HOST', 'LANGFUSE_PUBLIC_KEY', 'LANGFUSE_SECRET_KEY']:
    print(f"{key}: {os.getenv(key)}")
graph = _make_graph().with_config({"callbacks": [lf_handler]})
