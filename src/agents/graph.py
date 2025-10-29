from __future__ import annotations
from typing import Dict, Any
import os
from langgraph.graph import StateGraph, END
from agents.state import AppState
from agents.plugin_analyzer import extract_system_elements
from agents.splitter import split_task
from agents.planner import plan_tot
from agents.confidence import compute_plan_confidence
from agents.hitl import review_plan
from agents.formatter import format_plan_order

from langfuse.langchain import CallbackHandler

lf_handler = CallbackHandler()


def route_after_review(state: Dict[str, Any]) -> str:
    """Route after review: either format_plan_order if approved or back to plan_tot if replan needed."""
    return "plan_tot" if state.get("replan") else "format_plan_order"


def _make_graph():
    """Create the LangGraph state graph."""
    g = StateGraph(AppState)
    
    # Add nodes
    g.add_node("extract_system_elements", extract_system_elements)
    g.add_node("split_task", split_task)
    g.add_node("plan_tot", plan_tot)
    g.add_node("confidence", compute_plan_confidence)
    g.add_node("review_plan", review_plan)
    g.add_node("format_plan_order", format_plan_order)

    # Define edges
    g.set_entry_point("extract_system_elements")
    g.add_edge("extract_system_elements", "split_task")
    g.add_edge("split_task", "plan_tot")
    g.add_edge("plan_tot", "confidence")
    g.add_edge("confidence", "review_plan")
    g.add_conditional_edges(
        "review_plan",
        route_after_review,
        {"plan_tot": "plan_tot", "format_plan_order": "format_plan_order"},
    )
    g.add_edge("format_plan_order", END)
    
    # Compile with interrupt for HITL review
    return g.compile(interrupt_before=["review_plan"])


graph = _make_graph().with_config({"callbacks": [lf_handler]})
