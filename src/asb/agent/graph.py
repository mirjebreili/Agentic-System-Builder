"""Updated graph with package discovery pipeline."""
from __future__ import annotations
from typing import Dict, Any
import os
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
import asb_config.settings as s
from asb_config.settings import SETTINGS_UID
from asb.agent.state import AppState
from asb.agent.planner import plan_tot
from asb.agent.confidence import compute_plan_confidence
from asb.agent.hitl import review_plan
# New package discovery imports
from asb.agent.package_planner import package_planner_node
from asb.package_discoverer import discover_packages_node
from asb.package_ranker import rank_packages_node
from asb.agent.package_integrator import integrate_packages_node
from asb.package_validator import (
    validate_packages_node, 
    should_replan_packages, 
    replan_or_finalize_node
)

# Keep existing Langfuse setup
from langfuse.langchain import CallbackHandler
from langfuse import get_client
 
langfuse = get_client()
 
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")

langfuse_handler = CallbackHandler()

print("### USING settings_v2 FROM:", s.__file__)
print("### SETTINGS UID:", SETTINGS_UID)


def running_on_langgraph_api() -> bool:
    """Return ``True`` when executing within a LangGraph-managed runtime."""
    langgraph_env = os.environ.get("LANGGRAPH_ENV", "").lower()
    if langgraph_env == "cloud":
        return True
    return bool(os.environ.get("LANGGRAPH_API_URL"))


def route_after_review(state: Dict[str, Any]) -> str:
    """Route after HITL review - either replan or start package discovery."""
    return "plan_tot" if state.get("replan") else "package_planner"


def _make_graph(path: str | None = os.environ.get("ASB_SQLITE_DB_PATH")):
    g = StateGraph(AppState)
    
    # Original planning phase (keep unchanged)
    g.add_node("plan_tot", plan_tot)
    g.add_node("confidence", compute_plan_confidence)
    g.add_node("review_plan", review_plan)  # HITL interrupt
    
    # New package discovery pipeline
    g.add_node("package_planner", package_planner_node)
    g.add_node("discover_packages", discover_packages_node)
    g.add_node("rank_packages", rank_packages_node)
    g.add_node("integrate_packages", integrate_packages_node)
    g.add_node("validate_packages", validate_packages_node)
    g.add_node("finalize", replan_or_finalize_node)

    # Original flow up to HITL
    g.add_edge(START, "plan_tot")
    g.add_edge("plan_tot", "confidence")
    g.add_edge("confidence", "review_plan")
    
    # Route after HITL to package discovery
    g.add_conditional_edges(
        "review_plan",
        route_after_review,
        {
            "plan_tot": "plan_tot",           # Replan if needed
            "package_planner": "package_planner"  # Start package discovery
        },
    )
    
    # Package discovery pipeline
    g.add_edge("package_planner", "discover_packages")
    g.add_edge("discover_packages", "rank_packages")
    g.add_edge("rank_packages", "integrate_packages")
    g.add_edge("integrate_packages", "validate_packages")
    
    # Validation routing
    g.add_conditional_edges(
        "validate_packages",
        should_replan_packages,
        {
            "True": "finalize",   # Replan with stricter criteria
            "False": "finalize",  # Finalize solution
        },
    )
    
    # Loop back for replanning
    g.add_edge("finalize", "discover_packages")  # If replanning needed
    g.add_edge("finalize", END)  # Final completion

    # Checkpointer setup (keep existing logic)
    if running_on_langgraph_api():
        return g.compile(checkpointer=None).with_config({
            "callbacks": [langfuse_handler]
        })

    dev_server = os.environ.get("ASB_DEV_SERVER")
    if path and not dev_server:
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        memory = SqliteSaver(conn)
        return g.compile(checkpointer=memory)

    return g.compile().with_config({
        "callbacks": [langfuse_handler]
    })


graph = _make_graph()
