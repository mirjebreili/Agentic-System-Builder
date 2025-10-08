from __future__ import annotations
from typing import Dict, Any
import os
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from agents.state import AppState
from agents.planner import plan_tot
from agents.confidence import compute_plan_confidence
from agents.hitl import review_plan
from agents.plugin_analyzer import analyze_plugins

# from langfuse.callback import CallbackHandler
# langfuse_handler = CallbackHandler(
#     public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
#     secret_key=os.environ["LANGFUSE_SECRET_KEY"],
#     host="http://192.168.33.85:3000"
# )
from langfuse.langchain import CallbackHandler
from langfuse import get_client
 
langfuse = get_client()
 
# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")

# Initialize the Langfuse handler
langfuse_handler = CallbackHandler()


def route_initial(state: Dict[str, Any]) -> str:
    """Route to appropriate analyzer based on input type."""
    # Check if this is a plugin sequencing task
    if state.get("plugins") or "plugin" in state.get("goal", "").lower():
        return "analyze_plugins"
    else:
        return "plan_tot"

def running_on_langgraph_api() -> bool:
    """Return ``True`` when executing within a LangGraph-managed runtime."""

    langgraph_env = os.environ.get("LANGGRAPH_ENV", "").lower()
    if langgraph_env == "cloud":
        return True
    return bool(os.environ.get("LANGGRAPH_API_URL"))


def route_after_review(state: Dict[str, Any]) -> str:
    # Check if this is a plugin sequencing task
    if state.get("plugins") or "plugin" in state.get("goal", "").lower():
        if state.get("replan"):
            return "analyze_plugins"
        else:
            return "__end__"
    
    # Original logic for general planning
    return "plan_tot" if state.get("replan") else "__end__"


def _make_graph(path: str | None = os.environ.get("ASB_SQLITE_DB_PATH")):
    g = StateGraph(AppState)
    g.add_node("plan_tot", plan_tot)
    g.add_node("analyze_plugins", analyze_plugins)
    g.add_node("confidence", compute_plan_confidence)
    g.add_node("review_plan", review_plan)  # HITL interrupt; node re-executes after resume

    g.add_conditional_edges(START, route_initial, {"plan_tot": "plan_tot", "analyze_plugins": "analyze_plugins"})
    g.add_edge("plan_tot", "confidence")
    g.add_edge("analyze_plugins", "confidence")
    g.add_edge("confidence", "review_plan")
    g.add_conditional_edges(
        "review_plan",
        route_after_review,
        {"plan_tot": "plan_tot", "analyze_plugins": "analyze_plugins", "__end__": END},
    )

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
