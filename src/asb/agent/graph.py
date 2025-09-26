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
from asb.agent.requirements_analyzer import requirements_analyzer_node
from asb.agent.architecture_designer import architecture_designer_node
from asb.agent.state_generator import state_generator_node
from asb.agent.node_implementor import node_implementor_node
from asb.agent.build_coordinator import build_coordinator_node
from asb.agent.micro import bug_localizer_node, diff_patcher_node, sandbox_runner_node
from asb.agent.sandbox import comprehensive_sandbox_test as sandbox_smoke
from asb.agent.report import report

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

print("### USING settings_v2 FROM:", s.__file__)
print("### SETTINGS UID:", SETTINGS_UID)


def running_on_langgraph_api() -> bool:
    """Return ``True`` when executing within a LangGraph-managed runtime."""

    langgraph_env = os.environ.get("LANGGRAPH_ENV", "").lower()
    if langgraph_env == "cloud":
        return True
    return bool(os.environ.get("LANGGRAPH_API_URL"))


def route_after_review(state: Dict[str, Any]) -> str:
    return "plan_tot" if state.get("replan") else "requirements_analyzer"

def route_after_build_coordinator(state: Dict[str, Any]) -> str:
    decision = (state or {}).get("coordinator_decision")

    if decision == "proceed":
        return "sandbox_smoke"

    return "bug_localizer"


def _make_graph(path: str | None = os.environ.get("ASB_SQLITE_DB_PATH")):
    g = StateGraph(AppState)
    g.add_node("plan_tot", plan_tot)
    g.add_node("confidence", compute_plan_confidence)
    g.add_node("review_plan", review_plan)  # HITL interrupt; node re-executes after resume
    g.add_node("requirements_analyzer", requirements_analyzer_node)
    g.add_node("architecture_designer", architecture_designer_node)
    g.add_node("state_generator", state_generator_node)
    g.add_node("node_implementor", node_implementor_node)
    g.add_node("build_coordinator", build_coordinator_node)
    g.add_node("bug_localizer", bug_localizer_node)
    g.add_node("diff_patcher", diff_patcher_node)
    g.add_node("sandbox_runner", sandbox_runner_node)
    g.add_node("sandbox_smoke", sandbox_smoke)
    g.add_node("report", report)

    g.add_edge(START, "plan_tot")
    g.add_edge("plan_tot", "confidence")
    g.add_edge("confidence", "review_plan")
    g.add_conditional_edges(
        "review_plan",
        route_after_review,
        {"plan_tot": "plan_tot", "requirements_analyzer": "requirements_analyzer"},
    )
    g.add_edge("requirements_analyzer", "architecture_designer")
    g.add_edge("architecture_designer", "state_generator")
    g.add_edge("state_generator", "node_implementor")
    g.add_edge("node_implementor", "build_coordinator")
    g.add_conditional_edges(
        "build_coordinator",
        route_after_build_coordinator,
        {
            "sandbox_smoke": "sandbox_smoke",
            "bug_localizer": "bug_localizer",
        },
    )
    g.add_edge("bug_localizer", "diff_patcher")
    g.add_edge("diff_patcher", "sandbox_runner")
    g.add_edge("sandbox_runner", "report")
    g.add_edge("sandbox_smoke", "report")
    g.add_edge("report", END)

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
