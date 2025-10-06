"""State graph for the agent orchestration flow."""
from __future__ import annotations

import functools
import os
import sqlite3
import types
from typing import Any, Dict

from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

import asb_config.settings as s
from asb.agent.confidence import compute_plan_confidence
from asb.agent.hitl import review_plan
from asb.agent.planner import plan_tot
from asb.agent.report import report
from asb.agent.state import AppState
from asb.utils.state_preparer import prepare_initial_state
from asb_config.settings import SETTINGS_UID

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
    """Route after HITL review - either replan or report the plan."""
    return "plan_tot" if state.get("replan") else "report"


def _make_graph(path: str | None = os.environ.get("ASB_SQLITE_DB_PATH")):
    g = StateGraph(AppState)
    
    # Original planning phase (keep unchanged)
    g.add_node("plan_tot", plan_tot)
    g.add_node("confidence", compute_plan_confidence)
    g.add_node("review_plan", review_plan)  # HITL interrupt
    g.add_node("report", report)

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
            "report": "report"  # Report approved plan
        },
    )
    g.add_edge("report", END)

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


def _apply_attachment_preparer(graph_obj):
    """Patch a compiled graph so entrypoints prepare initial state attachments."""

    if getattr(graph_obj, "_attachment_preparer_applied", False):
        return graph_obj

    def _prepare_single(args, kwargs):
        if args:
            prepared_state = prepare_initial_state(args[0])
            args = (prepared_state, *args[1:])
        elif "state" in kwargs:
            kwargs = dict(kwargs)
            kwargs["state"] = prepare_initial_state(kwargs["state"])
        return args, kwargs

    def _prepare_many(args, kwargs):
        if args:
            prepared_states = [prepare_initial_state(state) for state in args[0]]
            args = (prepared_states, *args[1:])
        elif "states" in kwargs:
            kwargs = dict(kwargs)
            kwargs["states"] = [prepare_initial_state(state) for state in kwargs["states"]]
        return args, kwargs

    def _wrap_method(name, *, is_async: bool = False, is_batch: bool = False):
        if not hasattr(graph_obj, name):
            return

        original = getattr(graph_obj, name)
        if original is None:
            return

        preparer = _prepare_many if is_batch else _prepare_single

        if is_async:
            async def wrapper(self, *args, **kwargs):
                args, kwargs = preparer(args, kwargs)
                return await original(*args, **kwargs)
        else:
            def wrapper(self, *args, **kwargs):
                args, kwargs = preparer(args, kwargs)
                return original(*args, **kwargs)

        setattr(
            graph_obj,
            name,
            types.MethodType(
                functools.wraps(original)(wrapper),
                graph_obj,
            ),
        )

    _wrap_method("invoke")
    _wrap_method("ainvoke", is_async=True)
    _wrap_method("stream")
    _wrap_method("astream", is_async=True)
    _wrap_method("astream_events", is_async=True)
    _wrap_method("batch", is_batch=True)
    _wrap_method("abatch", is_async=True, is_batch=True)

    setattr(graph_obj, "_attachment_preparer_applied", True)
    return graph_obj


def graph_factory():
    return _apply_attachment_preparer(_make_graph())


graph = graph_factory()
