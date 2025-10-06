"""State graph for the agent orchestration flow."""
from __future__ import annotations

import os
import sqlite3
import types
from typing import Any, Dict
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

import asb_config.settings as s
from asb.agent.confidence import compute_plan_confidence
from asb.agent.hitl import review_plan
from asb.agent.planner import plan_tot
from asb.agent.report import report
from asb.agent.state import AppState
from asb.utils.state_preparer import prepare_initial_state

def _init_langfuse_handler():
    """Best-effort initialization of the Langfuse callback handler."""

    try:
        from langfuse import Langfuse  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"Langfuse client is unavailable: {exc}")
        return None

    try:
        from langfuse.callback.langchain import (  # type: ignore
            CallbackHandler as LangfuseCallbackHandler,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"Langfuse LangChain callback unavailable: {exc}")
        return None

    try:
        client = Langfuse()
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"Failed to initialize Langfuse client: {exc}")
        return None

    try:
        if client.auth_check():
            print("Langfuse client is authenticated and ready!")
        else:
            print("Langfuse authentication failed. Please check your credentials and host.")
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"Langfuse authentication check failed: {exc}")

    try:
        return LangfuseCallbackHandler(client=client)
    except TypeError:
        try:
            return LangfuseCallbackHandler()
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"Langfuse callback initialization failed: {exc}")
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"Langfuse callback initialization failed: {exc}")

    return None


_LANGFUSE_HANDLER = _init_langfuse_handler()


def _apply_callbacks(graph):
    """Attach callbacks to a compiled graph when available."""

    if _LANGFUSE_HANDLER:
        return graph.with_config({"callbacks": [_LANGFUSE_HANDLER]})
    return graph


def _apply_state_preparer(graph):
    """Patch graph execution methods to normalize incoming state."""

    def _prepare(state):
        return prepare_initial_state(state)

    if hasattr(graph, "invoke"):
        original_invoke = graph.invoke

        def invoke(self, state, *args, **kwargs):
            return original_invoke(_prepare(state), *args, **kwargs)

        graph.invoke = types.MethodType(invoke, graph)

    if hasattr(graph, "ainvoke"):
        original_ainvoke = graph.ainvoke

        async def ainvoke(self, state, *args, **kwargs):
            return await original_ainvoke(_prepare(state), *args, **kwargs)

        graph.ainvoke = types.MethodType(ainvoke, graph)

    if hasattr(graph, "stream"):
        original_stream = graph.stream

        def stream(self, state, *args, **kwargs):
            for chunk in original_stream(_prepare(state), *args, **kwargs):
                yield chunk

        graph.stream = types.MethodType(stream, graph)

    if hasattr(graph, "astream"):
        original_astream = graph.astream

        async def astream(self, state, *args, **kwargs):
            async for chunk in original_astream(_prepare(state), *args, **kwargs):
                yield chunk

        graph.astream = types.MethodType(astream, graph)

    if hasattr(graph, "astream_events"):
        original_astream_events = graph.astream_events

        async def astream_events(self, state, *args, **kwargs):
            async for event in original_astream_events(_prepare(state), *args, **kwargs):
                yield event

        graph.astream_events = types.MethodType(astream_events, graph)

    if hasattr(graph, "batch"):
        original_batch = graph.batch

        def batch(self, states, *args, **kwargs):
            prepared = [_prepare(state) for state in states]
            return original_batch(prepared, *args, **kwargs)

        graph.batch = types.MethodType(batch, graph)

    if hasattr(graph, "abatch"):
        original_abatch = graph.abatch

        async def abatch(self, states, *args, **kwargs):
            prepared = [_prepare(state) for state in states]
            return await original_abatch(prepared, *args, **kwargs)

        graph.abatch = types.MethodType(abatch, graph)

    return graph

print("### USING settings_v2 FROM:", s.__file__)
print("### SETTINGS UID:", s.SETTINGS_UID)


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
    def finalize(compiled):
        return _apply_state_preparer(_apply_callbacks(compiled))

    if running_on_langgraph_api():
        return finalize(g.compile(checkpointer=None))

    dev_server = os.environ.get("ASB_DEV_SERVER")
    if path and not dev_server:
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        memory = SqliteSaver(conn)
        return finalize(g.compile(checkpointer=memory))

    return finalize(g.compile())


graph = _make_graph()
