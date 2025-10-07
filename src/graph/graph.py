from __future__ import annotations

from typing import Any, Dict, List

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ..agents.hitl import perform_hitl
from ..agents.planner import plan_candidates
from ..tools.registry import build_registry
from ..utils.parsing import parse_first_message
from ..utils.types import PlannerState


def _extract_first_message(state: PlannerState) -> str:
    if state.get("input_text"):
        return str(state["input_text"])

    messages = state.get("messages")
    if not messages:
        return ""

    for message in messages:
        role = message.get("role") if isinstance(message, dict) else None
        if role == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: List[str] = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
                if parts:
                    return "\n".join(parts)
    return ""


def ingest_first_message(state: PlannerState) -> PlannerState:
    raw = _extract_first_message(state)
    parsed = parse_first_message(raw)
    question = parsed.get("question", "")
    plugin_docs = parsed.get("plugin_docs", {})
    return {"question": question, "plugin_docs": plugin_docs}


def build_registry_node(state: PlannerState) -> PlannerState:
    registry = build_registry(state.get("plugin_docs", {}))
    return {"registry": registry}


def planner_node(state: PlannerState) -> PlannerState:
    question = state.get("question", "")
    registry = state.get("registry", {})
    result = plan_candidates(question, registry, k=3)
    pending_plan: List[str] = []
    if result["candidates"]:
        pending_plan = list(result["candidates"][result["chosen"]].plan)
    return {"planner_result": result, "pending_plan": pending_plan}


def hitl_node(state: PlannerState) -> PlannerState:
    update = perform_hitl(state)
    if "approved_plan" not in update and "approved_plan" in state:
        update["approved_plan"] = state["approved_plan"]
    return update


def get_graph() -> Any:
    builder = StateGraph(PlannerState)
    builder.add_node("ingest_first_message", ingest_first_message)
    builder.add_node("build_registry", build_registry_node)
    builder.add_node("planner", planner_node)
    builder.add_node("hitl", hitl_node)

    builder.add_edge(START, "ingest_first_message")
    builder.add_edge("ingest_first_message", "build_registry")
    builder.add_edge("build_registry", "planner")
    builder.add_edge("planner", "hitl")
    builder.add_edge("hitl", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)
