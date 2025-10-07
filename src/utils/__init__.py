"""Utility types and helpers for the Agentic System Builder."""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langchain_core.messages import BaseMessage


class PlannerState(TypedDict, total=False):
    """State structure shared across planner graph nodes."""

    input_text: str
    messages: List[BaseMessage]
    available_plugins: List[Dict[str, Any]]
    plugin_sequence: List[Dict[str, Any]]

