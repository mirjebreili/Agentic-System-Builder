"""Core state schema shared across agent nodes."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import NotRequired, Required, TypedDict


class AppState(TypedDict, total=False):
    """Normalized agent state tracked throughout the workflow."""

    input_text: Required[str]
    messages: Annotated[List[AnyMessage], add_messages]

    goal: NotRequired[str]
    plan: NotRequired[Annotated[Dict[str, Any], operator.or_]]
    architecture: NotRequired[Annotated[Dict[str, Any], operator.or_]]

    result: NotRequired[str]
    final_output: NotRequired[str]

    error: NotRequired[str]
    errors: NotRequired[Annotated[List[str], operator.add]]

    scratch: NotRequired[Annotated[Dict[str, Any], operator.or_]]
    scaffold: NotRequired[Annotated[Dict[str, Any], operator.or_]]
    sandbox: NotRequired[Annotated[Dict[str, Any], operator.or_]]
    self_correction: NotRequired[Annotated[Dict[str, Any], operator.or_]]
    tot: NotRequired[Annotated[Dict[str, Any], operator.or_]]

    confidence: NotRequired[float]
    replan: NotRequired[bool]
    review: NotRequired[Dict[str, Any]]
    debug: NotRequired[Annotated[Dict[str, Any], operator.or_]]
    tests: NotRequired[Dict[str, Any]]
    report: NotRequired[Dict[str, Any]]
    final_response: NotRequired[str]
