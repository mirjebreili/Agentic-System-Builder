from __future__ import annotations

from typing import Annotated, Any, Dict, List, TypedDict
import operator

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class AppState(TypedDict, total=False):
    # Conversation/history
    messages: Annotated[List[AnyMessage], add_messages]

    # Core inputs
    goal: str
    input_text: str
    
    # Planning/architecture
    plan: Dict[str, Any]
    architecture: Dict[str, Any]

    # Execution outputs
    result: str
    final_output: str

    # Diagnostics
    error: str
    errors: Annotated[List[str], operator.add]

    # Flexible scratchpad for intermediate values
    scratch: Annotated[Dict[str, Any], operator.or_]

    # Build-time scaffolding diagnostics
    scaffold: Annotated[Dict[str, Any], operator.or_]

    # Adaptive improvement metadata
    self_correction: Annotated[Dict[str, Any], operator.or_]

    # Advanced reasoning containers
    tot: Annotated[Dict[str, Any], operator.or_]


def update_state_with_circuit_breaker(state: Dict[str, Any]) -> Dict[str, Any]:
    """Add circuit breaker logic to prevent infinite loops."""

    if "fix_attempts" not in state:
        state["fix_attempts"] = 0

    if "consecutive_failures" not in state:
        state["consecutive_failures"] = 0

    if "repair_start_time" not in state:
        import time

        state["repair_start_time"] = time.time()

    return state
