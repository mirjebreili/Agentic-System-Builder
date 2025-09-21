from __future__ import annotations
from typing import Any, Dict, List, Literal, TypedDict

class ChatMessage(TypedDict, total=False):
    role: Literal["human","user","assistant","system","tool"]
    content: str

class AppState(TypedDict, total=False):
    messages: List[ChatMessage]
    plan: Dict[str, Any]
    flags: Dict[str, bool]
    metrics: Dict[str, Any]
    debug: Dict[str, Any]
    tests: Dict[str, Any]
    artifacts: Dict[str, Any]
    review: Dict[str, Any]
    replan: bool
    passed: bool
    scaffold: Dict[str, Any]
    sandbox: Dict[str, Any]
    report: Dict[str, Any]
    requirements: Dict[str, Any]


def update_state_with_circuit_breaker(state: Dict[str, Any]) -> Dict[str, Any]:
    """Add circuit breaker logic to prevent infinite loops"""

    if "fix_attempts" not in state:
        state["fix_attempts"] = 0

    if "consecutive_failures" not in state:
        state["consecutive_failures"] = 0

    if "repair_start_time" not in state:
        import time

        state["repair_start_time"] = time.time()

    return state
