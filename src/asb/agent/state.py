from __future__ import annotations
from typing import Any, Dict, List, Literal, TypedDict

class ChatMessage(TypedDict, total=False):
    role: Literal["human","user","assistant","system","tool"]
    content: str

class AppState(TypedDict, total=False):
    architecture: Dict[str, Any]
    artifacts: Dict[str, Any]
    build_attempts: int
    code_fixes: Dict[str, Any]
    code_validation: Dict[str, Any]
    consecutive_failures: int
    coordinator_decision: str
    current_step: Dict[str, bool]
    debug: Dict[str, Any]
    error: str
    evaluations: List[Dict[str, Any]]
    fix_attempts: int
    fix_strategy_used: str | None
    flags: Dict[str, bool]
    generated_files: Dict[str, str]
    goal: str
    implemented_nodes: List[Dict[str, Any]]
    input_text: str
    last_implemented_node: str | None
    last_user_input: str
    messages: List[ChatMessage]
    metrics: Dict[str, Any]
    next_action: str
    passed: bool
    plan: Dict[str, Any]
    replan: bool
    repair_start_time: float
    report: Dict[str, Any]
    requirements: Dict[str, Any]
    review: Dict[str, Any]
    sandbox: Dict[str, Any]
    scaffold: Dict[str, Any]
    selected_thought: Dict[str, Any]
    syntax_validation: Dict[str, Any]
    tests: Dict[str, Any]
    thoughts: List[str]
    tot: Dict[str, Any]
    validation_errors: List[str]


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
