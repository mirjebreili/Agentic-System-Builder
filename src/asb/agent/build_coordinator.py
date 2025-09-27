from __future__ import annotations

"""Coordinated orchestration of Agentic System Builder micro-agents."""

import logging
from typing import Any, Callable, Dict, List, MutableMapping, Tuple

from asb.agent.architecture_designer import architecture_designer_node
from asb.agent.micro import (
    bug_localizer_node,
    context_collector_node,
    critic_judge_node,
    diff_patcher_node,
    final_check_fallback_node,
    final_check_node,
    import_resolver_node,
    logic_implementor_node,
    sandbox_runner_node,
    skeleton_writer_node,
    state_schema_writer_node,
    tot_variant_maker_node,
    unit_test_writer_node,
)
from asb.agent.node_implementor import node_implementor_node
from asb.agent.report import report
from asb.agent.requirements_analyzer import requirements_analyzer_node
from asb.agent.scaffold import scaffold_project
from asb.agent.state import update_state_with_circuit_breaker
from asb.agent.state_generator import state_generator_node
from asb.agent.syntax_validator import syntax_validator_node

logger = logging.getLogger(__name__)

StepCallable = Callable[[Dict[str, Any]], Dict[str, Any]]
StepSpec = Tuple[str, StepCallable]

MAX_REPAIR_ATTEMPTS = 2

BASE_SEQUENCE: Tuple[StepSpec, ...] = (
    ("context_collector", context_collector_node),
    ("requirements_analyzer", requirements_analyzer_node),
    ("architecture_designer", architecture_designer_node),
    ("state_generator", state_generator_node),
    ("node_implementor", node_implementor_node),
    ("syntax_validator", syntax_validator_node),
    ("scaffold_project", scaffold_project),
    ("state_schema_writer", state_schema_writer_node),
    ("skeleton_writer", skeleton_writer_node),
    ("import_resolver", import_resolver_node),
    ("logic_implementor", logic_implementor_node),
    ("unit_test_writer", unit_test_writer_node),
)

REPAIR_SEQUENCE: Tuple[StepSpec, ...] = (
    ("bug_localizer", bug_localizer_node),
    ("tot_variant_maker", tot_variant_maker_node),
    ("critic_judge", critic_judge_node),
    ("diff_patcher", diff_patcher_node),
)


def _ensure_trace(state: MutableMapping[str, Any]) -> List[Dict[str, Any]]:
    debug = state.setdefault("debug", {})
    coordinator = debug.setdefault("build_coordinator", {})
    trace = coordinator.setdefault("trace", [])
    return trace


def _record_trace(
    trace: List[Dict[str, Any]],
    *,
    step: str,
    status: str,
    attempt: int,
    phase: str,
    **details: Any,
) -> None:
    entry = {"step": step, "status": status, "attempt": attempt, "phase": phase}
    if details:
        entry.update(details)
    trace.append(entry)


def _run_step(
    state: Dict[str, Any],
    *,
    name: str,
    func: StepCallable,
    trace: List[Dict[str, Any]],
    attempt: int,
    phase: str,
) -> Tuple[Dict[str, Any], bool]:
    try:
        updated = func(state)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Micro-agent step '%s' raised an exception", name)
        _record_trace(trace, step=name, status="error", attempt=attempt, phase=phase, error=str(exc))
        state.setdefault("errors", []).append(f"{name}: {exc}")
        return state, False

    _record_trace(trace, step=name, status="success", attempt=attempt, phase=phase)
    return updated, True


def _run_sequence(
    state: Dict[str, Any],
    *,
    sequence: Tuple[StepSpec, ...],
    trace: List[Dict[str, Any]],
    attempt: int,
    phase: str,
) -> Tuple[Dict[str, Any], bool]:
    current = state
    for name, func in sequence:
        current, ok = _run_step(current, name=name, func=func, trace=trace, attempt=attempt, phase=phase)
        if not ok:
            return current, False
    return current, True


def _sandbox_success(state: Dict[str, Any]) -> bool:
    sandbox = state.get("sandbox")
    if isinstance(sandbox, dict):
        return bool(sandbox.get("ok"))
    return False


def coordinate_build(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the build pipeline and optional repair loop."""

    working_state = update_state_with_circuit_breaker(dict(state or {}))
    trace = _ensure_trace(working_state)

    attempt = int(working_state.get("build_attempts", 0)) + 1
    working_state["build_attempts"] = attempt

    working_state, ok = _run_sequence(
        working_state,
        sequence=BASE_SEQUENCE,
        trace=trace,
        attempt=attempt,
        phase="build",
    )
    if not ok:
        working_state.setdefault("coordinator_decision", "halt")
        working_state.setdefault("next_action", "halt")
        return working_state

    # Initial sandbox check.
    working_state, sandbox_ok = _run_step(
        working_state,
        name="sandbox_runner",
        func=sandbox_runner_node,
        trace=trace,
        attempt=attempt,
        phase="sandbox",
    )
    if sandbox_ok and _sandbox_success(working_state):
        working_state, _ = _run_step(
            working_state,
            name="final_check",
            func=final_check_node,
            trace=trace,
            attempt=attempt,
            phase="finalize",
        )
        if not (working_state.get("final_check") or {}).get("ok"):
            working_state, _ = _run_step(
                working_state,
                name="final_check_fallback",
                func=final_check_fallback_node,
                trace=trace,
                attempt=attempt,
                phase="finalize",
            )
        working_state = report(working_state)
        working_state["coordinator_decision"] = "proceed"
        working_state["next_action"] = "scaffold"
        working_state["consecutive_failures"] = 0
        return working_state

    repair_attempt = 0
    while repair_attempt < MAX_REPAIR_ATTEMPTS:
        repair_attempt += 1
        working_state, ok = _run_sequence(
            working_state,
            sequence=REPAIR_SEQUENCE,
            trace=trace,
            attempt=attempt,
            phase=f"repair_{repair_attempt}",
        )
        if not ok:
            working_state.setdefault("coordinator_decision", "halt")
            working_state.setdefault("next_action", "halt")
            return working_state

        working_state, sandbox_ok = _run_step(
            working_state,
            name=f"sandbox_runner_retry_{repair_attempt}",
            func=sandbox_runner_node,
            trace=trace,
            attempt=attempt,
            phase=f"sandbox_retry_{repair_attempt}",
        )
        if sandbox_ok and _sandbox_success(working_state):
            working_state, _ = _run_step(
                working_state,
                name=f"final_check_retry_{repair_attempt}",
                func=final_check_node,
                trace=trace,
                attempt=attempt,
                phase=f"finalize_retry_{repair_attempt}",
            )
            if not (working_state.get("final_check") or {}).get("ok"):
                working_state, _ = _run_step(
                    working_state,
                    name=f"final_check_fallback_retry_{repair_attempt}",
                    func=final_check_fallback_node,
                    trace=trace,
                    attempt=attempt,
                    phase=f"finalize_retry_{repair_attempt}",
                )
            working_state = report(working_state)
            working_state["coordinator_decision"] = "proceed"
            working_state["next_action"] = "scaffold"
            working_state["consecutive_failures"] = 0
            return working_state

    working_state.setdefault("errors", []).append("sandbox_validation_failed")
    working_state["coordinator_decision"] = "halt"
    working_state["next_action"] = "halt"
    working_state["consecutive_failures"] = int(working_state.get("consecutive_failures", 0)) + 1
    return working_state


def build_coordinator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph-style wrapper for :func:`coordinate_build`."""

    messages = list(state.get("messages") or [])
    try:
        updated = coordinate_build(state)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Unhandled error in build coordinator node")
        fallback = update_state_with_circuit_breaker(dict(state or {}))
        fallback.setdefault("errors", []).append(f"build_coordinator: {exc}")
        fallback["coordinator_decision"] = "halt"
        fallback["next_action"] = "halt"
        messages.append(
            {
                "role": "assistant",
                "content": "[build-coordinator-error]\nEncountered an unexpected error. Halting orchestration.",
            }
        )
        fallback["messages"] = messages
        return fallback

    decision = updated.get("coordinator_decision", "undecided")
    next_action = updated.get("next_action", "")
    messages.append(
        {
            "role": "assistant",
            "content": f"[build-coordinator]\nDecision: {decision}; next: {next_action or 'n/a'}",
        }
    )
    updated["messages"] = messages
    return updated


__all__ = [
    "BASE_SEQUENCE",
    "REPAIR_SEQUENCE",
    "MAX_REPAIR_ATTEMPTS",
    "coordinate_build",
    "build_coordinator_node",
]
