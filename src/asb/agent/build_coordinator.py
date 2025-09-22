from __future__ import annotations

"""Coordinated orchestration of Agentic System Builder micro-agents."""

import logging
from typing import Any, Callable, Dict, List, MutableMapping, Tuple

from asb.agent.architecture_designer import architecture_designer_node
from asb.agent.code_fixer import code_fixer_node
from asb.agent.node_implementor import node_implementor_node
from asb.agent.report import report
from asb.agent.requirements_analyzer import requirements_analyzer_node
from asb.agent.sandbox import comprehensive_sandbox_test as sandbox_smoke
from asb.agent.scaffold import scaffold_project
from asb.agent.state import update_state_with_circuit_breaker
from asb.agent.state_generator import state_generator_node
from asb.agent.syntax_validator import syntax_validator_node

logger = logging.getLogger(__name__)

StepCallable = Callable[[Dict[str, Any]], Dict[str, Any]]
StepSpec = Tuple[str, StepCallable]

MAX_RETRY_ATTEMPTS = 3
HALTING_ACTIONS = {"halt", "manual_review", "force_complete"}

BUILD_SEQUENCE: Tuple[StepSpec, ...] = (
    ("requirements_analyzer", requirements_analyzer_node),
    ("architecture_designer", architecture_designer_node),
    ("state_generator", state_generator_node),
    ("node_implementor", node_implementor_node),
    ("syntax_validator", syntax_validator_node),
    ("scaffold_project", scaffold_project),
    ("sandbox_smoke", sandbox_smoke),
    ("report", report),
)


def _ensure_debug_trace(state: MutableMapping[str, Any]) -> List[Dict[str, Any]]:
    debug = state.setdefault("debug", {})
    coordinator_debug = debug.setdefault("build_coordinator", {})
    trace = coordinator_debug.setdefault("trace", [])
    coordinator_debug.setdefault("retry_budget", MAX_RETRY_ATTEMPTS)
    return trace


def _record_trace(
    trace: List[Dict[str, Any]],
    *,
    step: str,
    status: str,
    attempt: int,
    **details: Any,
) -> None:
    entry = {"step": step, "status": status, "attempt": attempt}
    if details:
        entry.update(details)
    trace.append(entry)


def _force_complete(
    state: MutableMapping[str, Any],
    trace: List[Dict[str, Any]],
    *,
    attempt: int,
    reason: str,
) -> None:
    state["coordinator_decision"] = "force_complete"
    state["next_action"] = "force_complete"
    _record_trace(
        trace,
        step="build_coordinator",
        status="force_complete",
        attempt=attempt,
        reason=reason,
    )


def _run_validation_cycle(
    state: Dict[str, Any],
    trace: List[Dict[str, Any]],
    *,
    attempt: int,
) -> Tuple[Dict[str, Any], bool]:
    validation_attempts = 0
    while state.get("next_action") == "fix_code":
        if validation_attempts >= MAX_RETRY_ATTEMPTS:
            _force_complete(
                state,
                trace,
                attempt=attempt,
                reason="syntax_validation_retry_budget_exceeded",
            )
            return state, False

        try:
            state = code_fixer_node(state)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Code fixer node raised during validation cycle")
            _record_trace(
                trace,
                step="code_fixer",
                status="error",
                attempt=attempt,
                error=str(exc),
            )
            state["coordinator_decision"] = "halt"
            state["next_action"] = "halt"
            return state, False

        validation_attempts += 1
        fixer_action = state.get("next_action")
        _record_trace(
            trace,
            step="code_fixer",
            status="success",
            attempt=attempt,
            action=fixer_action,
            fix_attempt=validation_attempts,
        )

        if fixer_action in HALTING_ACTIONS - {"force_complete"}:
            state.setdefault("coordinator_decision", "halt")
            return state, False
        if fixer_action == "force_complete":
            state["coordinator_decision"] = "force_complete"
            return state, False
        if fixer_action != "validate_again":
            break

        try:
            state = syntax_validator_node(state)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Syntax validator raised during retry cycle")
            _record_trace(
                trace,
                step="syntax_validator",
                status="error",
                attempt=attempt,
                error=str(exc),
                retry=validation_attempts,
            )
            state["coordinator_decision"] = "halt"
            state["next_action"] = "halt"
            return state, False

        _record_trace(
            trace,
            step="syntax_validator",
            status="retry",
            attempt=attempt,
            retry=validation_attempts + 1,
            action=state.get("next_action"),
        )

    return state, True


def coordinate_build(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the build pipeline by invoking micro-agents in sequence."""

    working_state = update_state_with_circuit_breaker(dict(state or {}))
    trace = _ensure_debug_trace(working_state)

    build_attempts = int(working_state.get("build_attempts", 0)) + 1
    working_state["build_attempts"] = build_attempts
    attempt_number = build_attempts

    consecutive_failures = int(working_state.get("consecutive_failures", 0))

    if attempt_number > MAX_RETRY_ATTEMPTS:
        _force_complete(
            working_state,
            trace,
            attempt=attempt_number,
            reason="build_attempts_exceeded",
        )
        return working_state

    if consecutive_failures >= MAX_RETRY_ATTEMPTS:
        _force_complete(
            working_state,
            trace,
            attempt=attempt_number,
            reason="consecutive_failures_exceeded",
        )
        return working_state

    success = True
    failure_reason = ""

    for step_name, step_callable in BUILD_SEQUENCE:
        logger.info("Build coordinator invoking step: %s", step_name)
        try:
            working_state = step_callable(working_state)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Micro-agent step '%s' raised an exception", step_name)
            _record_trace(
                trace,
                step=step_name,
                status="error",
                attempt=attempt_number,
                error=str(exc),
            )
            success = False
            failure_reason = f"{step_name}_raised"
            working_state.setdefault("coordinator_decision", "halt")
            working_state.setdefault("next_action", "halt")
            break

        next_action = working_state.get("next_action")

        if step_name == "syntax_validator":
            working_state, cycle_success = _run_validation_cycle(
                working_state,
                trace,
                attempt=attempt_number,
            )
            next_action = working_state.get("next_action")
            if not cycle_success:
                success = False
                failure_reason = "validation_cycle_failed"
                break

        status_details: Dict[str, Any] = {
            "step": step_name,
            "status": "success",
            "attempt": attempt_number,
        }
        if next_action is not None:
            status_details["next_action"] = next_action
        trace.append(status_details)

        if next_action in HALTING_ACTIONS:
            success = False
            failure_reason = f"{step_name}_requested_{next_action}"
            working_state.setdefault("coordinator_decision", "halt")
            break

    if success:
        working_state["coordinator_decision"] = "proceed"
        working_state["next_action"] = "scaffold"
        working_state["consecutive_failures"] = 0
    else:
        consecutive_failures += 1
        working_state["consecutive_failures"] = consecutive_failures
        if working_state.get("coordinator_decision") == "force_complete":
            pass
        elif consecutive_failures >= MAX_RETRY_ATTEMPTS:
            _force_complete(
                working_state,
                trace,
                attempt=attempt_number,
                reason=failure_reason or "retry_budget_exceeded",
            )
        else:
            working_state.setdefault("coordinator_decision", "halt")
            working_state.setdefault("next_action", "halt")

    return working_state


def build_coordinator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph-style node wrapper for :func:`coordinate_build`."""

    messages = list(state.get("messages") or [])

    try:
        updated_state = coordinate_build(state)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Unhandled error in build coordinator node")
        fallback_state = update_state_with_circuit_breaker(dict(state or {}))
        fallback_state["consecutive_failures"] = int(
            fallback_state.get("consecutive_failures", 0)
        ) + 1
        fallback_state["coordinator_decision"] = "halt"
        fallback_state["next_action"] = "halt"
        messages.append(
            {
                "role": "assistant",
                "content": "[build-coordinator-error]\nEncountered an unexpected error. Halting orchestration.",
            }
        )
        fallback_state["messages"] = messages
        trace = _ensure_debug_trace(fallback_state)
        _record_trace(
            trace,
            step="build_coordinator",
            status="error",
            attempt=int(fallback_state.get("build_attempts", 0) or 0) + 1,
            error=str(exc),
        )
        return fallback_state

    decision = updated_state.get("coordinator_decision", "undecided")
    next_action = updated_state.get("next_action", "")
    messages.append(
        {
            "role": "assistant",
            "content": f"[build-coordinator]\nDecision: {decision}; next: {next_action or 'n/a'}",
        }
    )
    updated_state["messages"] = messages
    return updated_state


__all__ = [
    "BUILD_SEQUENCE",
    "MAX_RETRY_ATTEMPTS",
    "coordinate_build",
    "build_coordinator_node",
]
