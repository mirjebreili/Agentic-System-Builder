"""Coordinator for executing scaffold build, validation, and repair phases."""

from __future__ import annotations

import time
from collections.abc import Mapping, MutableMapping
from typing import Any, Iterable, Tuple

from .subgraphs import (
    create_build_subgraph,
    create_repair_subgraph,
    create_validate_subgraph,
)

ScaffoldState = MutableMapping[str, Any]

MAX_VALIDATION_ATTEMPTS = 3

_BUILD_SUBGRAPH = create_build_subgraph()
_VALIDATE_SUBGRAPH = create_validate_subgraph()
_REPAIR_SUBGRAPH = create_repair_subgraph()


def _ensure_scaffold_container(state: ScaffoldState) -> MutableMapping[str, Any]:
    scaffold = state.setdefault("scaffold", {})
    if not isinstance(scaffold, MutableMapping):  # pragma: no cover - defensive
        scaffold = {}
        state["scaffold"] = scaffold
    return scaffold


def _collect_scaffold_errors(state: Mapping[str, Any]) -> list[str]:
    scaffold = state.get("scaffold") if isinstance(state, Mapping) else None
    if isinstance(scaffold, Mapping):
        errors = scaffold.get("errors")
        if isinstance(errors, Iterable):
            return [str(error) for error in errors if error]
    return []


def _record_failure(state: ScaffoldState, message: str) -> None:
    scaffold = _ensure_scaffold_container(state)
    errors = scaffold.setdefault("errors", [])
    if message and message not in errors:
        errors.append(message)
    scaffold["ok"] = False


def _start_phase(state: ScaffoldState, name: str, description: str) -> Tuple[dict[str, Any], float]:
    started = time.time()
    phase: dict[str, Any] = {
        "name": name,
        "description": description,
        "status": "in_progress",
        "started_at": started,
    }
    state["scaffold_phase"] = phase
    return phase, started


def _finish_phase(
    state: ScaffoldState,
    phase: MutableMapping[str, Any],
    started: float,
    *,
    success: bool,
    summary: str,
    details: Mapping[str, Any] | None = None,
    errors: Iterable[str] | None = None,
) -> None:
    completed = time.time()
    phase["completed_at"] = completed
    phase["duration"] = max(0.0, completed - started)
    phase["status"] = "complete" if success else "failed"
    if summary:
        phase["summary"] = summary
    if details is not None:
        phase["details"] = dict(details)
    error_messages = list(errors or [])
    if error_messages:
        phase["error"] = error_messages[0]
        if details is None:
            phase["details"] = {"errors": error_messages}
        else:
            phase.setdefault("details", {}).setdefault("errors", error_messages)
    state["scaffold_phase"] = phase


def _invoke_subgraph(graph: Any, state: ScaffoldState) -> ScaffoldState:
    result = graph.invoke(state)
    if isinstance(result, MutableMapping):
        return result  # type: ignore[return-value]
    if isinstance(result, Mapping):
        state.update(result)
    return state


def scaffold_coordinator(state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Run scaffold build/validate/repair subgraphs with retry handling."""

    working_state: ScaffoldState = dict(state or {})
    _ensure_scaffold_container(working_state)

    architecture_plan: dict[str, Any] = {}
    if isinstance(state, Mapping):
        arch_candidate = state.get("architecture")
        if isinstance(arch_candidate, dict):
            architecture_plan = arch_candidate
        else:
            plan_candidate = state.get("plan")
            if isinstance(plan_candidate, dict):
                architecture_plan = plan_candidate

    user_goal = ""
    if isinstance(state, Mapping):
        user_goal = (
            state.get("goal")
            or state.get("input_text")
            or state.get("last_user_input")
            or ""
        )

    working_state["_scaffold_architecture_plan"] = architecture_plan
    working_state["_scaffold_user_goal"] = str(user_goal)

    scaffold_container = working_state.get("scaffold")
    if isinstance(scaffold_container, Mapping):
        base_path = scaffold_container.get("path")
        if base_path:
            working_state["_scaffold_base_path"] = base_path

    print(f"🔍 SCAFFOLD DEBUG - Architecture plan: {bool(architecture_plan)}")
    print(f"🔍 SCAFFOLD DEBUG - User goal: {user_goal}")
    print(
        f"🔍 SCAFFOLD DEBUG - Plan nodes: {architecture_plan.get('nodes', []) if isinstance(architecture_plan, dict) else []}"
    )

    build_phase, started = _start_phase(
        working_state,
        "build",
        "Generate scaffolded project files and configuration.",
    )
    try:
        working_state = _invoke_subgraph(_BUILD_SUBGRAPH, working_state)
    except Exception as exc:  # pragma: no cover - defensive
        error_message = f"Build phase failed: {exc}"
        _finish_phase(
            working_state,
            build_phase,
            started,
            success=False,
            summary="Scaffold build encountered an error.",
            errors=[error_message],
        )
        _record_failure(working_state, error_message)
        return dict(working_state)

    _finish_phase(
        working_state,
        build_phase,
        started,
        success=True,
        summary="Scaffold build completed successfully.",
        details={},
    )

    attempt = 0
    while True:
        attempt += 1
        validation_phase, validation_started = _start_phase(
            working_state,
            "validate",
            "Validate scaffolded project modules for syntax, imports, and compilation.",
        )
        try:
            working_state = _invoke_subgraph(_VALIDATE_SUBGRAPH, working_state)
        except Exception as exc:  # pragma: no cover - defensive
            error_message = f"Validation phase failed: {exc}"
            _finish_phase(
                working_state,
                validation_phase,
                validation_started,
                success=False,
                summary="Scaffold validation encountered an error.",
                details={"attempt": attempt},
                errors=[error_message],
            )
            _record_failure(working_state, error_message)
            return dict(working_state)

        errors = _collect_scaffold_errors(working_state)
        scaffold = _ensure_scaffold_container(working_state)
        ok = bool(scaffold.get("ok", not errors))
        scaffold["ok"] = ok

        _finish_phase(
            working_state,
            validation_phase,
            validation_started,
            success=ok,
            summary=(
                "Scaffold validation succeeded."
                if ok
                else "Scaffold validation detected issues."
            ),
            details={"attempt": attempt, "errors": list(errors)},
            errors=errors,
        )

        if ok:
            return dict(working_state)

        if attempt >= MAX_VALIDATION_ATTEMPTS:
            _record_failure(
                working_state,
                "Validation retry budget exhausted before repairs succeeded.",
            )
            return dict(working_state)

        repair_phase, repair_started = _start_phase(
            working_state,
            "repair",
            "Attempt to repair scaffold validation issues.",
        )
        try:
            working_state = _invoke_subgraph(_REPAIR_SUBGRAPH, working_state)
        except Exception as exc:  # pragma: no cover - defensive
            error_message = f"Repair phase failed: {exc}"
            _finish_phase(
                working_state,
                repair_phase,
                repair_started,
                success=False,
                summary="Scaffold repair encountered an error.",
                details={"attempt": attempt},
                errors=[error_message],
            )
            _record_failure(working_state, error_message)
            return dict(working_state)

        remaining_errors = _collect_scaffold_errors(working_state)
        scaffold = _ensure_scaffold_container(working_state)
        repairs_successful = not remaining_errors
        scaffold["ok"] = repairs_successful if repairs_successful else scaffold.get("ok", False)

        _finish_phase(
            working_state,
            repair_phase,
            repair_started,
            success=repairs_successful,
            summary=(
                "Repairs applied successfully."
                if repairs_successful
                else "Repairs applied; issues remain."
            ),
            details={"attempt": attempt, "errors": list(remaining_errors)},
            errors=remaining_errors,
        )


__all__ = ["scaffold_coordinator", "MAX_VALIDATION_ATTEMPTS"]

