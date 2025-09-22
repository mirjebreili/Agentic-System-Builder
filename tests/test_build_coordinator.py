from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from asb.agent import build_coordinator


def _make_sequence(steps: List[Tuple[str, Any]]) -> Tuple[Tuple[str, Any], ...]:
    return tuple((name, func) for name, func in steps)


def test_coordinate_build_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[str] = []

    def make_step(name: str):
        def _step(state: Dict[str, Any]) -> Dict[str, Any]:
            calls.append(name)
            state.setdefault("messages", [])
            return state

        return _step

    def syntax_success(state: Dict[str, Any]) -> Dict[str, Any]:
        calls.append("syntax_validator")
        state["next_action"] = "complete"
        state["validation_errors"] = []
        return state

    sequence = _make_sequence(
        [
            ("requirements_analyzer", make_step("requirements")),
            ("architecture_designer", make_step("architecture")),
            ("state_generator", make_step("state")),
            ("node_implementor", make_step("implementor")),
            ("syntax_validator", syntax_success),
            ("scaffold_project", make_step("scaffold")),
            ("sandbox_smoke", make_step("sandbox")),
            ("report", make_step("report")),
        ]
    )
    monkeypatch.setattr(build_coordinator, "BUILD_SEQUENCE", sequence)

    initial_state: Dict[str, Any] = {"messages": []}
    result = build_coordinator.coordinate_build(initial_state)

    assert result["coordinator_decision"] == "proceed"
    assert result["next_action"] == "scaffold"
    assert result["build_attempts"] == 1
    assert result["consecutive_failures"] == 0
    trace_steps = [entry["step"] for entry in result["debug"]["build_coordinator"]["trace"]]
    assert "requirements_analyzer" in trace_steps
    assert "report" in trace_steps


def test_coordinate_build_retries_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    validation_calls: List[str] = []
    fixer_calls: List[int] = []

    def syntax_retry(state: Dict[str, Any]) -> Dict[str, Any]:
        validation_calls.append("run")
        attempt = len(validation_calls)
        state["next_action"] = "fix_code" if attempt == 1 else "complete"
        return state

    def fixer(state: Dict[str, Any]) -> Dict[str, Any]:
        fixer_calls.append(len(fixer_calls))
        state["next_action"] = "validate_again"
        state["fix_attempts"] = state.get("fix_attempts", 0) + 1
        return state

    sequence = _make_sequence(
        [
            ("requirements_analyzer", lambda s: s),
            ("architecture_designer", lambda s: s),
            ("state_generator", lambda s: s),
            ("node_implementor", lambda s: s),
            ("syntax_validator", syntax_retry),
            ("scaffold_project", lambda s: s),
            ("sandbox_smoke", lambda s: s),
            ("report", lambda s: s),
        ]
    )
    monkeypatch.setattr(build_coordinator, "BUILD_SEQUENCE", sequence)
    monkeypatch.setattr(build_coordinator, "syntax_validator_node", syntax_retry)
    monkeypatch.setattr(build_coordinator, "code_fixer_node", fixer)

    result = build_coordinator.coordinate_build({})

    assert result["coordinator_decision"] == "proceed"
    assert result["consecutive_failures"] == 0
    assert len(validation_calls) == 2
    assert len(fixer_calls) == 1
    trace = result["debug"]["build_coordinator"]["trace"]
    assert any(entry["step"] == "code_fixer" for entry in trace)


def test_coordinate_build_circuit_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    def syntax_always_fail(state: Dict[str, Any]) -> Dict[str, Any]:
        state["next_action"] = "fix_code"
        return state

    def fixer(state: Dict[str, Any]) -> Dict[str, Any]:
        state["next_action"] = "validate_again"
        state["fix_attempts"] = state.get("fix_attempts", 0) + 1
        return state

    def scaffold(state: Dict[str, Any]) -> Dict[str, Any]:
        pytest.fail("scaffold step should not run after circuit breaker")

    sequence = _make_sequence(
        [
            ("requirements_analyzer", lambda s: s),
            ("architecture_designer", lambda s: s),
            ("state_generator", lambda s: s),
            ("node_implementor", lambda s: s),
            ("syntax_validator", syntax_always_fail),
            ("scaffold_project", scaffold),
            ("sandbox_smoke", lambda s: s),
            ("report", lambda s: s),
        ]
    )
    monkeypatch.setattr(build_coordinator, "BUILD_SEQUENCE", sequence)
    monkeypatch.setattr(build_coordinator, "syntax_validator_node", syntax_always_fail)
    monkeypatch.setattr(build_coordinator, "code_fixer_node", fixer)

    result = build_coordinator.coordinate_build({})

    assert result["coordinator_decision"] == "force_complete"
    assert result["next_action"] == "force_complete"
    assert result["consecutive_failures"] >= 1
    trace = result["debug"]["build_coordinator"]["trace"]
    assert any(entry.get("reason") == "syntax_validation_retry_budget_exceeded" for entry in trace)
