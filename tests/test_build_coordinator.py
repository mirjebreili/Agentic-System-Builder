from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from asb.agent import build_coordinator


def _make_step(name: str, calls: List[str]) -> build_coordinator.StepCallable:
    def _step(state: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(name)
        return state

    return _step


def test_coordinate_build_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[str] = []
    base_sequence: Tuple[Tuple[str, build_coordinator.StepCallable], ...] = (
        ("context_collector", _make_step("context", calls)),
        ("requirements_analyzer", _make_step("requirements", calls)),
        ("architecture_designer", _make_step("architecture", calls)),
        ("state_generator", _make_step("state", calls)),
        ("node_implementor", _make_step("node", calls)),
        ("syntax_validator", _make_step("syntax", calls)),
        ("scaffold_project", _make_step("scaffold", calls)),
        ("state_schema_writer", _make_step("schema", calls)),
        ("skeleton_writer", _make_step("skeleton", calls)),
        ("import_resolver", _make_step("imports", calls)),
        ("logic_implementor", _make_step("logic", calls)),
        ("unit_test_writer", _make_step("tests", calls)),
    )
    monkeypatch.setattr(build_coordinator, "BASE_SEQUENCE", base_sequence)

    def sandbox_success(state: Dict[str, Any]) -> Dict[str, Any]:
        calls.append("sandbox")
        new_state = dict(state)
        new_state["sandbox"] = {"ok": True, "history": []}
        return new_state

    monkeypatch.setattr(build_coordinator, "sandbox_runner_node", sandbox_success)

    def report_stub(state: Dict[str, Any]) -> Dict[str, Any]:
        calls.append("report")
        state["report_called"] = True
        return state

    monkeypatch.setattr(build_coordinator, "report", report_stub)

    result = build_coordinator.coordinate_build({"messages": []})

    assert result["coordinator_decision"] == "proceed"
    assert result["next_action"] == "scaffold"
    assert result["consecutive_failures"] == 0
    assert result["report_called"] is True
    assert "sandbox" in calls
    trace_steps = [entry["step"] for entry in result["debug"]["build_coordinator"]["trace"]]
    assert trace_steps[: len(base_sequence)] == [name for name, _ in base_sequence]


def test_coordinate_build_triggers_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[str] = []
    repair_calls: List[str] = []

    base_sequence: Tuple[Tuple[str, build_coordinator.StepCallable], ...] = (
        ("context_collector", _make_step("context", calls)),
        ("scaffold_project", _make_step("scaffold", calls)),
    )
    monkeypatch.setattr(build_coordinator, "BASE_SEQUENCE", base_sequence)

    def sandbox_flaky(state: Dict[str, Any]) -> Dict[str, Any]:
        calls.append("sandbox")
        attempt = len([c for c in calls if c == "sandbox"])
        sandbox_state = {"ok": attempt > 1, "history": []}
        new_state = dict(state)
        new_state["sandbox"] = sandbox_state
        return new_state

    monkeypatch.setattr(build_coordinator, "sandbox_runner_node", sandbox_flaky)

    repair_sequence: Tuple[Tuple[str, build_coordinator.StepCallable], ...] = (
        ("bug_localizer", _make_step("bug_localizer", repair_calls)),
        ("tot_variant_maker", _make_step("tot", repair_calls)),
        ("critic_judge", _make_step("judge", repair_calls)),
        ("diff_patcher", _make_step("patch", repair_calls)),
    )
    monkeypatch.setattr(build_coordinator, "REPAIR_SEQUENCE", repair_sequence)
    monkeypatch.setattr(build_coordinator, "report", lambda s: s)

    result = build_coordinator.coordinate_build({})

    assert result["coordinator_decision"] == "proceed"
    assert any(call == "bug_localizer" for call in repair_calls)
    trace = result["debug"]["build_coordinator"]["trace"]
    assert any(entry["phase"].startswith("repair") for entry in trace)


def test_coordinate_build_exhausts_repair_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    base_sequence: Tuple[Tuple[str, build_coordinator.StepCallable], ...] = (
        ("context_collector", lambda s: s),
        ("scaffold_project", lambda s: s),
    )
    monkeypatch.setattr(build_coordinator, "BASE_SEQUENCE", base_sequence)

    def sandbox_fail(state: Dict[str, Any]) -> Dict[str, Any]:
        new_state = dict(state)
        new_state["sandbox"] = {"ok": False, "history": []}
        return new_state

    monkeypatch.setattr(build_coordinator, "sandbox_runner_node", sandbox_fail)

    def repair_step(state: Dict[str, Any]) -> Dict[str, Any]:
        return state

    repair_sequence: Tuple[Tuple[str, build_coordinator.StepCallable], ...] = (
        ("bug_localizer", repair_step),
        ("tot_variant_maker", repair_step),
        ("critic_judge", repair_step),
        ("diff_patcher", repair_step),
    )
    monkeypatch.setattr(build_coordinator, "REPAIR_SEQUENCE", repair_sequence)
    monkeypatch.setattr(build_coordinator, "report", lambda s: s)

    result = build_coordinator.coordinate_build({})

    assert result["coordinator_decision"] == "halt"
    assert result["consecutive_failures"] >= 1
    assert "sandbox_validation_failed" in result.get("errors", [])
    trace = result["debug"]["build_coordinator"]["trace"]
    sandbox_retries = [entry for entry in trace if entry["phase"].startswith("sandbox_retry")]
    assert len(sandbox_retries) == build_coordinator.MAX_REPAIR_ATTEMPTS
