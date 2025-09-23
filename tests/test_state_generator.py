from __future__ import annotations

import re
from typing import ForwardRef

import pytest

from asb.agent.scaffold import STATE_TEMPLATE
from asb.agent.state import AppState
from asb.agent.state_generator import generate_state_schema, state_generator_node


EXPECTED_APP_STATE_FIELDS = {
    "architecture": "Dict[str, Any]",
    "artifacts": "Dict[str, Any]",
    "build_attempts": "int",
    "code_fixes": "Dict[str, Any]",
    "code_validation": "Dict[str, Any]",
    "consecutive_failures": "int",
    "coordinator_decision": "str",
    "current_step": "Dict[str, bool]",
    "debug": "Dict[str, Any]",
    "error": "str",
    "evaluations": "List[Dict[str, Any]]",
    "fix_attempts": "int",
    "fix_strategy_used": "str | None",
    "flags": "Dict[str, bool]",
    "generated_files": "Dict[str, str]",
    "goal": "str",
    "implemented_nodes": "List[Dict[str, Any]]",
    "input_text": "str",
    "last_implemented_node": "str | None",
    "last_user_input": "str",
    "messages": "List[ChatMessage]",
    "metrics": "Dict[str, Any]",
    "next_action": "str",
    "passed": "bool",
    "plan": "Dict[str, Any]",
    "replan": "bool",
    "repair_start_time": "float",
    "report": "Dict[str, Any]",
    "requirements": "Dict[str, Any]",
    "review": "Dict[str, Any]",
    "sandbox": "Dict[str, Any]",
    "scaffold": "Dict[str, Any]",
    "self_correction": "SelfCorrectionState",
    "selected_thought": "Dict[str, Any]",
    "syntax_validation": "Dict[str, Any]",
    "tests": "Dict[str, Any]",
    "thoughts": "List[str]",
    "tot": "Dict[str, Any]",
    "validation_errors": "List[str]",
}

EXPECTED_TEMPLATE_FIELDS = dict(EXPECTED_APP_STATE_FIELDS)
EXPECTED_TEMPLATE_FIELDS["messages"] = "Annotated[List[AnyMessage], add_messages]"
EXPECTED_TEMPLATE_FIELDS.pop("self_correction", None)


def _parse_app_state_fields(source: str) -> dict[str, str]:
    match = re.search(
        r"class AppState\(TypedDict, total=False\):\n((?:    .+\n)+)", source
    )
    assert match, "AppState definition not found"
    body = match.group(1)
    fields: dict[str, str] = {}
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        name, annotation = stripped.split(":", 1)
        fields[name.strip()] = annotation.strip()
    return fields


def test_app_state_typed_dict_matches_expected_fields():
    normalized = {}
    for field, annotation in AppState.__annotations__.items():
        if isinstance(annotation, str):
            normalized[field] = annotation
        elif isinstance(annotation, ForwardRef):
            normalized[field] = annotation.__forward_arg__
        else:
            normalized[field] = repr(annotation).replace("typing.", "")

    assert normalized == EXPECTED_APP_STATE_FIELDS


def test_state_template_embeds_expected_schema():
    fields = _parse_app_state_fields(STATE_TEMPLATE)
    assert fields == EXPECTED_TEMPLATE_FIELDS


def test_generate_state_schema_infers_types_from_state_flow():
    state = {
        "architecture": {
            "plan": {
                "nodes": [
                    {"name": "Plan"},
                    {"name": "Do"},
                    {"name": "Finish"},
                ]
            },
            "state_flow": {
                "requirements": "entry -> analyze",
                "architecture": "design -> finish",
                "analysis_results": "design -> finish",
                "task_steps": "plan -> execute",
                "is_ready": "review -> deploy",
            }
        }
    }

    updated = generate_state_schema(state)

    generated = updated.get("generated_files", {}).get("state.py", "")
    fields = _parse_app_state_fields(generated)

    for field, annotation in EXPECTED_TEMPLATE_FIELDS.items():
        assert fields[field] == annotation

    assert fields["plan_result"] == "Dict[str, Any]"
    assert fields["do_result"] == "Dict[str, Any]"
    assert fields["finish_result"] == "Dict[str, Any]"
    assert fields["analysis_results"] == "Dict[str, Any]"
    assert fields["task_steps"] == "List[Any]"
    assert fields["is_ready"] == "bool"


def test_generate_state_schema_defaults_to_expected_fields_when_sparse():
    generated = generate_state_schema({}).get("generated_files", {}).get("state.py", "")
    fields = _parse_app_state_fields(generated)
    assert fields == EXPECTED_TEMPLATE_FIELDS


def test_generate_state_schema_adds_self_correction_fields_for_pattern():
    state = {
        "architecture": {
            "plan": {
                "workflow_pattern": "self_correcting_generation",
                "nodes": [
                    {"name": "Generate"},
                    {"name": "Validate"},
                    {"name": "Correct"},
                ],
            }
        }
    }

    generated = generate_state_schema(state).get("generated_files", {}).get("state.py", "")

    assert "class SelfCorrectionState" in generated

    fields = _parse_app_state_fields(generated)
    assert fields["self_correction"] == "SelfCorrectionState"


def test_state_generator_node_appends_summary_message():
    state = {
        "messages": [{"role": "user", "content": "Please build"}],
        "architecture": {"state_flow": {"architecture": "design -> finish"}},
    }

    updated = state_generator_node(state)

    assert "state.py" in updated.get("generated_files", {})
    assert updated["messages"][-1]["role"] == "assistant"
    assert "[state-schema]" in updated["messages"][-1]["content"]
    assert "architecture" in updated["messages"][-1]["content"]


def test_state_generator_node_handles_failure(monkeypatch: pytest.MonkeyPatch):
    def _boom(state):  # pragma: no cover - explicit failure path
        raise RuntimeError("boom")

    monkeypatch.setattr("asb.agent.state_generator.generate_state_schema", _boom)

    state = {"messages": []}
    updated = state_generator_node(state)

    assert updated["messages"][-1]["content"].startswith("[state-schema-error]")
