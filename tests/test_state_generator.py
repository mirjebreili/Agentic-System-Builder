from __future__ import annotations

import re
from typing import ForwardRef

import pytest

from asb.agent.scaffold import STATE_TEMPLATE
from asb.agent.state import AppState
from asb.agent.state_generator import generate_state_schema, state_generator_node


EXPECTED_APP_STATE_FIELDS = {
    "messages": "Annotated[List[AnyMessage], add_messages]",
    "goal": "str",
    "input_text": "str",
    "plan": "Annotated[Dict[str, Any], operator.or_]",
    "architecture": "Annotated[Dict[str, Any], operator.or_]",
    "result": "str",
    "final_output": "str",
    "error": "str",
    "errors": "Annotated[List[str], operator.add]",
    "scratch": "Annotated[Dict[str, Any], operator.or_]",
    "scaffold": "Annotated[Dict[str, Any], operator.or_]",
    "sandbox": "Annotated[Dict[str, Any], operator.or_]",
    "self_correction": "Annotated[Dict[str, Any], operator.or_]",
    "tot": "Annotated[Dict[str, Any], operator.or_]",
}

EXPECTED_TEMPLATE_FIELDS = dict(EXPECTED_APP_STATE_FIELDS)


def _parse_app_state_fields(source: str) -> dict[str, str]:
    sentinel = "class AppState(TypedDict, total=False):"
    lines = source.splitlines()
    try:
        start_index = lines.index(sentinel)
    except ValueError:  # pragma: no cover - defensive guard for malformed templates
        assert False, "AppState definition not found"

    fields: dict[str, str] = {}
    for line in lines[start_index + 1 :]:
        stripped = line.strip()
        if not line.startswith("    "):
            if not stripped:
                continue
            break
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
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

    assert fields["analysis_results"] == "Dict[str, Any]"
    assert fields["requirements"] == "Dict[str, Any]"
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

    fields = _parse_app_state_fields(generated)
    assert fields["self_correction"] == "Annotated[Dict[str, Any], operator.or_]"


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
