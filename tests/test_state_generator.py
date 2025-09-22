from __future__ import annotations

import pytest

from asb.agent.state_generator import generate_state_schema, state_generator_node


def test_generate_state_schema_infers_types_from_state_flow():
    state = {
        "architecture": {
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
    assert "class AppState" in generated
    assert "architecture: Dict[str, Any]" in generated
    assert "analysis_results: Dict[str, Any]" in generated
    assert "task_steps: List[Any]" in generated
    assert "is_ready: bool" in generated


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
