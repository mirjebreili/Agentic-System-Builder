from __future__ import annotations

import pytest

from asb.agent.architecture_designer import (
    architecture_designer_node,
    design_architecture,
)


def test_design_architecture_uses_requirements():
    state = {
        "requirements": {
            "nodes_needed": ["analyze", "design"],
            "state_schema": {"architecture": "object"},
        }
    }

    updated = design_architecture(state)

    architecture = updated.get("architecture")
    assert architecture is not None
    assert architecture["graph_structure"][0]["id"] == "entry"
    assert architecture["entry_exit_points"]["entry"] == ["entry"]
    assert "architecture" in architecture["state_flow"]


def test_architecture_designer_node_appends_summary():
    state = {
        "messages": [{"role": "user", "content": "Design a workflow"}],
        "requirements": {"nodes_needed": ["x"]},
    }

    updated = architecture_designer_node(state)

    assert "architecture" in updated
    assert updated["messages"][-1]["role"] == "assistant"
    assert "[architecture]" in updated["messages"][-1]["content"]


def test_architecture_designer_node_handles_exception(monkeypatch: pytest.MonkeyPatch):
    def _boom(state):  # pragma: no cover - minimal fallback path
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "asb.agent.architecture_designer.design_architecture",
        _boom,
    )

    state = {"messages": [], "requirements": {}}
    updated = architecture_designer_node(state)

    assert updated["architecture"]["graph_structure"] == []
    assert updated["messages"][-1]["content"].startswith("[architecture-error]")
