from __future__ import annotations

from asb.agent.code_validator import CodeValidator, code_validator_node


def test_code_validator_node_missing_path(tmp_path):
    missing_dir = tmp_path / "missing"
    state: dict[str, object] = {"scaffold": {"path": str(missing_dir)}}

    result = code_validator_node(state)

    assert result["next_action"] == "fix_code"
    validation = result.get("code_validation", {})
    assert validation.get("success") is False
    assert "project path not found" in result.get("validation_errors", [])


def test_code_validator_node_success(monkeypatch, tmp_path):
    project = tmp_path / "project"
    for relative in [
        "langgraph.json",
        "pyproject.toml",
        "README.md",
        "src/agent/graph.py",
        "src/agent/planner.py",
        "src/agent/executor.py",
    ]:
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")

    state: dict[str, object] = {"scaffold": {"path": str(project)}}

    for attr in (
        "_check_project_structure",
        "_check_dependencies",
        "_check_imports",
        "_check_langgraph_compatibility",
        "_check_runtime_execution",
    ):
        monkeypatch.setattr(CodeValidator, attr, lambda self, path, attr=attr: (True, {"message": attr}))

    result = code_validator_node(state)

    assert result["next_action"] == "complete"
    validation = result.get("code_validation", {})
    assert validation.get("overall_success") is True
    assert not validation.get("errors")
