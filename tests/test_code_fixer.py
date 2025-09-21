from __future__ import annotations

import json

from asb.agent.code_fixer import CodeFixer, code_fixer_node


def test_code_fixer_updates_paths(tmp_path):
    project = tmp_path / "project"
    project.mkdir()

    langgraph = {
        "graphs": {"agent": "agent.graph:graph"},
        "dependencies": ["."],
        "env": "./.env",
    }
    (project / "langgraph.json").write_text(json.dumps(langgraph), encoding="utf-8")

    tests_dir = project / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_smoke.py").write_text(
        "from agent.graph import graph\n\n", encoding="utf-8"
    )

    state: dict[str, object] = {
        "scaffold": {"path": str(project)},
        "code_validation": {
            "import_check": {"success": False},
        },
    }

    result = code_fixer_node(state)

    assert result["next_action"] == "validate_again"
    fixes = result.get("code_fixes", {})
    assert fixes.get("success") is True

    updated_config = json.loads((project / "langgraph.json").read_text(encoding="utf-8"))
    assert updated_config["graphs"]["agent"] == "src.agent.graph:graph"

    updated_test = (tests_dir / "test_smoke.py").read_text(encoding="utf-8")
    assert "from src.agent.graph import graph" in updated_test


def test_code_fixer_force_completes_after_repeated_attempts(tmp_path):
    project = tmp_path / "project"
    project.mkdir()

    langgraph = {
        "graphs": {"agent": "agent.graph:graph"},
        "dependencies": ["."],
        "env": "./.env",
    }
    (project / "langgraph.json").write_text(json.dumps(langgraph), encoding="utf-8")

    tests_dir = project / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_smoke.py").write_text(
        "from agent.graph import graph\n\n", encoding="utf-8"
    )

    state: dict[str, object] = {
        "scaffold": {"path": str(project)},
        "code_validation": {
            "import_check": {"success": False},
        },
    }

    for attempt in range(CodeFixer.MAX_FIX_ATTEMPTS - 1):
        result = code_fixer_node(state)
        assert result["next_action"] == "validate_again"
        assert result.get("fix_attempts") == attempt + 1

    final_result = code_fixer_node(state)
    assert final_result["next_action"] == "force_complete"
    assert not final_result["code_fixes"]["success"]
    assert (
        "Max fix attempts exceeded - forcing completion"
        in final_result["code_fixes"]["errors"][0]
    )
    assert final_result["fix_attempts"] == CodeFixer.MAX_FIX_ATTEMPTS
