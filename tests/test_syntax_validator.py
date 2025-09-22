from __future__ import annotations

from asb.agent.syntax_validator import syntax_validator_node, validate_syntax_only


def test_validate_syntax_only_success():
    state: dict[str, object] = {
        "generated_files": {
            "module.py": "def foo() -> int:\n    return 1\n",
        }
    }

    validate_syntax_only(state)

    validation = state.get("syntax_validation", {})
    assert validation.get("success") is True
    assert not validation.get("errors")
    file_result = validation.get("files", {}).get("module.py", {})
    assert file_result.get("success") is True
    assert file_result.get("message") == "Syntax valid"


def test_validate_syntax_only_reports_syntax_errors():
    state: dict[str, object] = {
        "generated_files": {
            "module.py": "def broken(:\n    pass\n",
        }
    }

    validate_syntax_only(state)

    validation = state.get("syntax_validation", {})
    assert validation.get("success") is False
    assert validation.get("errors")
    file_result = validation.get("files", {}).get("module.py", {})
    assert file_result.get("success") is False
    assert "Syntax error" in file_result.get("message", "")
    assert file_result.get("lineno") == 1


def test_syntax_validator_node_honors_fix_attempts():
    state: dict[str, object] = {
        "generated_files": {
            "module.py": "def broken(:\n    pass\n",
        },
        "fix_attempts": 3,
    }

    result = syntax_validator_node(state)

    assert result["next_action"] == "force_complete"
    assert result.get("validation_errors")
    assert result.get("syntax_validation", {}).get("success") is False


def test_syntax_validator_node_success_path():
    state: dict[str, object] = {
        "generated_files": {
            "module.py": "def ok() -> None:\n    return None\n",
        }
    }

    result = syntax_validator_node(state)

    assert result["next_action"] == "complete"
    assert result.get("validation_errors") == []
    assert result.get("syntax_validation", {}).get("success") is True
