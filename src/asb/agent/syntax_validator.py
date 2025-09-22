from __future__ import annotations

"""Lightweight syntax validation utilities for generated Agentic System Builder files."""

import ast
from typing import Any, Dict


def _format_syntax_error(exc: SyntaxError) -> Dict[str, Any]:
    message = exc.msg or "invalid syntax"
    location_parts = []
    if exc.lineno is not None:
        location_parts.append(f"line {exc.lineno}")
    if exc.offset is not None:
        location_parts.append(f"column {exc.offset}")
    location = f" ({', '.join(location_parts)})" if location_parts else ""
    details: Dict[str, Any] = {
        "success": False,
        "message": f"Syntax error: {message}{location}",
        "lineno": exc.lineno,
        "offset": exc.offset,
    }
    if exc.text is not None:
        details["source_line"] = exc.text.rstrip("\n")
    return details


def validate_syntax_only(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that generated Python files are syntactically correct.

    The validation reads Python sources from ``state["generated_files"]`` (when
    present) and records per-file diagnostics in ``state["syntax_validation"]``.
    Non-Python files are ignored.
    """

    validation_summary: Dict[str, Any] = {
        "success": True,
        "errors": [],
        "files": {},
    }

    generated_files = state.get("generated_files") or {}
    if not isinstance(generated_files, dict):
        message = "generated_files must be a mapping of file names to source strings"
        validation_summary.update({
            "success": False,
            "errors": [message],
        })
        state["syntax_validation"] = validation_summary
        return state

    python_files = {
        name: contents
        for name, contents in generated_files.items()
        if isinstance(name, str) and name.endswith(".py")
    }

    if not python_files:
        validation_summary["message"] = "No Python files to validate"
        state["syntax_validation"] = validation_summary
        return state

    for file_name, source in python_files.items():
        file_result: Dict[str, Any]
        if not isinstance(source, str):
            message = "Source contents must be a string"
            file_result = {
                "success": False,
                "message": f"{file_name}: {message}",
            }
            validation_summary["success"] = False
            validation_summary["errors"].append(file_result["message"])
        else:
            try:
                ast.parse(source, filename=file_name)
            except SyntaxError as exc:
                file_result = _format_syntax_error(exc)
                file_result["file"] = file_name
                validation_summary["success"] = False
                validation_summary["errors"].append(
                    f"{file_name}: {file_result['message']}"
                )
            except Exception as exc:  # pragma: no cover - defensive branch
                message = f"Unexpected failure during parsing: {exc}"
                file_result = {
                    "success": False,
                    "message": message,
                }
                validation_summary["success"] = False
                validation_summary["errors"].append(f"{file_name}: {message}")
            else:
                file_result = {"success": True, "message": "Syntax valid"}

        validation_summary["files"][file_name] = file_result

    validation_summary["message"] = (
        "Syntax validation passed"
        if validation_summary["success"]
        else "Syntax validation detected issues"
    )

    state["syntax_validation"] = validation_summary
    return state


def syntax_validator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Graph node wrapper around :func:`validate_syntax_only`."""

    fix_attempts = int(state.get("fix_attempts", 0)) if isinstance(state, dict) else 0

    validate_syntax_only(state)
    validation = state.get("syntax_validation", {})
    errors = validation.get("errors") or []

    state["validation_errors"] = errors

    if fix_attempts >= 3:
        state["next_action"] = "force_complete"
        return state

    state["next_action"] = "complete" if validation.get("success", True) else "fix_code"
    return state
