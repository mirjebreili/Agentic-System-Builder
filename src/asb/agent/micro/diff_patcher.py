"""Apply minimal diffs selected by the critic judge."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Dict, List

from asb.utils.fileops import atomic_write


def _determine_project_root(state: Dict[str, Any]) -> Path | None:
    scaffold = state.get("scaffold")
    if isinstance(scaffold, dict):
        path = scaffold.get("path")
        if path:
            return Path(path)
    candidate = state.get("project_root")
    if candidate:
        return Path(str(candidate))
    return None


def _apply_variant(base: Path, variant: Dict[str, Any]) -> Dict[str, Any] | None:
    path = variant.get("path")
    code = variant.get("code")
    if not isinstance(path, str) or not isinstance(code, str):
        return None

    file_path = base / path
    original = file_path.read_text(encoding="utf-8") if file_path.exists() else ""
    if original == code:
        return {"path": str(file_path), "diff": [], "changed": False}

    original_lines = original.splitlines()
    updated_lines = code.splitlines()
    diff = list(
        difflib.unified_diff(
            original_lines,
            updated_lines,
            fromfile=str(file_path),
            tofile=str(file_path),
            lineterm="",
        )
    )

    atomic_write(file_path, code)
    return {"path": str(file_path), "diff": diff, "changed": True}


def diff_patcher_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Persist the highest scoring variant to disk via unified diff."""

    working_state: Dict[str, Any] = dict(state or {})
    project_root = _determine_project_root(working_state)
    if project_root is None:
        return working_state

    scratch = dict(working_state.get("scratch") or {})
    selected: List[Dict[str, Any]] = list(scratch.get("selected_variants") or [])
    if not selected:
        scratch["applied_variants"] = []
        working_state["scratch"] = scratch
        return working_state

    applied: List[Dict[str, Any]] = []
    for variant in selected:
        result = _apply_variant(project_root, variant)
        if result is not None:
            applied.append(result)

    scratch["applied_variants"] = applied
    working_state["scratch"] = scratch
    return working_state


__all__ = ["diff_patcher_node"]
