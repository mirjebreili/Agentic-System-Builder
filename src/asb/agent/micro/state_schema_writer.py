"""Ensure the generated project exposes a consistent AppState schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from asb.agent.scaffold import generate_enhanced_state_schema
from asb.utils.fileops import atomic_write, ensure_dir


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


def state_schema_writer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Write the canonical state.py template into the generated project."""

    working_state: Dict[str, Any] = dict(state or {})
    project_root = _determine_project_root(working_state)
    if project_root is None:
        return working_state

    target = project_root / "src" / "agent" / "state.py"
    ensure_dir(target.parent)

    architecture_plan = working_state.get("architecture")
    if isinstance(architecture_plan, dict):
        schema = generate_enhanced_state_schema(architecture_plan)
    else:
        schema = generate_enhanced_state_schema({})

    current = target.read_text(encoding="utf-8") if target.exists() else ""
    if current != schema:
        atomic_write(target, schema)

    scratch = dict(working_state.get("scratch") or {})
    scratch.setdefault("artifacts", {})["state_schema"] = str(target)
    working_state["scratch"] = scratch
    return working_state


__all__ = ["state_schema_writer_node"]
