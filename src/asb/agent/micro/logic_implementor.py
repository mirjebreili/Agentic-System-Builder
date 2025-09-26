"""Fill in deterministic logic for generated node skeletons."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from asb.utils.fileops import atomic_write

_SENTINEL = "    raise NotImplementedError(\"node logic not yet implemented\")\n\n    return state"


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


def _iter_node_modules(agent_dir: Path):
    for path in agent_dir.glob("*.py"):
        if path.name in {"__init__.py", "state.py"}:
            continue
        yield path


def _implement_logic(source: str, module_name: str) -> str:
    replacement = (
        "    response = AIMessage(content=\"{name} step completed.\")\n"
        "    updated_messages = messages + [response]\n"
        "    updated_state = dict(state)\n"
        "    scratch = dict(updated_state.get(\"scratch\") or {})\n"
        "    completed = list(scratch.get(\"completed_nodes\") or [])\n"
        f"    if \"{module_name}\" not in completed:\n"
        "        completed.append(\"{module_name}\")\n"
        "    scratch[\"completed_nodes\"] = completed\n"
        "    scratch[\"last_node\"] = \"{module_name}\"\n"
        "    updated_state[\"scratch\"] = scratch\n"
        "    updated_state[\"messages\"] = updated_messages\n"
        "    return updated_state\n"
    ).format(name=module_name)
    return source.replace(_SENTINEL, replacement)


def logic_implementor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Replace placeholder raise statements with runnable logic."""

    working_state: Dict[str, Any] = dict(state or {})
    project_root = _determine_project_root(working_state)
    if project_root is None:
        return working_state

    agent_dir = project_root / "src" / "agent"
    if not agent_dir.exists():
        return working_state

    updated_files: list[str] = []
    for module_path in _iter_node_modules(agent_dir):
        contents = module_path.read_text(encoding="utf-8")
        if _SENTINEL not in contents:
            continue
        implemented = _implement_logic(contents, module_path.stem)
        if implemented != contents:
            atomic_write(module_path, implemented)
            updated_files.append(str(module_path))

    if updated_files:
        scratch = dict(working_state.get("scratch") or {})
        scratch.setdefault("artifacts", {})["implemented_nodes"] = updated_files
        working_state["scratch"] = scratch
    return working_state


__all__ = ["logic_implementor_node"]
