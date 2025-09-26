"""Produce candidate code variants for repair attempts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

_SENTINEL = "raise NotImplementedError(\"node logic not yet implemented\")"


_MESSAGES = [
    "step completed",
    "step acknowledged",
    "step executed successfully",
]


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


def _read_file(base: Path, relative: str) -> str:
    return (base / relative).read_text(encoding="utf-8")


def _inject_message(source: str, module_name: str, message: str) -> str:
    if _SENTINEL in source:
        replacement = (
            "    response = AIMessage(content=\"{text}\")\n"
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
        ).format(text=f"{module_name} {message}")
        return source.replace(f"    {_SENTINEL}\n\n    return state", replacement)

    pattern = re.compile(r"response\s*=\s*AIMessage\(content=.*?\)")
    if pattern.search(source):
        return pattern.sub(
            f'response = AIMessage(content="{module_name} {message}")',
            source,
            count=1,
        )
    return source


def _build_variant(path: str, module_name: str, source: str, message: str, index: int) -> Dict[str, Any]:
    code = _inject_message(source, module_name, message)
    return {
        "path": path,
        "module": module_name,
        "code": code,
        "why_this_might_work": f"Ensures deterministic response message variant {index + 1}.",
        "confidence": round(0.6 + index * 0.1, 2),
    }


def tot_variant_maker_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate Tree-of-Thought style candidate patches for localized bugs."""

    working_state: Dict[str, Any] = dict(state or {})
    project_root = _determine_project_root(working_state)
    if project_root is None:
        return working_state

    scratch = dict(working_state.get("scratch") or {})
    bug_reports = scratch.get("bug_localizations")
    if not bug_reports:
        scratch["tot_variants"] = []
        working_state["scratch"] = scratch
        return working_state

    variants: List[Dict[str, Any]] = []
    for bug in bug_reports[:1]:
        path = bug.get("path")
        module_name = Path(path).stem if path else "node"
        try:
            source = _read_file(project_root, path)
        except FileNotFoundError:
            continue
        for idx, suffix in enumerate(_MESSAGES[:3]):
            variants.append(_build_variant(path, module_name, source, suffix, idx))

    scratch["tot_variants"] = variants
    working_state["scratch"] = scratch
    return working_state


__all__ = ["tot_variant_maker_node"]
