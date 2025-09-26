"""Parse sandbox outputs to locate failing files and spans."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

_TRACE_PATTERN = re.compile(r'File "(?P<path>.+?)", line (?P<line>\d+)', re.MULTILINE)


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


def _normalize_path(path: str, project_root: Path | None) -> str:
    candidate = Path(path)
    if candidate.is_absolute() and project_root is not None:
        try:
            return str(candidate.relative_to(project_root))
        except ValueError:
            return str(candidate)
    if project_root is not None:
        absolute = project_root / path
        if absolute.exists():
            return path
    return str(candidate)


def _parse_entry(entry: Dict[str, Any], project_root: Path | None) -> List[Dict[str, Any]]:
    stderr = entry.get("stderr") or ""
    matches = list(_TRACE_PATTERN.finditer(stderr))
    localizations: List[Dict[str, Any]] = []
    for match in matches:
        path = _normalize_path(match.group("path"), project_root)
        line = int(match.group("line"))
        message = stderr.strip().splitlines()[-1] if stderr else ""
        localizations.append(
            {
                "command": entry.get("name"),
                "path": path,
                "line_start": line,
                "line_end": line,
                "message": message,
            }
        )
    return localizations


def bug_localizer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract structured bug localization data from sandbox history."""

    working_state: Dict[str, Any] = dict(state or {})
    project_root = _determine_project_root(working_state)

    sandbox = working_state.get("sandbox")
    history = []
    if isinstance(sandbox, dict):
        history = list(sandbox.get("last_run") or sandbox.get("history") or [])

    localizations: List[Dict[str, Any]] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        status = entry.get("status")
        if status == "ok":
            continue
        localizations.extend(_parse_entry(entry, project_root))

    scratch = dict(working_state.get("scratch") or {})
    scratch["bug_localizations"] = localizations
    working_state["scratch"] = scratch
    return working_state


__all__ = ["bug_localizer_node"]
