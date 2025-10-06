from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def _get_scaffold_path(state: Dict[str, Any]) -> Optional[Path]:
    """Return the scaffold path if provided and non-empty."""

    scaffold = state.get("scaffold") or {}
    path_value = scaffold.get("path")
    if isinstance(path_value, str) and path_value.strip():
        return Path(path_value)
    if isinstance(path_value, Path):
        return path_value
    return None


def report(state: Dict[str, Any]) -> Dict[str, Any]:
    """Persist a summary of the run when possible and store it in-state."""

    summary = {
        "goal": (state.get("plan") or {}).get("goal", ""),
        "plan": state.get("plan"),
        "confidence": (state.get("plan") or {}).get("confidence", None),
        "tests": state.get("tests"),
        "executor_passed": state.get("passed", False),
        "sandbox": state.get("sandbox"),
        "notes": "MVP run.",
    }

    scaffold_path = _get_scaffold_path(state)
    summary_path: Optional[Path] = None
    if scaffold_path is not None:
        reports_dir = scaffold_path / "reports"
        try:
            reports_dir.mkdir(parents=True, exist_ok=True)
            summary_path = reports_dir / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        except OSError:
            # Fall back to in-memory reporting when the file system is unavailable.
            summary_path = None

    report_payload: Dict[str, Any] = {"ok": True, "summary": summary}
    if summary_path is not None:
        report_payload["summary_path"] = str(summary_path)

    state["report"] = report_payload
    state.setdefault("final_response", json.dumps(summary, indent=2))
    return state
