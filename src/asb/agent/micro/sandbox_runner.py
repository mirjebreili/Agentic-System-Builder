"""Run smoke tests (langgraph + pytest) against generated projects."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

_REPO_ROOT = Path(__file__).resolve().parents[3]


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


def _run_command(command: Iterable[str], cwd: Path | None, timeout: int = 60) -> Dict[str, Any]:
    started = time.time()
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        finished = time.time()
        return {
            "command": list(command),
            "cwd": str(cwd) if cwd else None,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "status": "not_found",
            "started": started,
            "completed": finished,
            "duration": max(0.0, finished - started),
        }
    except subprocess.TimeoutExpired as exc:
        finished = time.time()
        return {
            "command": list(command),
            "cwd": str(cwd) if cwd else None,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + "\n[timeout]",
            "status": "timeout",
            "started": started,
            "completed": finished,
            "duration": max(0.0, finished - started),
        }
    finished = time.time()
    return {
        "command": list(command),
        "cwd": str(cwd) if cwd else None,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "status": "ok" if completed.returncode == 0 else "failed",
        "started": started,
        "completed": finished,
        "duration": max(0.0, finished - started),
    }


def sandbox_runner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute sandbox smoke checks and capture results on the state."""

    working_state: Dict[str, Any] = dict(state or {})
    project_root = _determine_project_root(working_state)

    sandbox = dict(working_state.get("sandbox") or {})
    history: List[Dict[str, Any]] = list(sandbox.get("history") or [])

    commands: List[Dict[str, Any]] = []
    commands.append(
        {
            "name": "meta_langgraph",
            "command": ["langgraph", "dev", "--check"],
            "cwd": _REPO_ROOT,
        }
    )
    if project_root is not None:
        commands.append(
            {
                "name": "project_langgraph",
                "command": ["langgraph", "dev", "--check"],
                "cwd": project_root,
            }
        )
        commands.append(
            {
                "name": "project_pytest",
                "command": ["pytest", "-q"],
                "cwd": project_root,
            }
        )

    run_results: List[Dict[str, Any]] = []
    for spec in commands:
        result = _run_command(spec["command"], spec.get("cwd"))
        result["name"] = spec["name"]
        run_results.append(result)
        history.append(result)

    sandbox["history"] = history[-10:]
    sandbox["last_run"] = run_results
    sandbox["ok"] = all(entry.get("status") == "ok" for entry in run_results)

    working_state["sandbox"] = sandbox
    return working_state


__all__ = ["sandbox_runner_node"]
