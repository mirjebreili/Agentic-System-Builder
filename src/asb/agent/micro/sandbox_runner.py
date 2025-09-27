"""Run smoke tests (langgraph + pytest) against generated projects."""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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
            "logs": {
                "stdout": {"text": "", "lines": []},
                "stderr": {"text": str(exc), "lines": [str(exc)]},
            },
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
            "logs": {
                "stdout": {
                    "text": exc.stdout or "",
                    "lines": (exc.stdout or "").splitlines(),
                },
                "stderr": {
                    "text": (exc.stderr or "") + "\n[timeout]",
                    "lines": ((exc.stderr or "") + "\n[timeout]").splitlines(),
                },
            },
            "status": "timeout",
            "started": started,
            "completed": finished,
            "duration": max(0.0, finished - started),
        }
    finished = time.time()
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    return {
        "command": list(command),
        "cwd": str(cwd) if cwd else None,
        "returncode": completed.returncode,
        "logs": {
            "stdout": {"text": stdout, "lines": stdout.splitlines()},
            "stderr": {"text": stderr, "lines": stderr.splitlines()},
        },
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

    commands: List[Dict[str, Any]] = [
        {
            "name": "meta_langgraph",
            "command": [
                "python",
                "-c",
                (
                    "import ast, pathlib; "
                    "[ast.parse(path.read_text(encoding='utf-8'), filename=str(path)) "
                    "for path in pathlib.Path('src/asb/agent').rglob('*.py')]"
                ),
            ],
            "cwd": _REPO_ROOT,
        }
    ]
    if project_root is not None:
        commands.append(
            {
                "name": "project_langgraph",
                "command": [
                    "python",
                    "-c",
                    (
                        "import ast, pathlib; "
                        "root = pathlib.Path('src/agent'); "
                        "[ast.parse(path.read_text(encoding='utf-8'), filename=str(path)) "
                        "for path in root.rglob('*.py') if path.is_file()]"
                    ),
                ],
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
    aggregated_errors: List[Dict[str, Any]] = []
    return_codes: Dict[str, Optional[int]] = {}
    logs: Dict[str, Dict[str, Any]] = {}
    statuses: Dict[str, str] = {}

    for spec in commands:
        result = _run_command(spec["command"], spec.get("cwd"))
        name = spec["name"]
        result["name"] = name
        run_results.append(result)

        return_codes[name] = result.get("returncode")
        logs[name] = result.get("logs", {})
        statuses[name] = result.get("status", "unknown")

        aggregated_errors.extend(
            _collect_errors(
                name,
                logs[name],
                statuses[name],
                result.get("returncode"),
                project_root,
            )
        )

    summary = {
        "cmds": [
            {
                "name": entry.get("name"),
                "argv": entry.get("command"),
                "cwd": entry.get("cwd"),
                "status": entry.get("status"),
                "duration": entry.get("duration"),
            }
            for entry in run_results
        ],
        "return_codes": return_codes,
        "logs": logs,
        "errors": aggregated_errors,
        "statuses": statuses,
        "timestamp": time.time(),
    }

    history.append(summary)

    sandbox["history"] = history[-10:]
    sandbox["last_run_summary"] = summary
    sandbox["last_run"] = summary.get("cmds", [])
    sandbox["ok"] = all(status == "ok" for status in statuses.values())

    working_state["sandbox"] = sandbox
    return working_state


_TRACE_PATTERN = re.compile(r'File "(?P<path>.+?)", line (?P<line>\d+)')


def _collect_errors(
    name: str,
    log_data: Dict[str, Any],
    status: str,
    returncode: Optional[int],
    project_root: Path | None,
) -> List[Dict[str, Any]]:
    if status == "ok" and (returncode == 0 or returncode is None):
        return []

    stderr_block = ""
    if isinstance(log_data, dict):
        stderr_entry = log_data.get("stderr")
        if isinstance(stderr_entry, dict):
            stderr_block = str(stderr_entry.get("text") or "")
        elif isinstance(stderr_entry, str):
            stderr_block = stderr_entry

    errors: List[Dict[str, Any]] = []
    matches = list(_TRACE_PATTERN.finditer(stderr_block)) if stderr_block else []
    message = _derive_message(stderr_block, status)

    if matches:
        for match in matches:
            normalized = _normalize_path(match.group("path"), project_root)
            line_no = int(match.group("line"))
            errors.append(
                {
                    "command": name,
                    "file": normalized,
                    "line": line_no,
                    "message": message,
                }
            )
    elif message:
        errors.append(
            {
                "command": name,
                "file": None,
                "line": None,
                "message": message,
            }
        )

    return errors


def _derive_message(stderr_block: str, status: str) -> str:
    text = stderr_block.strip()
    if text:
        return text.splitlines()[-1]
    return status


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


__all__ = ["sandbox_runner_node"]
