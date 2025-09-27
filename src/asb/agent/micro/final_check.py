from __future__ import annotations

import contextlib
import json
import os
import pathlib
import subprocess
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Iterable, Tuple

import requests

_PORT = "3000"
BASE_URL = f"http://127.0.0.1:{_PORT}"
DOCS_URL = f"{BASE_URL}/docs"
RUNS_WAIT_URL = f"{BASE_URL}/runs/wait"

_LOG_CHAR_LIMIT = 4096


def _read_assistant_id(project_root: pathlib.Path) -> str:
    config_path = project_root / "langgraph.json"
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return "agent"

    graphs = data.get("graphs") if isinstance(data, dict) else {}
    if isinstance(graphs, dict) and graphs:
        return str(next(iter(graphs.keys())))
    return "agent"


def _start_server(project_root: pathlib.Path) -> Tuple[subprocess.Popen[str], Deque[str], threading.Lock, threading.Thread]:
    process = subprocess.Popen(
        ["langgraph", "dev", "--port", _PORT],
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    buffer: Deque[str] = deque()
    lock = threading.Lock()

    def _reader() -> None:
        if not process.stdout:
            return
        for line in process.stdout:
            with lock:
                buffer.append(line)
                while sum(len(item) for item in buffer) > _LOG_CHAR_LIMIT * 2:
                    buffer.popleft()

    thread = threading.Thread(target=_reader, name="langgraph-dev-final-check", daemon=True)
    thread.start()
    return process, buffer, lock, thread


def _logs_tail(buffer: Iterable[str], lock: threading.Lock, limit: int = _LOG_CHAR_LIMIT) -> str:
    with lock:
        combined = "".join(buffer)
    if len(combined) > limit:
        return combined[-limit:]
    return combined


def _wait_for_docs(timeout_s: float = 30.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            response = requests.get(DOCS_URL, timeout=2)
            if response.status_code < 500:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.5)
    return False


def _headers_with_optional_key() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    api_key = os.getenv("LANGGRAPH_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def _terminate_process(process: subprocess.Popen[str] | None, thread: threading.Thread | None) -> None:
    if process is None:
        return
    try:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    except Exception:
        with contextlib.suppress(Exception):
            process.kill()
    finally:
        if process.stdout:
            try:
                process.stdout.close()
            except Exception:
                pass
    if thread and thread.is_alive():
        thread.join(timeout=1)


def _response_contains_pong(text: str) -> bool:
    return "PONG" in text.upper()


def final_check_node(state: Dict[str, Any]) -> Dict[str, Any]:
    scaffold = state.get("scaffold") or {}
    project_root_str = scaffold.get("project_root")
    if not project_root_str:
        state["final_check"] = {"ok": False, "phase": "start", "error": "missing project_root"}
        return state

    project_root = pathlib.Path(str(project_root_str))
    assistant_id = _read_assistant_id(project_root)

    try:
        process, buffer, lock, thread = _start_server(project_root)
    except OSError as exc:
        state["final_check"] = {
            "ok": False,
            "phase": "start",
            "error": f"failed to launch langgraph dev: {exc}",
            "assistant_id": assistant_id,
        }
        return state

    try:
        if not _wait_for_docs():
            state["final_check"] = {
                "ok": False,
                "phase": "start",
                "error": "server not ready",
                "assistant_id": assistant_id,
                "logs_tail": _logs_tail(buffer, lock),
            }
            return state

        probe = (state.get("acceptance") or {}).get("probe_prompt") or "Reply with the single word: PONG"
        payload = {
            "assistant_id": assistant_id,
            "input": {"messages": [{"role": "human", "content": probe}]},
        }
        headers = _headers_with_optional_key()

        try:
            response = requests.post(RUNS_WAIT_URL, json=payload, headers=headers, timeout=30)
        except requests.RequestException as exc:
            state["final_check"] = {
                "ok": False,
                "phase": "probe",
                "assistant_id": assistant_id,
                "error": f"runs/wait request failed: {exc}",
                "logs_tail": _logs_tail(buffer, lock),
            }
            return state

        body_text = response.text
        snippet = body_text[:400]
        success = response.status_code < 400 and _response_contains_pong(body_text)
        result = {
            "ok": bool(success),
            "phase": "probe",
            "assistant_id": assistant_id,
            "status": response.status_code,
            "snippet": snippet,
        }
        if not success:
            result["error"] = "missing expected response token" if response.status_code < 400 else f"HTTP {response.status_code}"
            result["logs_tail"] = _logs_tail(buffer, lock)
        state["final_check"] = result
        return state
    finally:
        _terminate_process(process, thread)


__all__ = [
    "final_check_node",
    "BASE_URL",
    "DOCS_URL",
    "RUNS_WAIT_URL",
    "_headers_with_optional_key",
    "_logs_tail",
    "_read_assistant_id",
    "_start_server",
    "_terminate_process",
    "_wait_for_docs",
]
