from __future__ import annotations

import contextlib
import json
import os
import pathlib
import subprocess
import threading
import time
from typing import Any, Dict, Iterable, List, Tuple

import requests

DEFAULT_URL = os.getenv("LG_DEV_URL", "http://127.0.0.1:2024")
DOCS_URL = f"{DEFAULT_URL}/docs"
RUNS_WAIT_URL = f"{DEFAULT_URL}/runs/wait"
RUNS_STREAM_URL = f"{DEFAULT_URL}/runs/stream"
_LOG_TAIL_LIMIT = 5000
_MAX_BUFFER_LINES = 1200


def _read_assistant_id(project_root: pathlib.Path) -> str:
    cfg = project_root / "langgraph.json"
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return "agent"

    graphs = data.get("graphs") if isinstance(data, dict) else {}
    if isinstance(graphs, dict) and graphs:
        return next(iter(graphs.keys()))
    return "agent"


def _start_dev_server(project_root: pathlib.Path) -> Tuple[subprocess.Popen, List[str], threading.Lock, threading.Thread]:
    proc = subprocess.Popen(
        ["langgraph", "dev"],
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    buffer: List[str] = []
    lock = threading.Lock()

    def _pump() -> None:
        if not proc.stdout:
            return
        try:
            for line in proc.stdout:
                with lock:
                    buffer.append(line)
                    if len(buffer) > _MAX_BUFFER_LINES:
                        del buffer[0 : len(buffer) - _MAX_BUFFER_LINES]
        except Exception:
            pass

    thread = threading.Thread(target=_pump, name="langgraph-dev-log-reader", daemon=True)
    thread.start()
    return proc, buffer, lock, thread


def _logs_tail(buffer: Iterable[str], lock: threading.Lock, limit: int = _LOG_TAIL_LIMIT) -> str:
    with lock:
        joined = "".join(buffer)
    if len(joined) > limit:
        return joined[-limit:]
    return joined


def _wait_server_ready(timeout_s: float = 30.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            resp = requests.get(DOCS_URL, timeout=2)
            if resp.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _contains_token(payload: Any, token: str) -> bool:
    upper_token = token.upper()

    def _walk(value: Any) -> bool:
        if isinstance(value, str):
            return upper_token in value.upper()
        if isinstance(value, dict):
            for item in value.values():
                if _walk(item):
                    return True
            return False
        if isinstance(value, (list, tuple)):
            for item in value:
                if _walk(item):
                    return True
        return False

    return _walk(payload)


def _snippet_from_json(data: Any, fallback: str, limit: int = 200) -> str:
    try:
        rendered = json.dumps(data, ensure_ascii=False)
    except TypeError:
        rendered = fallback
    if not rendered:
        rendered = fallback
    return rendered[:limit]


def _headers_with_optional_key() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    api_key = os.getenv("LANGGRAPH_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def _terminate_process(proc: subprocess.Popen, thread: threading.Thread) -> None:
    with contextlib.suppress(Exception):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
    with contextlib.suppress(Exception):
        if proc.stdout:
            proc.stdout.close()
    if thread.is_alive():
        thread.join(timeout=1)


def final_check_node(state: Dict[str, Any]) -> Dict[str, Any]:
    scaffold = state.get("scaffold") or {}
    project_root_str = scaffold.get("project_root")
    if not project_root_str:
        state["final_check"] = {
            "ok": False,
            "phase": "start",
            "error": "missing project_root",
            "logs_tail": "",
            "assistant_id": None,
        }
        return state

    project_root = pathlib.Path(project_root_str)
    assistant_id = None
    try:
        proc, log_buffer, lock, reader_thread = _start_dev_server(project_root)
    except OSError as exc:
        state["final_check"] = {
            "ok": False,
            "phase": "start",
            "error": f"failed to launch langgraph dev: {exc}",
            "logs_tail": "",
            "assistant_id": assistant_id,
        }
        return state

    try:
        assistant_id = _read_assistant_id(project_root)
        if not _wait_server_ready():
            tail = _logs_tail(log_buffer, lock)
            state["final_check"] = {
                "ok": False,
                "phase": "start",
                "error": "server not ready",
                "logs_tail": tail,
                "assistant_id": assistant_id,
            }
            return state

        probe_prompt = (state.get("acceptance") or {}).get("probe_prompt") or "Reply with the single word: PONG"
        headers = _headers_with_optional_key()
        payload = {
            "assistant_id": assistant_id,
            "input": {"messages": [{"role": "human", "content": probe_prompt}]},
        }

        try:
            response = requests.post(RUNS_WAIT_URL, json=payload, headers=headers, timeout=30)
        except requests.RequestException as exc:
            tail = _logs_tail(log_buffer, lock)
            state["final_check"] = {
                "ok": False,
                "phase": "probe",
                "assistant_id": assistant_id,
                "error": f"runs/wait request failed: {exc}",
                "logs_tail": tail,
                "used_api_key": "x-api-key" in headers,
            }
            return state

        snippet_source = response.text
        json_payload: Any
        try:
            json_payload = response.json()
            snippet = _snippet_from_json(json_payload, snippet_source)
        except ValueError:
            json_payload = None
            snippet = snippet_source[:200]

        pong_ok = False
        if response.status_code < 400 and json_payload is not None:
            pong_ok = _contains_token(json_payload, "PONG")
        elif response.status_code < 400:
            pong_ok = "PONG" in snippet.upper()

        result: Dict[str, Any]
        if pong_ok:
            result = {
                "ok": True,
                "phase": "probe",
                "assistant_id": assistant_id,
                "snippet": snippet,
                "status": response.status_code,
                "used_api_key": "x-api-key" in headers,
            }
        else:
            tail = _logs_tail(log_buffer, lock)
            error_msg = "missing expected response token" if response.status_code < 400 else f"HTTP {response.status_code}"
            result = {
                "ok": False,
                "phase": "probe",
                "assistant_id": assistant_id,
                "error": error_msg,
                "status": response.status_code,
                "response_snippet": snippet,
                "logs_tail": tail,
                "used_api_key": "x-api-key" in headers,
            }
        state["final_check"] = result
        return state
    finally:
        _terminate_process(proc, reader_thread)


__all__ = [
    "final_check_node",
    "_read_assistant_id",
    "_start_dev_server",
    "_logs_tail",
    "_wait_server_ready",
    "_headers_with_optional_key",
    "_contains_token",
    "_terminate_process",
    "RUNS_STREAM_URL",
]
