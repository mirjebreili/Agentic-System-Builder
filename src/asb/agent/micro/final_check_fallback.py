from __future__ import annotations

import pathlib
from typing import Any, Dict

import requests

from .final_check import (
    BASE_URL,
    _headers_with_optional_key,
    _logs_tail,
    _read_assistant_id,
    _start_server,
    _terminate_process,
    _wait_for_docs,
)

_THREADS_URL = f"{BASE_URL}/threads"


def final_check_fallback_node(state: Dict[str, Any]) -> Dict[str, Any]:
    current = dict(state.get("final_check") or {})
    if current.get("ok"):
        state["final_check"] = current
        return state

    scaffold = state.get("scaffold") or {}
    project_root_str = scaffold.get("project_root")
    if not project_root_str:
        current.update({"ok": False, "phase": "fallback-start", "error": "missing project_root"})
        state["final_check"] = current
        return state

    project_root = pathlib.Path(str(project_root_str))
    assistant_id = current.get("assistant_id") or _read_assistant_id(project_root)

    try:
        process, buffer, lock, thread = _start_server(project_root)
    except OSError as exc:
        current.update(
            {
                "ok": False,
                "phase": "fallback-start",
                "assistant_id": assistant_id,
                "error": f"failed to launch langgraph dev: {exc}",
            }
        )
        state["final_check"] = current
        return state

    try:
        if not _wait_for_docs():
            current.update(
                {
                    "ok": False,
                    "phase": "fallback-start",
                    "assistant_id": assistant_id,
                    "error": "server not ready",
                    "logs_tail": _logs_tail(buffer, lock),
                }
            )
            state["final_check"] = current
            return state

        headers = _headers_with_optional_key()
        probe = (state.get("acceptance") or {}).get("probe_prompt") or "Reply with the single word: PONG"

        try:
            thread_response = requests.post(_THREADS_URL, headers=headers, timeout=30)
        except requests.RequestException as exc:
            current.update(
                {
                    "ok": False,
                    "phase": "fallback",
                    "assistant_id": assistant_id,
                    "error": f"thread creation failed: {exc}",
                    "logs_tail": _logs_tail(buffer, lock),
                }
            )
            state["final_check"] = current
            return state

        thread_body = thread_response.text
        if thread_response.status_code >= 400:
            current.update(
                {
                    "ok": False,
                    "phase": "fallback",
                    "assistant_id": assistant_id,
                    "status": thread_response.status_code,
                    "snippet": thread_body[:400],
                    "error": "thread creation returned error",
                    "logs_tail": _logs_tail(buffer, lock),
                }
            )
            state["final_check"] = current
            return state

        try:
            thread_payload = thread_response.json()
        except ValueError:
            thread_payload = {}
        thread_id = (
            thread_payload.get("id")
            or thread_payload.get("thread_id")
            or thread_payload.get("data", {}).get("id")
        )
        if not thread_id:
            current.update(
                {
                    "ok": False,
                    "phase": "fallback",
                    "assistant_id": assistant_id,
                    "snippet": thread_body[:400],
                    "error": "thread creation missing identifier",
                    "logs_tail": _logs_tail(buffer, lock),
                }
            )
            state["final_check"] = current
            return state

        payload = {
            "assistant_id": assistant_id,
            "input": {"messages": [{"role": "human", "content": probe}]},
        }

        try:
            response = requests.post(
                f"{_THREADS_URL}/{thread_id}/runs/wait",
                json=payload,
                headers=headers,
                timeout=30,
            )
        except requests.RequestException as exc:
            current.update(
                {
                    "ok": False,
                    "phase": "fallback",
                    "assistant_id": assistant_id,
                    "error": f"threaded runs/wait failed: {exc}",
                    "logs_tail": _logs_tail(buffer, lock),
                }
            )
            state["final_check"] = current
            return state

        body = response.text
        snippet = body[:400]
        success = response.status_code < 400 and "PONG" in body.upper()
        result = {
            "ok": bool(success),
            "phase": "fallback",
            "assistant_id": assistant_id,
            "status": response.status_code,
            "snippet": snippet,
        }
        if not success:
            result["logs_tail"] = _logs_tail(buffer, lock)
        else:
            result["thread_id"] = thread_id
        state["final_check"] = result
        return state
    finally:
        _terminate_process(process, thread)


__all__ = ["final_check_fallback_node"]
