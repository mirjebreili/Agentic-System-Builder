from __future__ import annotations

import json
import pathlib
import time
from typing import Any, Dict, List

import requests

from .final_check import (
    DEFAULT_URL,
    RUNS_STREAM_URL,
    _contains_token,
    _headers_with_optional_key,
    _logs_tail,
    _read_assistant_id,
    _start_dev_server,
    _terminate_process,
    _wait_server_ready,
)

_AGENT_CARD_PATH = ".well-known/agent-card.json"
_STREAM_TIMEOUT_S = 10.0
_RESPONSE_CAPTURE_LIMIT = 5000


def _collect_stream_snippet(response: requests.Response) -> List[str]:
    snippet_parts: List[str] = []
    start = time.time()
    for line in response.iter_lines(decode_unicode=True):
        if time.time() - start > _STREAM_TIMEOUT_S:
            break
        if not line:
            continue
        snippet_parts.append(line)
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            payload = line
        if isinstance(payload, (dict, list)):
            if _contains_token(payload, "PONG"):
                break
        elif isinstance(payload, str) and "PONG" in payload.upper():
            break
    return snippet_parts


def final_check_fallback_node(state: Dict[str, Any]) -> Dict[str, Any]:
    current = dict(state.get("final_check") or {})
    if current.get("ok") is True:
        state["final_check"] = current
        return state

    scaffold = state.get("scaffold") or {}
    project_root_str = scaffold.get("project_root")
    if not project_root_str:
        current.update(
            {
                "ok": False,
                "phase": "fallback",
                "error": "missing project_root",
                "logs_tail": current.get("logs_tail", ""),
                "assistant_id": current.get("assistant_id"),
            }
        )
        state["final_check"] = current
        return state

    project_root = pathlib.Path(project_root_str)
    proc = None
    log_buffer: List[str] = []
    lock = None
    reader_thread = None

    try:
        proc, log_buffer, lock, reader_thread = _start_dev_server(project_root)
    except OSError as exc:
        current.update(
            {
                "ok": False,
                "phase": "start",
                "error": f"failed to launch langgraph dev: {exc}",
                "logs_tail": current.get("logs_tail", ""),
                "assistant_id": current.get("assistant_id"),
            }
        )
        state["final_check"] = current
        return state

    try:
        if not _wait_server_ready():
            tail = _logs_tail(log_buffer, lock)
            current.update(
                {
                    "ok": False,
                    "phase": "start",
                    "error": "server not ready",
                    "logs_tail": tail,
                    "assistant_id": current.get("assistant_id") or _read_assistant_id(project_root),
                }
            )
            state["final_check"] = current
            return state

        assistant_id = current.get("assistant_id") or _read_assistant_id(project_root)
        headers = _headers_with_optional_key()
        used_api_key = "x-api-key" in headers
        card_url = f"{DEFAULT_URL}/{_AGENT_CARD_PATH}"
        card_errors: Dict[str, str] = {}

        try:
            primary = requests.get(
                card_url,
                params={"assistant_id": assistant_id},
                headers=headers,
                timeout=10,
            )
        except requests.RequestException as exc:
            tail = _logs_tail(log_buffer, lock)
            current.update(
                {
                    "ok": False,
                    "phase": "fallback",
                    "assistant_id": assistant_id,
                    "error": f"agent-card fetch failed: {exc}",
                    "logs_tail": tail,
                    "used_api_key": used_api_key,
                }
            )
            state["final_check"] = current
            return state

        if not primary.ok:
            card_errors[str(assistant_id)] = primary.text[:200]
            try:
                secondary = requests.get(
                    card_url,
                    params={"assistant_id": "agent"},
                    headers=headers,
                    timeout=10,
                )
            except requests.RequestException as exc:
                tail = _logs_tail(log_buffer, lock)
                current.update(
                    {
                        "ok": False,
                        "phase": "fallback",
                        "assistant_id": assistant_id,
                        "error": f"agent-card fallback failed: {exc}",
                        "logs_tail": tail,
                        "card_errors": card_errors,
                        "used_api_key": used_api_key,
                    }
                )
                state["final_check"] = current
                return state

            if secondary.ok:
                assistant_id = "agent"
            else:
                card_errors["agent"] = secondary.text[:200]

        probe_prompt = (state.get("acceptance") or {}).get("probe_prompt") or "Reply with the single word: PONG"
        payload = {
            "assistant_id": assistant_id,
            "input": {"messages": [{"role": "human", "content": probe_prompt}]},
            "stream_mode": "messages-tuple",
        }

        try:
            response = requests.post(
                RUNS_STREAM_URL,
                json=payload,
                headers=headers,
                timeout=_STREAM_TIMEOUT_S,
                stream=True,
            )
        except requests.RequestException as exc:
            tail = _logs_tail(log_buffer, lock)
            current.update(
                {
                    "ok": False,
                    "phase": "fallback",
                    "assistant_id": assistant_id,
                    "error": f"runs/stream request failed: {exc}",
                    "logs_tail": tail,
                    "card_errors": card_errors or None,
                    "used_api_key": used_api_key,
                }
            )
            state["final_check"] = current
            return state

        captured_body: List[str] = []
        if response.status_code >= 400:
            for chunk in response.iter_content(chunk_size=512, decode_unicode=True):
                if not chunk:
                    continue
                captured_body.append(chunk)
                if sum(len(part) for part in captured_body) > _RESPONSE_CAPTURE_LIMIT:
                    break
            tail = _logs_tail(log_buffer, lock)
            response.close()
            current.update(
                {
                    "ok": False,
                    "phase": "fallback",
                    "assistant_id": assistant_id,
                    "error": f"runs/stream HTTP {response.status_code}",
                    "logs_tail": tail,
                    "response_body": "".join(captured_body)[:200],
                    "card_errors": card_errors or None,
                    "used_api_key": used_api_key,
                }
            )
            state["final_check"] = current
            return state

        snippet_parts = _collect_stream_snippet(response)
        response.close()
        snippet = "\n".join(snippet_parts)[:200]

        success = False
        for part in snippet_parts:
            try:
                parsed: Any = json.loads(part)
            except json.JSONDecodeError:
                parsed = part
            if isinstance(parsed, (dict, list)) and _contains_token(parsed, "PONG"):
                success = True
                break
            if isinstance(parsed, str) and "PONG" in parsed.upper():
                success = True
                break

        if success:
            current.update(
                {
                    "ok": True,
                    "phase": "fallback",
                    "assistant_id": assistant_id,
                    "snippet": snippet,
                    "card_errors": card_errors or None,
                    "used_api_key": used_api_key,
                }
            )
        else:
            tail = _logs_tail(log_buffer, lock)
            current.update(
                {
                    "ok": False,
                    "phase": "fallback",
                    "assistant_id": assistant_id,
                    "error": "streaming probe missing expected token",
                    "logs_tail": tail,
                    "response_snippet": snippet,
                    "card_errors": card_errors or None,
                    "used_api_key": used_api_key,
                }
            )

        state["final_check"] = current
        return state
    finally:
        if proc is not None and reader_thread is not None:
            _terminate_process(proc, reader_thread)


__all__ = ["final_check_fallback_node"]
