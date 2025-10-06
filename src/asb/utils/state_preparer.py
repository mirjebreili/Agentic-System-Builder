"""Utilities for preparing the initial agent state before graph execution."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List

from asb.utils.message_utils import normalize_content


_ATTACHMENT_KEYS = ("attachments", "files", "input_files", "uploaded_files")


def _collect_attachment_candidates(state: Mapping[str, Any]) -> List[Any]:
    """Return a list of attachment-like payloads from the incoming state."""

    candidates: List[Any] = []
    for key in _ATTACHMENT_KEYS:
        value = state.get(key)
        if not value:
            continue

        if isinstance(value, (str, bytes, bytearray)):
            candidates.append(value)
            continue

        if isinstance(value, Mapping):
            candidates.append(value)
            continue

        if isinstance(value, Sequence):
            for item in value:
                if item is None:
                    continue
                candidates.append(item)
            continue

        candidates.append(value)

    return candidates


def _normalize_attachment(entry: Any) -> Dict[str, Any] | None:
    """Convert arbitrary attachment representations into message blocks."""

    if entry is None:
        return None

    if isinstance(entry, Mapping):
        block = dict(entry)
        block.setdefault("type", "file")
        if block.get("type") != "file":
            return None
        return block

    if isinstance(entry, (str, os.PathLike)):
        return {"type": "file", "file_path": os.fspath(entry)}

    if isinstance(entry, (bytes, bytearray)):
        return {"type": "file", "data": bytes(entry)}

    return None


def _is_user_message(message: Any) -> bool:
    if isinstance(message, Mapping):
        role = (message.get("role") or message.get("type") or "").lower()
        return role in {"user", "human"}

    if hasattr(message, "type"):
        role_value = getattr(message, "type", "")
        if isinstance(role_value, str) and role_value.lower() in {"user", "human"}:
            return True

    return False


def _message_to_dict(message: Any) -> Dict[str, Any]:
    if isinstance(message, Mapping):
        result = dict(message)
        result.setdefault("role", (message.get("type") or "user"))
        return result

    role = getattr(message, "type", None) or getattr(message, "role", None) or "user"
    content = getattr(message, "content", message)
    return {"role": role, "content": content}


def _ensure_content_list(content: Any) -> List[Any]:
    if isinstance(content, list):
        return list(content)
    if content is None:
        return []
    return [content]


def prepare_initial_state(state: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Augment the incoming state with attachment-aware user messages."""

    if not isinstance(state, Mapping):
        return dict(state or {})  # type: ignore[arg-type]

    attachments_raw = _collect_attachment_candidates(state)
    attachments = [_normalize_attachment(entry) for entry in attachments_raw]
    attachments = [att for att in attachments if att]

    if not attachments:
        return dict(state)

    working: Dict[str, Any] = dict(state)
    messages: List[Any] = list(working.get("messages") or [])

    target_index = None
    for idx in range(len(messages) - 1, -1, -1):
        if _is_user_message(messages[idx]):
            messages[idx] = _message_to_dict(messages[idx])
            target_index = idx
            break

    if target_index is None:
        base_text = working.get("input_text") or working.get("input") or ""
        base_content: List[Any] = []
        if isinstance(base_text, str) and base_text:
            base_content.append(base_text)
        messages.append({"role": "user", "content": base_content})
        target_index = len(messages) - 1

    target = _message_to_dict(messages[target_index])
    content_list = _ensure_content_list(target.get("content"))
    content_list.extend(attachments)
    target["content"] = content_list
    target["role"] = (target.get("role") or "user").lower()
    if target["role"] not in {"user", "human"}:
        target["role"] = "user"
    messages[target_index] = target
    working["messages"] = messages

    base_input_text = working.get("input_text")
    if not isinstance(base_input_text, str):
        base_input_text = str(base_input_text or "")

    attachment_texts = [normalize_content(att) for att in attachments]
    attachment_texts = [text for text in attachment_texts if text]

    if attachment_texts:
        parts = [part for part in [base_input_text.strip()] if part]
        parts.extend(attachment_texts)
        working["input_text"] = "\n\n".join(parts)
    else:
        working["input_text"] = base_input_text

    return working


__all__ = ["prepare_initial_state"]

