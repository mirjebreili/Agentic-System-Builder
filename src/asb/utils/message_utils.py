"""Utilities for safely handling LangChain message objects."""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence


TEXT_MIME_PREFIXES = ("text/", "application/json", "application/xml")
MAX_ATTACHMENT_SIZE = 1_048_576  # 1 MiB


def _looks_like_text_mime(mime: str | None) -> bool:
    if not mime:
        return True
    lowered = mime.lower()
    if any(lowered.startswith(prefix) for prefix in TEXT_MIME_PREFIXES):
        return True
    return lowered in {
        "application/javascript",
        "application/x-javascript",
        "application/yaml",
        "application/x-yaml",
        "application/toml",
    }


def _decode_bytes(data: bytes, encoding: str | None = None) -> str:
    encodings: Sequence[str] = []
    if encoding:
        encodings = [encoding]
    encodings = [*encodings, "utf-8", "latin-1"]
    for enc in encodings:
        try:
            return data.decode(enc, errors="ignore")
        except (LookupError, UnicodeDecodeError):
            continue
    return ""


def _read_text_file(path_value: Any, mime_type: str | None = None) -> str:
    path = Path(os.fspath(path_value))
    if not path.exists() or not path.is_file():
        return ""
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    if size > MAX_ATTACHMENT_SIZE:
        return ""

    guessed_mime, _ = mimetypes.guess_type(str(path))
    effective_mime = mime_type or guessed_mime
    if not _looks_like_text_mime(effective_mime):
        return ""

    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _iter_block_candidates(block: Mapping[str, Any]) -> Iterable[tuple[str, Any]]:
    for key in ("text", "content", "data", "value", "body", "payload"):
        if key in block:
            yield key, block[key]


def _normalize_content(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, bytes):
        return _decode_bytes(value)

    if isinstance(value, (list, tuple)):
        pieces = [piece for piece in (_normalize_content(part) for part in value) if piece]
        return "\n".join(pieces)

    if isinstance(value, Mapping):
        mime_type = value.get("mime_type") or value.get("media_type")
        encoding = value.get("encoding") or value.get("charset")

        for key, candidate in _iter_block_candidates(value):
            if isinstance(candidate, bytes):
                if _looks_like_text_mime(mime_type):
                    text = _decode_bytes(candidate, encoding=encoding)
                    if text:
                        return text
                continue

            text = _normalize_content(candidate)
            if text:
                return text

        file_path = (
            value.get("file_path")
            or value.get("path")
            or value.get("filepath")
            or value.get("file")
        )
        if file_path:
            text = _read_text_file(file_path, mime_type=mime_type)
            if text:
                return text

        file_id = value.get("file_id") or value.get("attachment_id")
        if file_id and isinstance(file_id, (str, os.PathLike)):
            text = _read_text_file(file_id, mime_type=mime_type)
            if text:
                return text

        data_value = value.get("bytes")
        if isinstance(data_value, bytes) and _looks_like_text_mime(mime_type):
            return _decode_bytes(data_value, encoding=encoding)

        return ""

    if hasattr(value, "content"):
        content_attr = getattr(value, "content")
        normalized = _normalize_content(content_attr)
        if normalized:
            return normalized

    if hasattr(value, "text"):
        return _normalize_content(getattr(value, "text"))

    if value not in (None, ""):
        return str(value)

    return ""


def extract_last_message_content(messages: List[Any], default: str = "") -> str:
    """Safely extract content from the last message in a list."""
    if not messages:
        return default

    last_message = messages[-1]

    value: Any = None
    if hasattr(last_message, "content"):
        value = getattr(last_message, "content", None)
    elif isinstance(last_message, dict):
        value = last_message.get("content")
    elif isinstance(last_message, str):
        value = last_message
    else:
        value = last_message

    normalized = _normalize_content(value)
    return normalized if normalized else default


def extract_user_messages_content(messages: List[Any]) -> List[str]:
    """Extract content from all user/human messages."""
    user_content: List[str] = []

    for msg in messages:
        if hasattr(msg, "content") and hasattr(msg, "type"):
            if getattr(msg, "type", "").lower() in {"human", "user"}:
                content = getattr(msg, "content", "")
                normalized = _normalize_content(content)
                if normalized:
                    user_content.append(normalized)
        elif isinstance(msg, dict):
            if (msg.get("role") or "").lower() in {"human", "user"}:
                content = msg.get("content", "")
                normalized = _normalize_content(content)
                if normalized:
                    user_content.append(normalized)

    return user_content


def safe_message_access(message: Any, field: str, default: Any = "") -> Any:
    """Safely access any field from a message object or dict."""
    if hasattr(message, field):
        return getattr(message, field, default)

    if isinstance(message, dict):
        return message.get(field, default)

    return default


def normalize_content(value: Any) -> str:
    """Public wrapper around the internal content normalizer."""

    return _normalize_content(value)


__all__ = [
    "extract_last_message_content",
    "extract_user_messages_content",
    "safe_message_access",
    "normalize_content",
]
