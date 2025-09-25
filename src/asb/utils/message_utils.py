"""Utilities for safely handling LangChain message objects."""

from typing import Any, List


def extract_last_message_content(messages: List[Any], default: str = "") -> str:
    """Safely extract content from the last message in a list."""
    if not messages:
        return default

    last_message = messages[-1]

    if hasattr(last_message, "content"):
        value = getattr(last_message, "content", default)
        if value in (None, ""):
            return default
        return str(value)

    if isinstance(last_message, dict):
        value = last_message.get("content", default)
        if value in (None, ""):
            return default
        return str(value)

    if isinstance(last_message, str):
        return last_message

    return default


def extract_user_messages_content(messages: List[Any]) -> List[str]:
    """Extract content from all user/human messages."""
    user_content: List[str] = []

    for msg in messages:
        if hasattr(msg, "content") and hasattr(msg, "type"):
            if getattr(msg, "type", "").lower() in {"human", "user"}:
                content = getattr(msg, "content", "")
                if content not in (None, ""):
                    user_content.append(str(content))
        elif isinstance(msg, dict):
            if (msg.get("role") or "").lower() in {"human", "user"}:
                content = msg.get("content", "")
                if content not in (None, ""):
                    user_content.append(str(content))

    return user_content


def safe_message_access(message: Any, field: str, default: Any = "") -> Any:
    """Safely access any field from a message object or dict."""
    if hasattr(message, field):
        return getattr(message, field, default)

    if isinstance(message, dict):
        return message.get(field, default)

    return default


__all__ = [
    "extract_last_message_content",
    "extract_user_messages_content",
    "safe_message_access",
]
