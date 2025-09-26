"""Micro-node for harvesting conversational context metadata."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from asb.utils.message_utils import (
    extract_last_message_content,
    extract_user_messages_content,
)

_CONSTRAINT_PATTERNS: Dict[str, re.Pattern[str]] = {
    "require_streaming": re.compile(r"\bstream(?:ing|ed)\b", re.IGNORECASE),
    "prefer_rag": re.compile(r"\brag\b|retrieval-augmented", re.IGNORECASE),
    "avoid_network": re.compile(r"\boffline\b|\bno\s+internet\b", re.IGNORECASE),
    "must_use_tests": re.compile(r"\bpytest\b|\btests?\b", re.IGNORECASE),
}


def _normalize_texts(messages: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    for text in messages:
        candidate = text.strip()
        if candidate:
            normalized.append(candidate)
    return normalized


def _detect_constraints(texts: Iterable[str]) -> Dict[str, bool]:
    constraints: Dict[str, bool] = {}
    for key, pattern in _CONSTRAINT_PATTERNS.items():
        constraints[key] = any(pattern.search(text) for text in texts)
    return constraints


def context_collector_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Populate scratch goal and constraint metadata."""

    working_state: Dict[str, Any] = dict(state or {})
    messages = list(working_state.get("messages") or [])
    latest_goal = extract_last_message_content(messages, str(working_state.get("goal", "")))
    user_texts = _normalize_texts(extract_user_messages_content(messages))

    scratch = dict(working_state.get("scratch") or {})
    if latest_goal:
        scratch["goal"] = latest_goal.strip()
    scratch["constraints"] = _detect_constraints(user_texts)
    scratch.setdefault("conversation", {})
    scratch["conversation"]["recent_user_messages"] = user_texts[-3:]
    working_state["scratch"] = scratch
    return working_state


__all__ = ["context_collector_node"]
