"""Minimal executor module for compatibility in tests."""

from __future__ import annotations

from typing import Any, Dict

from asb.llm.client import get_chat_model  # re-exported for monkeypatching in tests


def update_node_implementations(plan: Dict[str, Any]) -> None:
    """Placeholder implementation used in tests."""

    return None


__all__ = ["update_node_implementations", "get_chat_model"]
