"""Minimal architecture designer module for compatibility in tests."""

from __future__ import annotations

from typing import Any, Dict

from asb.llm.client import get_chat_model  # re-exported for monkeypatching in tests


def design_architecture(state: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder architecture design implementation used in tests."""

    return state


__all__ = ["design_architecture", "get_chat_model"]
