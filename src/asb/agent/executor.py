"""Minimal executor module for compatibility in tests."""

from __future__ import annotations

from typing import Any, Dict

from asb.llm.client import get_chat_model  # re-exported for monkeypatching in tests


def update_node_implementations(plan: Dict[str, Any]) -> None:
    """Placeholder implementation used in tests."""

    return None


def execute_deep(state: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal executor stub that records capability execution progress."""

    plan = state.get("plan") or {}
    capabilities = plan.get("capabilities") or []
    messages = list(state.get("messages") or [])

    if capabilities:
        messages.append(
            {
                "role": "assistant",
                "content": f"[execution] Prepared {len(capabilities)} capability integrations.",
            }
        )
    else:
        messages.append(
            {
                "role": "assistant",
                "content": "[execution] No capabilities defined; nothing to execute.",
            }
        )

    state["messages"] = messages
    state.setdefault("passed", True)
    return state


__all__ = ["update_node_implementations", "execute_deep", "get_chat_model"]
