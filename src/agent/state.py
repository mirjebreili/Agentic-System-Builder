from __future__ import annotations
from typing import Any, Dict, List, Literal, TypedDict

class ChatMessage(TypedDict, total=False):
    role: Literal["human", "user", "assistant", "system", "tool"]
    content: str

class AppState(TypedDict, total=False):
    messages: List[ChatMessage]
    plan: Dict[str, Any]
    flags: Dict[str, bool]
    metrics: Dict[str, Any]
    debug: Dict[str, Any]
