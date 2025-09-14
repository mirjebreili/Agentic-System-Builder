from __future__ import annotations
from typing import Any, Dict, List, Literal, TypedDict

class ChatMessage(TypedDict, total=False):
    role: Literal["human","user","assistant","system","tool"]
    content: str

class AppState(TypedDict, total=False):
    messages: List[ChatMessage]
    plan: Dict[str, Any]
    flags: Dict[str, bool]
    metrics: Dict[str, Any]
    debug: Dict[str, Any]
    tests: Dict[str, Any]
    artifacts: Dict[str, Any]
    review: Dict[str, Any]
    replan: bool
    passed: bool
    scaffold: Dict[str, Any]
    sandbox: Dict[str, Any]
    report: Dict[str, Any]
