"""TypedDict definitions for scaffold execution phases."""

from __future__ import annotations

from typing import Any, Dict, Literal, TypedDict


class ScaffoldPhase(TypedDict, total=False):
    """Structured metadata describing a scaffold execution phase."""

    name: str
    description: str
    status: Literal["pending", "in_progress", "complete", "failed", "skipped"]
    started_at: float
    completed_at: float
    duration: float
    summary: str
    details: Dict[str, Any]
    error: str
