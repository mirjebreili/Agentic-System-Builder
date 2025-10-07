from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, TypedDict


class ToolSpec(TypedDict, total=False):
    """Structured metadata describing a tool/operator."""

    name: str
    description: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    role: Literal["producer", "transformer", "consumer", "mixed"]
    metadata: Dict[str, Any]


Registry = Dict[str, ToolSpec]


class CandidateScores(TypedDict, total=False):
    coverage: float
    io: float
    simplicity: float
    constraints: float


@dataclass
class PlanCandidate:
    plan: List[str]
    rationale: str
    scores: CandidateScores = field(default_factory=dict)  # type: ignore[assignment]
    raw_score: float = 0.0
    confidence: float = 0.0


class PlanningResult(TypedDict):
    candidates: List[PlanCandidate]
    chosen: int


class ParsedMessage(TypedDict, total=False):
    question: str
    plugin_docs: Dict[str, str]
    raw: str


class PlannerState(TypedDict, total=False):
    input_text: str
    messages: List[Dict[str, Any]]
    question: str
    plugin_docs: Dict[str, str]
    registry: Registry
    planner_result: PlanningResult
    pending_plan: List[str]
    approved_plan: List[str]
    hitl_status: str
    hitl_feedback: str
    hitl_prompt: str
