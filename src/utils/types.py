from typing import Any, Dict, List, Literal, TypedDict


class ToolSpec(TypedDict):
    name: str
    description: str
    role: Literal["producer", "transformer", "consumer", "mixed"]
    inputs: Dict[str, Any]  # args/config schema from docs
    outputs: Dict[
        str, Any
    ]  # e.g. {"type":"stream|json","path":"data.result.*","keys_field":"keys"}


class PlanCandidate(TypedDict):
    plan: List[str]
    rationale: str
    scores: Dict[str, float]  # coverage, io, simplicity, constraints
    raw_score: float
    confidence: float


Registry = Dict[str, ToolSpec]