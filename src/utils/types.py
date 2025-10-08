from typing import List, Dict, Any, TypedDict, Literal

class ToolSpec(TypedDict):
    name: str
    description: str
    role: Literal["producer", "transformer", "consumer", "mixed"]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]

Registry = Dict[str, ToolSpec]

class PlanCandidate(TypedDict):
    plan: List[str]
    rationale: str
    scores: Dict[str, float]
    raw_score: float
    confidence: float