from typing import TypedDict, List, Dict, Any, Optional

class Candidate(TypedDict):
    plan: List[str]
    rationale: str

class GraphState(TypedDict):
    """
    Represents the state of the graph.
    """
    initial_message: str
    tool_registry: Dict[str, Any]
    candidates: List[Candidate]
    chosen: Optional[int]
    note: str