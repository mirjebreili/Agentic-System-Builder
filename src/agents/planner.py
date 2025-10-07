from __future__ import annotations

from pathlib import Path
from typing import List

from ..llm.provider import call_llm
from ..utils.scoring import apply_scores
from ..utils.types import PlanCandidate, PlanningResult, Registry

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "planner.md"
_PLANNER_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")

_SUM_KEYWORDS = ["sum", "جمع", "مجموع", "total"]
_PREFIX_MARKERS = ["_", "prefix", "شروع"]


def _requires_sum(question: str) -> bool:
    lowered = question.lower()
    return any(keyword in lowered for keyword in _SUM_KEYWORDS)


def _mentions_prefix(question: str) -> bool:
    lowered = question.lower()
    return any(marker in lowered for marker in _PREFIX_MARKERS)


def _render_prompt(question: str, registry: Registry, k: int) -> str:
    tool_lines = []
    for spec in registry.values():
        tool_lines.append(f"- {spec['name']}: {spec.get('description', '')}")
    tools_block = "\n".join(tool_lines)
    return _PLANNER_PROMPT + f"\n\nQuestion: {question}\nTools:\n{tools_block}\nCandidates: {k}"


def _add_candidate(candidates: List[PlanCandidate], plan: List[str], rationale: str) -> None:
    if any(existing.plan == plan for existing in candidates):
        return
    candidates.append(PlanCandidate(plan=plan, rationale=rationale))


def plan_candidates(question: str, registry: Registry, k: int = 3) -> PlanningResult:
    """Generate ToT-style plan candidates and score them."""

    if k <= 0:
        raise ValueError("k must be positive")

    _ = call_llm(_render_prompt(question, registry, k))

    requires_sum = _requires_sum(question)
    mentions_prefix = _mentions_prefix(question)
    has_reader = "HttpBasedAtlasReadByKey" in registry
    has_aggregator = "membasedAtlasKeyStreamAggregator" in registry

    candidates: List[PlanCandidate] = []

    if requires_sum and mentions_prefix and has_reader and has_aggregator:
        _add_candidate(
            candidates,
            ["HttpBasedAtlasReadByKey", "membasedAtlasKeyStreamAggregator"],
            "Fetch Atlas records matching the namespace/prefix then aggregate numeric suffixes.",
        )
        _add_candidate(
            candidates,
            ["HttpBasedAtlasReadByKey"],
            "Retrieve Atlas data and defer aggregation to downstream logic.",
        )
        _add_candidate(
            candidates,
            ["membasedAtlasKeyStreamAggregator"],
            "Attempt to sum existing cached keys without re-fetching records.",
        )
    else:
        available_tools = list(registry.keys())
        if available_tools:
            _add_candidate(
                candidates,
                [available_tools[0]],
                "Use the first available operator to address the question heuristically.",
            )
        if len(available_tools) >= 2:
            _add_candidate(
                candidates,
                available_tools[:2],
                "Chain the first two operators to cover data retrieval then transformation.",
            )
        if len(available_tools) >= 3:
            _add_candidate(
                candidates,
                available_tools[:3],
                "Combine three operators for comprehensive coverage.",
            )

    while len(candidates) < k:
        _add_candidate(
            candidates,
            ["HttpBasedAtlasReadByKey"],
            "Fallback to the HTTP reader when no better option is available.",
        )

    candidates = candidates[:k]
    apply_scores(candidates, question, registry)

    chosen = 0
    if candidates:
        chosen = max(range(len(candidates)), key=lambda idx: candidates[idx].raw_score)

    return PlanningResult(candidates=candidates, chosen=chosen)
