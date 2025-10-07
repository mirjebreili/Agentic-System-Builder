from __future__ import annotations

import math
from typing import Dict, List, Sequence

from .types import CandidateScores, PlanCandidate, Registry


_SUM_KEYWORDS = ["sum", "جمع", "مجموع", "total"]
_PREFIX_KEYWORDS = ["prefix", "شروع", "start", "شروع میشن"]


def _needs_sum(question: str) -> bool:
    lowered = question.lower()
    return any(keyword in lowered for keyword in _SUM_KEYWORDS)


def _mentions_prefix(question: str) -> bool:
    lowered = question.lower()
    return "_" in lowered or any(keyword in lowered for keyword in _PREFIX_KEYWORDS)


def _has_tool(plan: Sequence[str], tool_name: str) -> bool:
    return any(step == tool_name for step in plan)


def _io_score(plan: Sequence[str], registry: Registry) -> float:
    if not plan:
        return 0.0

    for step in plan:
        if step not in registry:
            return 0.0

    first_tool = registry[plan[0]]
    first_consumes = set(first_tool.get("metadata", {}).get("consumes", []))
    if first_consumes:
        return 0.0

    if len(plan) == 1:
        return 1.0

    for current_name, next_name in zip(plan, plan[1:]):
        current = registry.get(current_name)
        nxt = registry.get(next_name)
        if not current or not nxt:
            return 0.0
        current_outputs = set(current.get("metadata", {}).get("produces", []))
        next_inputs = set(nxt.get("metadata", {}).get("consumes", []))
        if next_inputs and not current_outputs.intersection(next_inputs):
            return 0.0
    return 1.0


def _coverage_score(plan: Sequence[str], question: str) -> float:
    requires_sum = _needs_sum(question)
    requires_prefix = _mentions_prefix(question)

    if not plan:
        return 0.0

    if requires_sum and requires_prefix:
        has_reader = _has_tool(plan, "HttpBasedAtlasReadByKey")
        has_aggregator = _has_tool(plan, "membasedAtlasKeyStreamAggregator")
        if has_reader and has_aggregator:
            return 0.98
        if has_aggregator:
            return 0.6
        if has_reader:
            return 0.55
        return 0.2

    if requires_sum:
        return 0.8 if _has_tool(plan, "membasedAtlasKeyStreamAggregator") else 0.4

    return 0.75


def _simplicity_score(plan: Sequence[str]) -> float:
    if not plan:
        return 0.0
    if len(plan) == 1:
        return 1.0
    if len(plan) == 2:
        return 0.95
    return max(0.4, 0.95 - 0.2 * (len(plan) - 2))


def _constraints_score(plan: Sequence[str], question: str, registry: Registry) -> float:
    if not plan:
        return 0.0

    for step in plan:
        if step not in registry:
            return 0.0

    if _needs_sum(question) and _mentions_prefix(question):
        if not _has_tool(plan, "membasedAtlasKeyStreamAggregator"):
            return 0.3
        if plan[-1] != "membasedAtlasKeyStreamAggregator":
            return 0.6
    return 1.0


def score_plan(plan: Sequence[str], question: str, registry: Registry) -> CandidateScores:
    coverage = _coverage_score(plan, question)
    io = _io_score(plan, registry)
    simplicity = _simplicity_score(plan)
    constraints = _constraints_score(plan, question, registry)
    return CandidateScores(coverage=coverage, io=io, simplicity=simplicity, constraints=constraints)


def compute_raw_score(scores: CandidateScores) -> float:
    values = [scores.get("coverage", 0.0), scores.get("io", 0.0), scores.get("simplicity", 0.0), scores.get("constraints", 0.0)]
    return sum(values) / len(values)


def softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    if total == 0:
        return [0.0 for _ in values]
    return [value / total for value in exps]


def apply_scores(candidates: List[PlanCandidate], question: str, registry: Registry) -> None:
    raw_scores: List[float] = []
    for candidate in candidates:
        scores = score_plan(candidate.plan, question, registry)
        candidate.scores = scores
        candidate.raw_score = compute_raw_score(scores)
        raw_scores.append(candidate.raw_score)

    confidences = softmax(raw_scores)
    for candidate, confidence in zip(candidates, confidences):
        candidate.confidence = confidence
