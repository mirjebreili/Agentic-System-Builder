from __future__ import annotations

from itertools import product
from typing import Iterable, List, Sequence, Tuple

from ..utils.scoring import compute_raw_score, score_plan, softmax
from ..utils.types import PlanCandidate, PlanningResult, Registry

_MAX_PLAN_LENGTH = 3
_RELEVANT_CHAIN = (
    "HttpBasedAtlasReadByKey",
    "membasedAtlasKeyStreamAggregator",
)
_SUM_KEYWORDS = ("sum", "جمع", "مجموع", "total")
_PREFIX_KEYWORDS = ("prefix", "شروع", "start", "price_", "prefix:")


def _requires_sum(question: str) -> bool:
    lowered = question.lower()
    return any(keyword in lowered for keyword in _SUM_KEYWORDS)


def _mentions_prefix(question: str) -> bool:
    lowered = question.lower()
    return "_" in lowered or any(keyword in lowered for keyword in _PREFIX_KEYWORDS)


def _ordered_tools(question: str, registry: Registry) -> List[str]:
    tools = list(registry.keys())
    if not tools:
        return []

    requires_sum = _requires_sum(question)
    mentions_prefix = _mentions_prefix(question)

    def sort_key(name: str) -> Tuple[int, int]:
        priority = 5
        secondary = tools.index(name)

        if name == _RELEVANT_CHAIN[0]:  # HttpBasedAtlasReadByKey
            priority = 0 if requires_sum or mentions_prefix else 1
        elif name == _RELEVANT_CHAIN[1]:  # membasedAtlasKeyStreamAggregator
            priority = 1 if requires_sum and mentions_prefix else 2

        role = registry[name].get("role", "mixed")
        if role == "producer":
            secondary = min(secondary, 1)
        elif role == "consumer":
            secondary = secondary + 5

        return (priority, secondary)

    return sorted(tools, key=sort_key)


def _generate_thoughts(question: str, ordered_tools: Sequence[str]) -> Iterable[Sequence[str]]:
    if not ordered_tools:
        return []

    # Breadth-first Tree-of-Thought exploration up to length 3.
    queue: List[Tuple[Tuple[str, ...], int]] = [(tuple(), 0)]
    thoughts: List[Tuple[str, ...]] = []

    while queue:
        current, depth = queue.pop(0)
        if depth >= _MAX_PLAN_LENGTH:
            continue
        for tool in ordered_tools:
            new_plan = current + (tool,)
            thoughts.append(new_plan)
            queue.append((new_plan, depth + 1))

    return thoughts


def _tool_summary(name: str, registry: Registry) -> str:
    spec = registry.get(name, {})
    description = str(spec.get("description", "")).strip()
    if not description:
        return name
    first_sentence = description.split(".")[0].strip()
    return first_sentence or description


def _compose_rationale(plan: Sequence[str], question: str, registry: Registry) -> str:
    if not plan:
        return "No available operators to address the request."

    parts: List[str] = []
    requires_sum = _requires_sum(question)
    mentions_prefix = _mentions_prefix(question)

    for index, step in enumerate(plan, start=1):
        summary = _tool_summary(step, registry)
        if index == 1:
            parts.append(f"Step {index}: Use {step} to {summary.lower()}.")
        else:
            parts.append(f"Step {index}: Then call {step} to {summary.lower()}.")

    if requires_sum and mentions_prefix:
        parts.append("This covers fetching prefixed keys and summing their numeric suffixes.")
    elif requires_sum:
        parts.append("This sequence focuses on aggregating numeric suffixes as requested.")

    return " ".join(parts)


def _deduplicate(plans: Iterable[Sequence[str]]) -> List[List[str]]:
    seen = set()
    unique: List[List[str]] = []
    for plan in plans:
        key = tuple(plan)
        if key in seen:
            continue
        seen.add(key)
        unique.append(list(plan))
    return unique


def _fallback_plans(registry: Registry, needed: int) -> List[List[str]]:
    if not registry:
        return [[] for _ in range(needed)]

    tool_names = list(registry.keys())
    fallback_sequences: List[List[str]] = []
    for length in range(1, _MAX_PLAN_LENGTH + 1):
        for combo in product([tool_names[0]], repeat=length):
            fallback_sequences.append(list(combo))
            if len(fallback_sequences) >= needed:
                return fallback_sequences
    return fallback_sequences


def plan_candidates(question: str, registry: Registry, k: int = 3) -> PlanningResult:
    """Generate Tree-of-Thought plan candidates and score them."""

    if k <= 0:
        raise ValueError("k must be positive")

    ordered_tools = _ordered_tools(question, registry)
    generated = _generate_thoughts(question, ordered_tools)
    unique_plans = _deduplicate(generated)

    candidates: List[PlanCandidate] = []
    for plan in unique_plans:
        rationale = _compose_rationale(plan, question, registry)
        candidate = PlanCandidate(plan=plan, rationale=rationale)
        candidate.scores = score_plan(plan, question, registry)
        candidate.raw_score = compute_raw_score(candidate.scores)
        candidates.append(candidate)

    if len(candidates) < k:
        needed = k - len(candidates)
        for fallback_plan in _fallback_plans(registry, needed):
            rationale = _compose_rationale(fallback_plan, question, registry)
            candidate = PlanCandidate(plan=fallback_plan, rationale=rationale)
            candidate.scores = score_plan(fallback_plan, question, registry)
            candidate.raw_score = compute_raw_score(candidate.scores)
            candidates.append(candidate)

    candidates.sort(key=lambda item: item.raw_score, reverse=True)
    candidates = candidates[:k]

    confidences = softmax([candidate.raw_score for candidate in candidates])
    for candidate, confidence in zip(candidates, confidences):
        candidate.confidence = confidence

    chosen = 0
    if candidates:
        chosen = max(range(len(candidates)), key=lambda idx: candidates[idx].raw_score)

    return PlanningResult(candidates=candidates, chosen=chosen)
