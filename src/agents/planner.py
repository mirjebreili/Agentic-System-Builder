from typing import Any, Dict, List
import numpy as np

from src.utils.types import PlanCandidate, Registry
from src.utils.scoring import score_plan, softmax_confidences
from src.llm.provider import get_llm_provider

def plan_candidates(question: str, registry: Registry, k: int = 3) -> Dict[str, Any]:
    """
    Generates, scores, and ranks k plan candidates.
    """
    llm = get_llm_provider()

    # 1. Propose k candidate plans using the LLM
    llm_candidates = llm.plan(question, registry, k)

    candidates: List[PlanCandidate] = []
    raw_scores: List[float] = []

    for cand in llm_candidates:
        plan = cand.get("plan", [])
        rationale = cand.get("rationale", "No rationale provided.")

        # 2. Validate I/O and score each plan
        scores = score_plan(plan, question, registry)

        # 3. Compute raw score (mean of individual scores)
        raw_score = np.mean(list(scores.values())) if scores else 0.0

        # Create a PlanCandidate dictionary
        candidate: PlanCandidate = {
            "plan": plan,
            "rationale": rationale,
            "scores": scores,
            "raw_score": raw_score,
            "confidence": 0.0,  # Placeholder, will be updated next
        }
        candidates.append(candidate)
        raw_scores.append(raw_score)

    # 4. Compute softmax confidence scores for all candidates
    if raw_scores:
        confidences = softmax_confidences(raw_scores)
        for i, candidate in enumerate(candidates):
            candidate["confidence"] = confidences[i]

    # 5. Determine the chosen plan (argmax confidence)
    chosen_index = -1
    if candidates:
        # Sort candidates by confidence in descending order
        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        chosen_index = 0  # The top candidate is now at index 0

    return {"candidates": candidates, "chosen": chosen_index}