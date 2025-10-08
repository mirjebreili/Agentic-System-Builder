from typing import Any, Dict, List
import numpy as np

from src.utils.types import PlanCandidate, Registry
from src.utils.scoring import score_plans_in_batch, softmax_confidences
from src.llm.provider import LLM

def plan_candidates(question: str, registry: Registry, k: int = 3) -> Dict[str, Any]:
    """
    Generates, scores, and ranks k plan candidates.
    Ensures all numeric types are Python native for serialization.
    """
    llm = LLM()

    # 1. Propose k candidate plans
    llm_candidates = llm.plan(question, registry, k)

    if not llm_candidates:
        return {"candidates": [], "chosen": -1}

    # 2. Score all plans in a single batch
    plans_to_score = [cand["plan"] for cand in llm_candidates]
    all_scores = score_plans_in_batch(plans_to_score, question, registry)

    candidates: List[PlanCandidate] = []
    raw_scores: List[float] = []

    # 3. Process each candidate, ensuring native Python types
    for i, cand in enumerate(llm_candidates):
        scores = all_scores[i]

        # Explicitly cast numpy float to Python float
        raw_score = float(np.mean(list(scores.values())))

        candidate: PlanCandidate = {
            "plan": cand["plan"],
            "rationale": cand["rationale"],
            "scores": scores,
            "raw_score": raw_score,
            "confidence": 0.0,
        }
        candidates.append(candidate)
        raw_scores.append(raw_score)

    # 4. Compute softmax confidence scores
    if raw_scores:
        confidences = softmax_confidences(raw_scores)
        for i, candidate in enumerate(candidates):
            # Cast to Python float
            candidate["confidence"] = float(confidences[i])

    # 5. Determine the chosen plan
    chosen_index = -1
    if candidates:
        # Cast numpy int to Python int
        chosen_index = int(np.argmax([c["confidence"] for c in candidates]))

    return {"candidates": candidates, "chosen": chosen_index}