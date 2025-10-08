from typing import Any, Dict, List
import numpy as np

from src.utils.types import PlanCandidate, Registry
from src.utils.scoring import score_plan, softmax_confidences
from src.llm.provider import LLM

def plan_candidates(question: str, registry: Registry, k: int = 3) -> Dict[str, Any]:
    """
    Generates, scores, and ranks k plan candidates.
    """
    llm = LLM()

    # 1. Propose k candidate plans using the LLM
    # The stubbed LLM will return a fixed list of candidates.
    llm_candidates = llm.plan(question, registry, k)

    candidates: List[PlanCandidate] = []
    raw_scores: List[float] = []

    for cand in llm_candidates:
        plan = cand["plan"]
        rationale = cand["rationale"]

        # 2. Validate I/O and score each plan
        scores = score_plan(plan, question, registry)

        # 3. Compute raw score (mean of individual scores)
        raw_score = np.mean(list(scores.values()))

        # Create a PlanCandidate dictionary, leaving confidence for later
        candidate: PlanCandidate = {
            "plan": plan,
            "rationale": rationale,
            "scores": scores,
            "raw_score": raw_score,
            "confidence": 0.0,  # Placeholder
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
        chosen_index = np.argmax([c["confidence"] for c in candidates])

    return {"candidates": candidates, "chosen": chosen_index}