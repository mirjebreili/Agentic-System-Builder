import numpy as np
from typing import List, Dict, Any

from src.llm.provider import LLM
from src.utils.types import Registry

def score_plans_in_batch(
    plans: List[List[str]], question: str, registry: Registry
) -> List[Dict[str, float]]:
    """
    Scores a batch of plans using the LLM.
    """
    if not plans:
        return []

    llm = LLM()
    scores = llm.score(plans, question, registry)

    if len(scores) != len(plans):
        raise ValueError("The number of scores returned by the LLM does not match the number of plans.")

    return scores


def softmax_confidences(raw_scores: List[float]) -> List[float]:
    """
    Computes softmax confidences from a list of raw scores.
    A lower temperature makes the distribution sharper (more confident).
    """
    if not raw_scores:
        return []

    temperature = 0.4 # Lowered temperature for a sharper distribution
    scores = np.array(raw_scores) / temperature
    e_scores = np.exp(scores - np.max(scores))
    softmax = e_scores / e_scores.sum()

    return softmax.tolist()