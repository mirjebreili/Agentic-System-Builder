import numpy as np
from typing import List, Dict, Any
from src.utils.types import Registry

def score_plan(plan: List[str], question: str, registry: Registry) -> Dict[str, float]:
    """
    Scores a plan based on coverage, I/O compatibility, simplicity, and constraints.
    """
    # Check for I/O compatibility
    io_score = 1.0
    for i in range(len(plan) - 1):
        producer_spec = registry.get(plan[i])
        consumer_spec = registry.get(plan[i+1])
        if producer_spec and consumer_spec:
            if producer_spec['role'] == 'producer' and consumer_spec['role'] not in ['consumer', 'transformer', 'mixed']:
                io_score = 0.0
                break

    # Coverage score
    has_producer = any(registry.get(p, {}).get('role') == 'producer' for p in plan)
    has_consumer = any(registry.get(p, {}).get('role') in ['consumer', 'transformer', 'mixed'] for p in plan)
    needs_aggregation = "مجموع" in question or "sum" in question or "aggregate" in question

    coverage_score = 0.5
    if has_producer and has_consumer:
        coverage_score = 0.95
    elif has_producer and needs_aggregation:
        # Penalize if aggregation is needed but not provided
        coverage_score = 0.2
    elif has_producer:
        coverage_score = 0.8
    else:
        coverage_score = 0.1

    # Simplicity score is only a virtue for plans with good coverage
    if coverage_score < 0.8:
        simplicity_score = 0.2
    else:
        simplicity_score = max(0, 1.0 - 0.2 * (len(plan) - 1))

    # Constraints score (dummy)
    constraints_score = 0.95

    return {
        "coverage": coverage_score,
        "io": io_score,
        "simplicity": simplicity_score,
        "constraints": constraints_score
    }

def softmax_confidences(raw_scores: List[float]) -> List[float]:
    """
    Computes softmax confidences from a list of raw scores.
    """
    if not raw_scores:
        return []

    # A lower temperature makes the distribution sharper (more confident)
    temperature = 0.8
    scores = np.array(raw_scores) / temperature
    e_scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
    softmax = e_scores / e_scores.sum()

    return softmax.tolist()