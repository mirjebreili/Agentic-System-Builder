import numpy as np
from typing import List, Dict, Any
from src.utils.types import Registry

def score_plan(plan: List[str], question: str, registry: Registry) -> Dict[str, float]:
    """
    Scores a plan based on coverage, I/O compatibility, simplicity, and constraints.
    """
    # I/O Compatibility Score
    io_score = 1.0
    for i in range(len(plan) - 1):
        producer_name = plan[i]
        consumer_name = plan[i+1]

        producer_spec = registry.get(producer_name)
        consumer_spec = registry.get(consumer_name)

        if not producer_spec or not consumer_spec:
            io_score = 0.0
            break

        is_producer = producer_spec.get('role') in ['producer', 'mixed']
        is_consumer = consumer_spec.get('role') in ['consumer', 'transformer', 'mixed']

        if is_producer and not is_consumer:
            # A producer must be followed by a consumer or transformer
            io_score = 0.0
            break

        producer_output_type = producer_spec.get('outputs', {}).get('type')
        if producer_output_type not in ['stream/json', 'json', 'number'] and is_consumer:
            # If a consumer follows a non-data-producing tool
            io_score = 0.0
            break

    # Coverage Score
    has_producer = any(registry.get(p, {}).get('role') == 'producer' for p in plan)
    has_consumer = any(registry.get(p, {}).get('role') in ['consumer', 'transformer', 'mixed'] for p in plan)

    # Simple check for whether the question implies aggregation
    needs_aggregation = "sum" in question.lower() or "مجموع" in question

    coverage_score = 0.0
    if has_producer and needs_aggregation and has_consumer:
        coverage_score = 1.0  # Best case: has a producer and a consumer for an aggregation task
    elif has_producer and not needs_aggregation:
        coverage_score = 0.9 # Good if it has a producer and no aggregation is needed
    elif has_producer and needs_aggregation and not has_consumer:
        coverage_score = 0.4 # Penalize heavily if aggregation is needed but no consumer is present
    elif not has_producer:
        coverage_score = 0.1 # Very low score if no data source is identified

    # Simplicity Score
    # A shorter plan is simpler. We penalize longer plans.
    # The penalty is harsher if coverage is low.
    simplicity_score = max(0.0, 1.0 - 0.1 * (len(plan) - 1))
    if coverage_score < 0.5:
        simplicity_score *= 0.5 # Further penalize simplicity if coverage is poor

    # Constraints Score (dummy implementation for now)
    # This could be expanded to check for specific constraints mentioned in the question.
    constraints_score = 0.9

    return {
        "coverage": coverage_score,
        "io": io_score,
        "simplicity": simplicity_score,
        "constraints": constraints_score,
    }

def softmax_confidences(raw_scores: List[float]) -> List[float]:
    """
    Computes softmax confidences from a list of raw scores.
    """
    if not raw_scores:
        return []

    scores = np.array(raw_scores)
    e_scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
    softmax = e_scores / e_scores.sum()

    return softmax.tolist()