from typing import Dict, Any, Tuple
from prompt2graph.config.settings import settings
from .planner import Plan

def _score_structure(plan: Plan) -> float:
    """
    Scores the plan's structural integrity based on MVP requirements.
    Returns 1.0 if the basic 3-node loop structure is present, 0.0 otherwise.
    """
    try:
        expected_nodes = {"plan", "do", "finish"}
        actual_nodes = {node.id for node in plan.nodes}
        if expected_nodes != actual_nodes:
            return 0.0

        # Check for the essential edges that form the loop
        edges = {(edge.from_, edge.to) for edge in plan.edges}
        if not {("plan", "do"), ("do", "do"), ("do", "finish")}.issubset(edges):
            return 0.0

        # Check that prompts are not empty
        if any(not node.prompt for node in plan.nodes):
            return 0.0

    except (AttributeError, TypeError):
        return 0.0

    return 1.0

def _score_tool_coverage(plan: Plan) -> float:
    """
    Scores tool coverage. This is a placeholder for future implementation
    and will always return 0.0 in the MVP since no tools are used.
    """
    return 0.0

def _score_prior_success(metrics: Dict[str, Any]) -> float:
    """
    Scores based on the historical success rate of the agent.
    Defaults to 0.5 if no history is available.
    """
    attempts = metrics.get("prior_attempts", 0)
    successes = metrics.get("prior_successes", 0)
    if attempts == 0:
        return 0.5  # Default confidence if no history
    return successes / attempts

def compute_plan_confidence(plan: Plan, state: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Computes a blended confidence score for a given plan and returns the
    final score along with the debug terms used in the calculation.
    """
    w = settings.confidence_weights

    self_score = plan.confidence if plan.confidence is not None else 0.5
    struct_score = _score_structure(plan)
    cov_score = _score_tool_coverage(plan)
    prior_score = _score_prior_success(state.get("metrics", {}))

    # Blend the scores using weights from settings
    blended_confidence = (
        self_score * w.conf_w_self +
        struct_score * w.conf_w_struct +
        cov_score * w.conf_w_cov +
        prior_score * w.conf_w_prior
    )

    # Clamp the final score between 0.0 and 1.0
    final_confidence = min(max(blended_confidence, 0.0), 1.0)

    confidence_terms = {
        "self_score": self_score,
        "structural": struct_score,
        "coverage": cov_score,
        "prior": prior_score,
        "final": final_confidence,
        "weights": w.model_dump()
    }

    return final_confidence, confidence_terms
