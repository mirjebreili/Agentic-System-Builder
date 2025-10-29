from __future__ import annotations
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


def _structural_score(plan: Dict[str, Any]) -> float:
    """Score based on plan complexity - simpler is better."""
    nodes = plan.get("nodes", []) or []
    edges = plan.get("edges", []) or []
    steps = len(nodes)
    
    # Count branching points
    out_counts: Dict[str, int] = {}
    for e in edges:
        src = e.get("from") or e.get("from_")
        if src:
            out_counts[src] = out_counts.get(src, 0) + 1
    branches = sum(1 for count in out_counts.values() if count > 1)
    
    # Start with high score, penalize complexity
    score = 0.9
    score -= max(0, steps - 4) * 0.05  # Penalize plans with > 4 steps
    score -= branches * 0.10  # Penalize branching
    
    return max(0.0, min(1.0, score))


def compute_plan_confidence(state: Dict[str, Any]) -> Dict[str, Any]:
    """Compute confidence score for the plan based on self-assessment and structure."""
    plan = dict(state.get("plan") or {})
    
    # Get LLM's self-assessed confidence
    self_score = float(plan.get("confidence", 0.5) or 0.5)
    
    # Calculate structural score
    structural = _structural_score(plan)
    
    # Weighted combination (70% LLM, 30% structural)
    confidence = 0.7 * self_score + 0.3 * structural
    confidence = max(0.0, min(1.0, confidence))

    logger.info(f"Confidence: LLM={self_score:.3f}, structural={structural:.3f}, final={confidence:.3f}")

    # Store debug information
    debug = dict(state.get("debug") or {})
    debug["confidence_terms"] = {
        "self": round(self_score, 3),
        "structural": round(structural, 3),
        "final": round(confidence, 3),
    }

    plan["confidence"] = confidence
    
    return {
        "plan": plan,
        "debug": debug
    }
