import os
import json
from typing import Any, Dict, List

class StubLLM:
    """A stub LLM for testing without credentials."""

    def _plan_response(self):
        return {
            "candidates": [
                {
                    "plan": ["HttpBasedAtlasReadByKey", "membasedAtlasKeyStreamAggregator"],
                    "rationale": "Fetch Atlas data; aggregate numeric suffixes by prefix."
                },
                {
                    "plan": ["HttpBasedAtlasReadByKey"],
                    "rationale": "Fetch Atlas data."
                }
            ]
        }

    def _score_response(self, plans: List[List[str]]):
        """
        Returns tweaked scores to ensure a clear winner for testing confidence.
        """
        scores = []
        for plan in plans:
            if len(plan) == 2: # The good plan
                scores.append({"coverage": 0.95, "io": 1.0, "simplicity": 0.8, "constraints": 0.95})
            else: # The less good plan
                scores.append({"coverage": 0.4, "io": 1.0, "simplicity": 0.7, "constraints": 0.9})
        return {"scores": scores}

    def invoke(self, *args, **kwargs) -> Dict[str, Any]:
        """
        A simple router for the stub LLM. Distinguishes calls by unique keys.
        """
        is_scoring_call = "plans" in kwargs
        is_planning_call = "k" in kwargs

        if is_scoring_call:
            return self._score_response(kwargs["plans"])
        elif is_planning_call:
            return self._plan_response()

        return {}


def get_llm():
    """
    Returns an LLM provider, falling back to a stub if not configured.
    """
    provider = os.environ.get("LLM_PROVIDER")
    if not provider:
        return StubLLM()
    return StubLLM()


class LLM:
    """A wrapper for a generic LLM provider."""

    def __init__(self):
        self.client = get_llm()

    def plan(self, question: str, registry: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        response = self.client.invoke(question=question, registry=registry, k=k)
        return response.get("candidates", [])

    def score(self, plans: List[List[str]], question: str, registry: Dict[str, Any]) -> List[Dict[str, float]]:
        response = self.client.invoke(plans=plans, question=question, registry=registry)
        return response.get("scores", [])