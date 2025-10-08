import os
import json
from typing import Any, Dict, List


class StubLLM:
    """A stub LLM for testing without credentials."""

    def invoke(self, *args, **kwargs) -> Dict[str, Any]:
        # This stub returns a canned response that fits the planner's needs for the test case.
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

def get_llm():
    """
    Returns an LLM provider based on environment variables.
    Falls back to a stub if no provider is configured.
    """
    provider = os.environ.get("LLM_PROVIDER")

    if not provider:
        print("LLM_PROVIDER not set, using stub LLM.")
        return StubLLM()

    # In a real scenario, you would instantiate a real LLM client here
    # based on the provider and other environment variables (e.g., API keys).
    # For example:
    # if provider == "openai":
    #     from langchain_openai import ChatOpenAI
    #     return ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    # else:
    #     raise ValueError(f"Unsupported LLM provider: {provider}")

    # For this project, we'll just return the stub if any provider is set
    # but not implemented.
    print(f"LLM_PROVIDER set to '{provider}', but only stub is implemented. Using stub LLM.")
    return StubLLM()


class LLM:
    """A wrapper for a generic LLM provider."""

    def __init__(self):
        self.client = get_llm()

    def plan(self, question: str, registry: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        # The actual prompt would be built here using a template from src/prompts/planner.md
        # and passed to the LLM.
        # For the stub, we just call invoke and expect the canned response.
        response = self.client.invoke(question=question, registry=registry, k=k)
        return response.get("candidates", [])

    def score(self, plans: List[List[str]], question: str, registry: Dict[str, Any]) -> List[Dict[str, float]]:
        # Similar to plan, this would use a prompt from src/prompts/scoring.md.
        # For now, we'll return dummy scores.
        return [
            {"coverage": 0.95, "io": 1.0, "simplicity": 0.9, "constraints": 0.95}
            for _ in plans
        ]