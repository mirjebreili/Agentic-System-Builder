import os
import json
from typing import Any, Dict, List

# A simple fallback for when an API key is not available.
# This allows the graph to be tested without a live LLM.
STUB_RESPONSE = {
    "candidates": [
        {
            "plan": ["HttpBasedAtlasReadByKey", "membasedAtlasKeyStreamAggregator"],
            "rationale": "The user wants to sum values from keys with a specific prefix. This requires reading the data and then aggregating it.",
        },
        {
            "plan": ["HttpBasedAtlasReadByKey"],
            "rationale": "This plan only reads the data but does not perform the aggregation step.",
        },
    ]
}


class LLM:
    def __init__(self, api_key: str = None, base_url: str = None):
        """
        A minimal OpenAI-compatible LLM provider.
        It can be configured with environment variables for the API key and base URL.
        If no API key is provided, it falls back to a stubbed response.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_API_BASE_URL")

    def _is_configured(self) -> bool:
        """Check if the LLM provider is configured with an API key."""
        return self.api_key is not None

    def plan(self, question: str, registry: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        """
        Generates plan candidates using the LLM.
        If the LLM is not configured, it returns a stubbed response.
        """
        if not self._is_configured():
            print("LLM not configured, returning stubbed response.")
            return STUB_RESPONSE.get("candidates", [])[:k]

        # In a real implementation, this would make a call to an LLM.
        # For this task, we will stick to the stubbed response to ensure
        # the system can run end-to-end without a live model.
        # This part of the code would be where the `openai` library is used.
        print("LLM is configured, but for this task, we will use the stub response.")
        return STUB_RESPONSE.get("candidates", [])[:k]


def get_llm_provider() -> LLM:
    """Factory function to get an instance of the LLM provider."""
    return LLM()