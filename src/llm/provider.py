import json
from typing import Any, Dict, List

from src.settings import settings
import os

try:
    import httpx
except Exception:
    httpx = None

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
    def __init__(self):
        """A minimal LLM provider wired to project settings."""
        self.api_key = settings.OPENAI_API_KEY
        self.base_url = settings.OPENAI_BASE_URL
        self.model = settings.MODEL
        self.temperature = settings.TEMPERATURE

    def _is_configured(self) -> bool:
        """Check if the LLM provider is configured with an API key."""
        return bool(self.api_key)

    def plan(self, question: str, registry: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        """Generates plan candidates using the LLM or the stub if not configured."""
        if not self._is_configured():
            print("LLM not configured, returning stubbed response.")
            return STUB_RESPONSE.get("candidates", [])[:k]

        print(f"LLM configured: base_url={self.base_url}, model={self.model}")

        # Guarded HTTP calls: only perform network requests when explicitly allowed.
        allow_requests = os.getenv("HTTPX_ALLOW_REQUESTS", "0") in ("1", "true", "True")
        if not allow_requests or httpx is None:
            print("HTTP requests disabled or httpx not available — returning stubbed response.")
            return STUB_RESPONSE.get("candidates", [])[:k]

        # Build a simple request payload following an OpenAI-like chat completion schema.
        payload = {
            "model": self.model,
            "temperature": float(self.temperature),
            "messages": [
                {"role": "system", "content": "You are an Agent Planner. Output only JSON as specified."},
                {"role": "user", "content": question},
            ],
            "n": 1,
        }

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            url = self.base_url.rstrip("/") + "/chat/completions"
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            # Try to locate JSON candidates inside the response; this depends on provider.
            # We expect the provider to return choices[0].message.content with JSON.
            choices = data.get("choices") or []
            if choices:
                content = choices[0].get("message", {}).get("content") or choices[0].get("text")
                if content:
                    # attempt to parse JSON out of the text
                    try:
                        parsed = json.loads(content)
                        return parsed.get("candidates", STUB_RESPONSE.get("candidates", []))[:k]
                    except Exception:
                        # fallback: return stub
                        print("Could not parse candidates from LLM response; falling back to stub.")
                        return STUB_RESPONSE.get("candidates", [])[:k]

        except Exception as e:
            print("LLM request failed, falling back to stub:", str(e))
            return STUB_RESPONSE.get("candidates", [])[:k]

        return STUB_RESPONSE.get("candidates", [])[:k]


def get_llm_provider() -> LLM:
    return LLM()