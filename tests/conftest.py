import json
import pytest
from asb.llm.client import get_chat_model

class FakeResponse:
    def __init__(self, content: str):
        self.content = content

class FakeChatModel:
    """Deterministic stand‑in for the real chat model."""

    def __init__(self):
        self._plan_json = json.dumps(
            {
                "goal": "Test goal",
                "nodes": [
                    {"id": "plan", "type": "llm", "prompt": "step 1"},
                    {"id": "do", "type": "llm", "prompt": "step 2"},
                    {"id": "finish", "type": "llm", "prompt": "step 3"},
                ],
                "edges": [
                    {"from": "plan", "to": "do"},
                    {"from": "do", "to": "do", "if": "more_steps"},
                    {"from": "do", "to": "finish", "if": "steps_done"},
                ],
                "confidence": 0.9
            }
        )

    def invoke(self, messages, **kwargs):
        system_content = messages[0].content if messages else ""
        if "score 0..1" in system_content.lower():
            return FakeResponse('{"score": 1.0, "reason": "looks good"}')
        if "json array of plan objects" in system_content.lower():
             # In the ToT planning, we ask for an array of plans
            return FakeResponse(f"[{self._plan_json}]")
        return FakeResponse(self._plan_json)

    def ainvoke(self, messages, **kwargs):
        return self.invoke(messages)


@pytest.fixture(autouse=True)
def mock_chat_model(monkeypatch):
    """Automatically patch get_chat_model to return the fake model for all tests."""
    fake = FakeChatModel()
    monkeypatch.setattr("asb.llm.client.get_chat_model", lambda **kwargs: fake)
    monkeypatch.setattr("asb.agent.planner.get_chat_model", lambda **kwargs: fake)
    monkeypatch.setattr("asb.agent.executor.get_chat_model", lambda **kwargs: fake)
    return fake
