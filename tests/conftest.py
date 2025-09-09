import json
import pytest
import sys
import types
from pathlib import Path

# Expose the project under the expected package name `prompt2graph`
ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(ROOT))
pkg = types.ModuleType("prompt2graph")
pkg.__path__ = [str(ROOT)]
sys.modules.setdefault("prompt2graph", pkg)


class FakeResponse:
    def __init__(self, content: str):
        self.content = content

    def result(self):
        # mimic the future interface used in plan_tot
        return self


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
            }
        )

    def ainvoke(self, messages):
        return FakeResponse(self._plan_json)

    def invoke(self, messages):
        system_content = messages[0].content if messages else ""
        if "plan evaluator" in system_content.lower():
            return FakeResponse('{"score": 1.0, "reason": "looks good"}')
        return FakeResponse(self._plan_json)


@pytest.fixture(autouse=True)
def mock_chat_model(monkeypatch):
    """Automatically patch get_chat_model to return the fake model."""
    fake = FakeChatModel()
    monkeypatch.setattr("prompt2graph.llm.client.get_chat_model", lambda: fake)
    # planner imports get_chat_model directly, so patch it as well
    monkeypatch.setattr("prompt2graph.agent.planner.get_chat_model", lambda: fake, raising=False)
    return fake
