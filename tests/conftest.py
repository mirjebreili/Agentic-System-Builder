import importlib
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from asb.llm.client import get_chat_model
# Provide lightweight compatibility shims for legacy package names used in tests
prompt2graph_pkg = types.ModuleType("prompt2graph")
sys.modules.setdefault("prompt2graph", prompt2graph_pkg)

prompt2graph_llm_pkg = types.ModuleType("prompt2graph.llm")
sys.modules.setdefault("prompt2graph.llm", prompt2graph_llm_pkg)

prompt2graph_llm = types.ModuleType("prompt2graph.llm.client")
prompt2graph_llm.get_chat_model = get_chat_model
sys.modules.setdefault("prompt2graph.llm.client", prompt2graph_llm)

_STATIC_PLAN = {
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
    "confidence": 0.9,
}

_STATIC_ARCHITECTURE = {
    "graph_structure": [
        {"id": "entry", "type": "input", "description": "Initial state"},
        {"id": "analyze", "type": "llm", "description": "Review requirements"},
        {"id": "design", "type": "llm", "description": "Outline architecture"},
        {"id": "finish", "type": "output", "description": "Provide summary"},
    ],
    "state_flow": {
        "requirements": "entry -> analyze",
        "architecture": "design -> finish",
    },
    "conditional_edges": [
        {"from": "design", "to": "finish", "condition": "design_complete"},
    ],
    "entry_exit_points": {"entry": ["entry"], "exit": ["finish"]},
}

def _static_plan_tot(state: dict) -> dict:
    return {
        "plan": json.loads(json.dumps(_STATIC_PLAN)),
        "messages": list(state.get("messages", [])),
        "flags": {"more_steps": True, "steps_done": False},
    }

prompt2graph_planner = types.ModuleType("prompt2graph.agent.planner")
prompt2graph_planner.plan_tot = _static_plan_tot
sys.modules.setdefault("prompt2graph.agent.planner", prompt2graph_planner)

prompt2graph_agent_pkg = types.ModuleType("prompt2graph.agent")
sys.modules.setdefault("prompt2graph.agent", prompt2graph_agent_pkg)

src_pkg = types.ModuleType("src")
sys.modules.setdefault("src", src_pkg)

src_agent_pkg = types.ModuleType("src.agent")
sys.modules.setdefault("src.agent", src_agent_pkg)
setattr(src_pkg, "agent", src_agent_pkg)

for module_name in ("executor", "planner", "architecture_designer"):
    asb_module = importlib.import_module(f"asb.agent.{module_name}")
    shim_name = f"src.agent.{module_name}"
    sys.modules.setdefault(shim_name, asb_module)
    setattr(src_agent_pkg, module_name, asb_module)

class FakeResponse:
    def __init__(self, content: str):
        self.content = content

class FakeChatModel:
    """Deterministic stand‑in for the real chat model."""

    def __init__(self):
        self._plan_json = json.dumps(_STATIC_PLAN)
        self._architecture_json = json.dumps(_STATIC_ARCHITECTURE)

    def invoke(self, messages, **kwargs):
        system_content = messages[0].content if messages else ""
        system_lower = system_content.lower()
        if "architecture" in system_lower or "graph_structure" in system_lower:
            return FakeResponse(self._architecture_json)
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
    monkeypatch.setattr("asb.agent.architecture_designer.get_chat_model", lambda **kwargs: fake)
    monkeypatch.setattr(
        sys.modules["prompt2graph.llm.client"],
        "get_chat_model",
        lambda **kwargs: fake,
        raising=False,
    )

    def _fake_prompt_plan(state: dict) -> dict:
        plan_data = json.loads(fake._plan_json)
        return {
            "plan": plan_data,
            "messages": list(state.get("messages", [])),
            "flags": {"more_steps": True, "steps_done": False},
        }

    monkeypatch.setattr(
        sys.modules["prompt2graph.agent.planner"],
        "plan_tot",
        _fake_prompt_plan,
        raising=False,
    )
    if "tests.test_planner" in sys.modules:
        monkeypatch.setattr(
            sys.modules["tests.test_planner"],
            "plan_tot",
            _fake_prompt_plan,
            raising=False,
        )
    return fake
