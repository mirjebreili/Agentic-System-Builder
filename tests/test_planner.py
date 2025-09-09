import sys
import types

# Provide a dummy compute_plan_confidence to avoid circular imports
conf_stub = types.ModuleType("prompt2graph.agent.confidence")
conf_stub.compute_plan_confidence = lambda plan, state: (0.5, {})
sys.modules.setdefault("prompt2graph.agent.confidence", conf_stub)

from prompt2graph.agent.planner import plan_tot


def test_plan_tot_uses_fake_model():
    state = {"messages": [{"role": "user", "content": "Build a plan"}]}
    new_state = plan_tot(state)

    assert new_state["plan"]["goal"] == "Test goal"
    assert new_state["plan"]["nodes"][0]["id"] == "plan"
    assert "confidence" in new_state["plan"]
