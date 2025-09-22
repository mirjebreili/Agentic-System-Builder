import types

from asb.agent import node_implementor
from asb.agent.node_implementor import _select_next_unimplemented_node


def _base_state() -> dict:
    return {
        "architecture": {
            "graph_structure": [
                {"id": "alpha", "type": "llm", "description": "A"},
                {"id": "beta", "type": "tool", "description": "B"},
            ],
            "state_flow": {"alpha": "entry -> alpha"},
            "conditional_edges": [
                {"from": "beta", "to": "gamma", "condition": "ok"},
            ],
        },
        "generated_files": {},
        "messages": [],
    }


class _SequencedFakeModel:
    def __init__(self, responses: dict[str, str]):
        self.responses = responses
        self.invocations: list[str] = []

    def invoke(self, messages, **kwargs):
        user_prompt = messages[-1].content
        self.invocations.append(user_prompt)
        for node_id, code in self.responses.items():
            if f"Node id: {node_id}" in user_prompt:
                return types.SimpleNamespace(content=f"```python\n{code}\n```")
        raise AssertionError(f"Unexpected prompt: {user_prompt}")


def test_select_next_unimplemented_node_skips_existing():
    state = _base_state()
    state["generated_files"] = {"alpha.py": "print('done')"}

    node_id, node = _select_next_unimplemented_node(state)

    assert node_id == "beta"
    assert node.get("description") == "B"


def test_implement_single_node_no_remaining_nodes():
    state = _base_state()
    state["generated_files"] = {"alpha.py": "# alpha", "beta.py": "# beta"}

    updated = node_implementor.node_implementor_node(state)

    assert updated["generated_files"] == state["generated_files"]
    assert updated["messages"][-1]["content"].startswith("[node-implementor]\nNo nodes remaining")
    assert updated.get("implemented_nodes") == []


def test_node_implementor_error_message(monkeypatch):
    state = _base_state()
    monkeypatch.setattr(
        node_implementor,
        "implement_single_node",
        lambda _state: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    updated = node_implementor.node_implementor_node(state)

    assert updated["generated_files"] == state["generated_files"]
    assert updated["messages"][-1]["content"].startswith("[node-implementor-error]")


def test_generated_code_stored_under_expected_key(monkeypatch):
    state = _base_state()
    fake_model = _SequencedFakeModel(
        {
            "alpha": (
                "from typing import Any, Dict\n\n"
                "def run_alpha(state: Dict[str, Any]) -> Dict[str, Any]:\n"
                "    return state\n"
            ),
            "beta": (
                "from typing import Any, Dict\n\n"
                "def run_beta(state: Dict[str, Any]) -> Dict[str, Any]:\n"
                "    return state\n"
            ),
        }
    )

    monkeypatch.setattr(node_implementor, "get_chat_model", lambda: fake_model)

    updated = node_implementor.implement_single_node(state)

    assert set(updated["generated_files"].keys()) == {"alpha.py", "beta.py"}
    assert updated["generated_files"]["alpha.py"].lstrip().startswith("from typing")
    assert updated["generated_files"]["beta.py"].lstrip().startswith("from typing")
    assert updated["implemented_nodes"] == [
        {"node_id": "alpha", "filename": "alpha.py", "new_code": True},
        {"node_id": "beta", "filename": "beta.py", "new_code": True},
    ]
    assert updated["last_implemented_node"] == "beta"
    assert len(fake_model.invocations) == 2


def test_node_with_only_node_key_is_implemented(monkeypatch):
    state = {
        "architecture": {
            "graph_structure": [
                {"node": "Gamma Node", "type": "tool", "description": "C"},
            ]
        },
        "generated_files": {},
        "messages": [],
    }

    class FakeModel:
        def invoke(self, messages, **kwargs):
            return types.SimpleNamespace(
                content=(
                    "```python\n"
                    "def run_gamma(state):\n"
                    "    return state\n"
                    "```"
                )
            )

    monkeypatch.setattr(node_implementor, "get_chat_model", lambda: FakeModel())

    updated = node_implementor.implement_single_node(state)

    assert "Gamma_Node.py" in updated["generated_files"]
    assert updated["implemented_nodes"] == [
        {"node_id": "Gamma Node", "filename": "Gamma_Node.py", "new_code": True}
    ]
    assert updated["last_implemented_node"] == "Gamma Node"


def test_node_implementor_summarizes_all_nodes(monkeypatch):
    state = _base_state()
    fake_model = _SequencedFakeModel(
        {
            "alpha": "def run_alpha(state):\n    return state",
            "beta": "def run_beta(state):\n    return state",
        }
    )

    monkeypatch.setattr(node_implementor, "get_chat_model", lambda: fake_model)

    updated = node_implementor.node_implementor_node(state)

    assert set(updated["generated_files"].keys()) == {"alpha.py", "beta.py"}
    message = updated["messages"][-1]["content"]
    assert "Implemented nodes:" in message
    assert "- alpha (new code) -> alpha.py" in message
    assert "- beta (new code) -> beta.py" in message
    assert len(fake_model.invocations) == 2
