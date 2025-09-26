from __future__ import annotations

import importlib.util
import json
import shutil

import pytest

from asb.agent.micro.sandbox_runner import sandbox_runner_node
from asb.agent.scaffold import generate_enhanced_state_schema


@pytest.mark.skipif(
    shutil.which("langgraph") is None or importlib.util.find_spec("langgraph") is None,
    reason="langgraph CLI or library is not installed",
)
def test_e2e_generate_then_dev(tmp_path) -> None:
    project_root = tmp_path / "project"
    (project_root / "src" / "agent").mkdir(parents=True)
    (project_root / "tests").mkdir()

    langgraph_json = {
        "graphs": {"agent": "src.agent.graph:graph"},
        "dependencies": ["."],
    }
    (project_root / "langgraph.json").write_text(json.dumps(langgraph_json), encoding="utf-8")

    state_py = generate_enhanced_state_schema({})
    (project_root / "src" / "agent" / "state.py").write_text(state_py, encoding="utf-8")

    graph_py = """from __future__ import annotations

from langgraph.graph import StateGraph

from .state import AppState


def _build_graph() -> StateGraph[AppState]:
    graph = StateGraph(AppState)

    def _echo(state: AppState) -> AppState:
        return state

    graph.add_node("echo", _echo)
    graph.set_entry_point("echo")
    return graph


graph = _build_graph().compile()
"""
    (project_root / "src" / "agent" / "graph.py").write_text(graph_py, encoding="utf-8")
    (project_root / "src" / "agent" / "__init__.py").write_text("", encoding="utf-8")

    test_py = """from src.agent.state import AppState


def test_state_import() -> None:
    state = AppState(messages=[])
    assert isinstance(state, dict)
"""
    (project_root / "tests" / "test_state.py").write_text(test_py, encoding="utf-8")

    state = {
        "scaffold": {"path": str(project_root)},
        "sandbox": {},
    }

    result = sandbox_runner_node(state)

    commands = {entry["name"] for entry in result["sandbox"]["last_run"]}
    assert {"meta_langgraph", "project_langgraph", "project_pytest"}.issubset(commands)
