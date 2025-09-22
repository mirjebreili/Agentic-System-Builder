import importlib
import json
import shutil
import sys
from pathlib import Path

import pytest

from asb.agent import scaffold


TEMPLATE_FILES = {
    "src/config/settings.py": "SETTING = True\n",
    "src/asb/llm/client.py": "def get_chat_model():\n    return object()\n",
    "src/asb/agent/state.py": "class AgentState(dict):\n    pass\n",
    "src/asb/agent/prompts_util.py": (
        "from pathlib import Path\n\n"
        "def find_prompts_dir() -> Path:\n"
        "    return Path(__file__).resolve().parents[2] / \"prompts\"\n"
    ),
}

UPDATED_DEPENDENCIES = [
    '  "langgraph>=0.6,<0.7",',
    '  "langchain-core>=0.3,<0.4",',
    '  "langchain-openai>=0.3,<0.4",',
    '  "pydantic>=2.7,<3",',
    '  "langgraph-checkpoint-sqlite>=2.0.0",',
    '  "aiosqlite>=0.17.0",',
    '  "pytest>=7.0.0",',
    '  "langgraph-cli[inmem]>=0.1.0",',
    '  "requests>=2.25.0",',
    '  "black>=22.0.0",',
    '  "isort>=5.0.0",',
    '  "mypy>=1.0.0",',
    '  "bandit[toml]>=1.7.0",',
]


def _write_template_files(root: Path) -> None:
    for relative, content in TEMPLATE_FILES.items():
        file_path = root / relative
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")


def test_missing_template_records_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(scaffold, "ROOT", tmp_path)
    state: dict = {}
    result = scaffold.scaffold_project(state)
    project_dir = Path(result["scaffold"]["path"])
    try:
        expected = tmp_path / "src/config/settings.py"
        missing = result["scaffold"].get("missing", [])
        assert str(expected) in missing
    finally:
        if project_dir.exists():
            shutil.rmtree(project_dir)


def test_scaffold_project_generates_expected_files(tmp_path, monkeypatch):
    monkeypatch.setattr(scaffold, "ROOT", tmp_path)
    _write_template_files(tmp_path)

    state = {"plan": {"goal": "Temporary Agent"}}
    result = scaffold.scaffold_project(state)
    project_dir = Path(result["scaffold"]["path"])

    try:
        assert project_dir.exists()

        graph_contents = (project_dir / "src" / "agent" / "graph.py").read_text(encoding="utf-8")
        assert "from langgraph.checkpoint.sqlite import SqliteSaver" in graph_contents
        assert "from .state import AppState" in graph_contents
        assert "StateGraph(AppState)" in graph_contents

        langgraph_config = json.loads((project_dir / "langgraph.json").read_text(encoding="utf-8"))
        assert langgraph_config["graphs"]["agent"] == "src.agent.graph:graph"

        for package_file in (
            "src/__init__.py",
            "src/agent/__init__.py",
            "src/llm/__init__.py",
            "src/config/__init__.py",
        ):
            assert (project_dir / package_file).exists()

        smoke_contents = (project_dir / "tests" / "test_smoke.py").read_text(encoding="utf-8")
        assert '"""Smoke tests for the generated agent project."""' in smoke_contents
        assert "import importlib" in smoke_contents
        assert "from pathlib import Path" in smoke_contents
        assert "def test_import_graph():" in smoke_contents
        assert "def test_state_structure():" in smoke_contents
        assert "def test_graph_execution(tmp_path: Path):" in smoke_contents
        assert 'assert isinstance(result["messages"], list)' in smoke_contents
        assert 'assert any(' in smoke_contents
        assert 'if __name__ == "__main__":' in smoke_contents
        assert "pytest.main([__file__])" in smoke_contents

        pyproject_text = (project_dir / "pyproject.toml").read_text(encoding="utf-8")
        for dependency in UPDATED_DEPENDENCIES:
            assert dependency in pyproject_text
    finally:
        if project_dir.exists():
            shutil.rmtree(project_dir)


def test_scaffold_project_prefers_generated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(scaffold, "ROOT", tmp_path)
    _write_template_files(tmp_path)

    generated_state = "from __future__ import annotations\nSTATE_DEFINED = True\n"
    state = {
        "plan": {"goal": "Generated Agent"},
        "generated_files": {"state.py": generated_state},
    }

    result = scaffold.scaffold_project(state)
    project_dir = Path(result["scaffold"]["path"])

    try:
        state_path = project_dir / "src" / "agent" / "state.py"
        assert state_path.read_text(encoding="utf-8") == generated_state
    finally:
        if project_dir.exists():
            shutil.rmtree(project_dir)


def test_scaffold_project_builds_architecture_modules(tmp_path, monkeypatch):
    monkeypatch.setattr(scaffold, "ROOT", tmp_path)
    _write_template_files(tmp_path)

    architecture = {
        "graph_structure": [
            {"id": "entry", "type": "input"},
            {"id": "analyze", "type": "llm"},
            {"id": "design", "type": "llm"},
            {"id": "finish", "type": "output"},
        ]
    }

    generated_files = {
        "entry.py": (
            "from typing import Any, Dict\n\n"
            "def run(state: Dict[str, Any]) -> Dict[str, Any]:\n"
            "    state['visited'] = state.get('visited', []) + ['entry']\n"
            "    return state\n"
        ),
        "src/agent/analyze.py": (
            "from typing import Any, Dict\n\n"
            "def analyze(state: Dict[str, Any]) -> Dict[str, Any]:\n"
            "    state['visited'] = state.get('visited', []) + ['analyze']\n"
            "    return state\n"
        ),
        "agent/design.py": (
            "from typing import Any, Dict\n\n"
            "def run_design(state: Dict[str, Any]) -> Dict[str, Any]:\n"
            "    state['visited'] = state.get('visited', []) + ['design']\n"
            "    return state\n"
        ),
        "finish.py": (
            "from typing import Any, Dict\n\n"
            "def finish_run(state: Dict[str, Any]) -> Dict[str, Any]:\n"
            "    state['visited'] = state.get('visited', []) + ['finish']\n"
            "    return state\n"
        ),
    }

    state = {
        "plan": {"goal": "Architecture Agent"},
        "architecture": architecture,
        "generated_files": generated_files,
    }

    result = scaffold.scaffold_project(state)
    project_dir = Path(result["scaffold"]["path"])

    try:
        agent_dir = project_dir / "src" / "agent"
        for filename, contents in generated_files.items():
            module_name = filename.split("/")[-1]
            module_path = agent_dir / module_name
            assert module_path.exists(), f"Missing module for {module_name}"
            assert module_path.read_text(encoding="utf-8") == contents

        executor_text = (agent_dir / "executor.py").read_text(encoding="utf-8")
        assert "_NODE_SPECS" in executor_text
        assert "('entry', 'entry'" in executor_text
        assert "('finish', 'finish'" in executor_text

        monkeypatch.syspath_prepend(str(project_dir))
        monkeypatch.setenv("LANGGRAPH_ENV", "cloud")

        import langgraph.graph as langgraph_graph

        class DummyStateGraph:
            last_instance = None

            def __init__(self, state_cls):
                self.state_cls = state_cls
                self.nodes = []
                self.edges = []
                self._compiled = False
                DummyStateGraph.last_instance = self

            def add_node(self, name, func):
                self.nodes.append(name)

            def add_edge(self, source, target):
                self.edges.append((source, target))

            def compile(self, *, checkpointer=None):
                self._compiled = True
                self.checkpointer = checkpointer
                return self

        monkeypatch.setattr(langgraph_graph, "StateGraph", DummyStateGraph)

        removed_modules = {}
        for name in list(sys.modules):
            if name == "src" or name.startswith("src."):
                removed_modules[name] = sys.modules.pop(name)

        try:
            graph_module = importlib.import_module("src.agent.graph")
            dummy = DummyStateGraph.last_instance
            assert dummy is not None
            assert dummy.nodes == ["entry", "analyze", "design", "finish"]
            expected_edges = [
                (langgraph_graph.START, "entry"),
                ("entry", "analyze"),
                ("analyze", "design"),
                ("design", "finish"),
                ("finish", langgraph_graph.END),
            ]
            assert dummy.edges == expected_edges

            executor_module = importlib.import_module("src.agent.executor")
            node_order = [node_id for node_id, _ in executor_module.NODE_IMPLEMENTATIONS]
            assert node_order == ["entry", "analyze", "design", "finish"]
        finally:
            for name in list(sys.modules):
                if name == "src" or name.startswith("src."):
                    if name not in removed_modules:
                        sys.modules.pop(name)
            for name, module in removed_modules.items():
                sys.modules[name] = module
    finally:
        if project_dir.exists():
            shutil.rmtree(project_dir)


def test_scaffold_project_generates_tot_templates(tmp_path, monkeypatch):
    monkeypatch.setattr(scaffold, "ROOT", tmp_path)
    _write_template_files(tmp_path)

    architecture = {
        "graph_structure": [
            {"id": "generate_thoughts"},
            {"id": "evaluate_thoughts"},
            {"id": "select_best_thought"},
            {"id": "final_answer"},
        ]
    }

    state = {
        "plan": {"goal": "Tree of Thoughts Agent"},
        "architecture": architecture,
        "generated_files": {},
    }

    result = scaffold.scaffold_project(state)
    project_dir = Path(result["scaffold"]["path"])

    try:
        agent_dir = project_dir / "src" / "agent"
        utils_path = agent_dir / "utils.py"
        utils_text = utils_path.read_text(encoding="utf-8")
        assert "def extract_input_text" in utils_text
        assert "def parse_approaches" in utils_text
        assert "def score_thoughts" in utils_text
        assert "def capture_tot_error" in utils_text

        generate_text = (agent_dir / "generate_thoughts.py").read_text(encoding="utf-8")
        assert "from ..llm import client" in generate_text
        assert "client.get_chat_model()" in generate_text
        assert "parse_approaches" in generate_text
        assert "capture_tot_error" in generate_text

        evaluate_text = (agent_dir / "evaluate_thoughts.py").read_text(encoding="utf-8")
        assert "score_thoughts" in evaluate_text
        assert "get_thoughts" in evaluate_text
        assert "client.get_chat_model()" in evaluate_text

        select_text = (agent_dir / "select_best_thought.py").read_text(encoding="utf-8")
        assert "select_top_evaluation" in select_text
        assert "capture_tot_error" in select_text

        final_text = (agent_dir / "final_answer.py").read_text(encoding="utf-8")
        assert "client.get_chat_model()" in final_text
        assert "get_selected_thought" in final_text
        assert "update_tot_state" in final_text
    finally:
        if project_dir.exists():
            shutil.rmtree(project_dir)
