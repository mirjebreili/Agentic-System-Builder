import json
import shutil
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
        "    return Path(__file__).parent\n"
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
        assert "from . import db_setup" in graph_contents

        langgraph_config = json.loads((project_dir / "langgraph.json").read_text(encoding="utf-8"))
        assert langgraph_config["graphs"]["agent"] == "src.agent.graph:graph"

        db_setup_path = project_dir / "src" / "agent" / "db_setup.py"
        assert db_setup_path.exists()
        db_setup_contents = db_setup_path.read_text(encoding="utf-8")
        assert "def ensure_sqlite_db" in db_setup_contents
        assert "def setup_environment" in db_setup_contents
        assert "setup_environment()" in db_setup_contents

        for package_file in (
            "src/__init__.py",
            "src/agent/__init__.py",
            "src/llm/__init__.py",
            "src/config/__init__.py",
        ):
            assert (project_dir / package_file).exists()

        smoke_contents = (project_dir / "tests" / "test_smoke.py").read_text(encoding="utf-8")
        assert "from src.agent.graph import graph" in smoke_contents

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
