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
        assert "from langgraph.checkpoint.memory import MemorySaver" in graph_contents

        pyproject_text = (project_dir / "pyproject.toml").read_text(encoding="utf-8")
        for dependency in UPDATED_DEPENDENCIES:
            assert dependency in pyproject_text
    finally:
        if project_dir.exists():
            shutil.rmtree(project_dir)
