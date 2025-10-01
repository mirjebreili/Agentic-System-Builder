from __future__ import annotations

import importlib
import importlib
import re
import shutil
import sys
from pathlib import Path

from asb.agent import scaffold

TEMPLATE_FILES = {
    "src/asb_config/settings.py": "SETTING = True\n",
    "src/asb/llm/client.py": "def get_chat_model():\n    return object()\n",
    "src/asb/agent/state.py": (
        "from __future__ import annotations\n\n"
        "from typing import Annotated, Any, Dict, List, TypedDict\n"
        "import operator\n\n"
        "from langchain_core.messages import AnyMessage\n"
        "from langgraph.graph import add_messages\n\n\n"
        "class AppState(TypedDict, total=False):\n"
        "    messages: Annotated[List[AnyMessage], add_messages]\n"
        "    goal: str\n"
        "    input_text: str\n"
        "    plan: Dict[str, Any]\n"
        "    architecture: Dict[str, Any]\n\n"
        "    result: str\n"
        "    final_output: str\n\n"
        "    error: str\n"
        "    errors: Annotated[List[str], operator.add]\n"
        "    scratch: Annotated[Dict[str, Any], operator.or_]\n"
        "    scaffold: Annotated[Dict[str, Any], operator.or_]\n"
        "    self_correction: Annotated[Dict[str, Any], operator.or_]\n"
        "    tot: Annotated[Dict[str, Any], operator.or_]\n"
    ),
    "src/asb/agent/prompts_util.py": (
        "from pathlib import Path\n\n"
        "def find_prompts_dir() -> Path:\n"
        "    return Path(__file__).resolve().parents[2] / \"prompts\"\n"
    ),
}


def _write_template_files(root: Path) -> None:
    for relative, content in TEMPLATE_FILES.items():
        file_path = root / relative
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")


def _generate_project(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(scaffold, "ROOT", tmp_path)
    _write_template_files(tmp_path)
    state = {
        "plan": {"goal": "Concurrency Check"},
        "architecture": {
            "nodes": [
                {"id": "plan"},
                {"id": "do"},
                {"id": "finish"},
            ],
            "edges": [
                {"from": "plan", "to": "do"},
                {"from": "do", "to": "finish"},
            ],
        },
    }
    result = scaffold.scaffold_project(state)
    return Path(result["scaffold"]["path"])


def test_graph_draw_no_invalid_concurrent_update(tmp_path, monkeypatch):
    project_dir = _generate_project(tmp_path, monkeypatch)
    sys.path.insert(0, str(project_dir))
    try:
        for module in list(sys.modules):
            if module == "src" or module.startswith("src."):
                sys.modules.pop(module, None)
        graph_module = importlib.import_module("src.agent.graph")
        compiled = graph_module.graph
        compiled.get_graph(xray=True)
    finally:
        sys.path.pop(0)
        for module in list(sys.modules):
            if module == "src" or module.startswith("src."):
                sys.modules.pop(module, None)
        shutil.rmtree(project_dir, ignore_errors=True)


def test_sequential_graph_has_single_start_edge(tmp_path, monkeypatch):
    project_dir = _generate_project(tmp_path, monkeypatch)
    sys.path.insert(0, str(project_dir))
    try:
        for module in list(sys.modules):
            if module == "src" or module.startswith("src."):
                sys.modules.pop(module, None)
        graph_module = importlib.import_module("src.agent.graph")
        start_edges = [
            edge
            for edge in graph_module.graph.get_graph().edges
            if edge.source == "__start__" and edge.target != "__end__"
        ]
        assert len(start_edges) == 1
    finally:
        sys.path.pop(0)
        for module in list(sys.modules):
            if module == "src" or module.startswith("src."):
                sys.modules.pop(module, None)
        shutil.rmtree(project_dir, ignore_errors=True)


def test_state_schema_declares_aggregated_keys(tmp_path, monkeypatch):
    project_dir = _generate_project(tmp_path, monkeypatch)
    try:
        state_text = (project_dir / "src" / "agent" / "state.py").read_text(encoding="utf-8")
        annotated_fields = dict(
            re.findall(
                r"^    ([a-zA-Z0-9_]+): (Annotated\[.*\])$",
                state_text,
                flags=re.MULTILINE,
            )
        )
        assert annotated_fields.get("messages") == "Annotated[List[AnyMessage], add_messages]"
        assert annotated_fields.get("errors") == "Annotated[List[str], operator.add]"
        assert annotated_fields.get("scratch") == "Annotated[Dict[str, Any], operator.or_]"
        assert annotated_fields.get("scaffold") == "Annotated[Dict[str, Any], operator.or_]"
        assert annotated_fields.get("self_correction") == "Annotated[Dict[str, Any], operator.or_]"
        assert annotated_fields.get("tot") == "Annotated[Dict[str, Any], operator.or_]"
    finally:
        shutil.rmtree(project_dir, ignore_errors=True)
