from __future__ import annotations

from pathlib import Path

from asb.scaffold.repair_nodes import (
    fix_empty_nodes,
    fix_graph_compilation,
    fix_import_errors,
)


def _create_project(tmp_path: Path) -> tuple[dict, Path, Path]:
    project = tmp_path / "project"
    agent_dir = project / "src" / "agent"
    llm_dir = project / "src" / "llm"

    agent_dir.mkdir(parents=True)
    llm_dir.mkdir(parents=True)
    (project / "src").mkdir(exist_ok=True)

    (project / "src" / "__init__.py").write_text("", encoding="utf-8")
    (agent_dir / "__init__.py").write_text("", encoding="utf-8")
    (llm_dir / "__init__.py").write_text("", encoding="utf-8")
    (agent_dir / "state.py").write_text("class AppState(dict):\n    pass\n", encoding="utf-8")
    (llm_dir / "client.py").write_text(
        "def get_chat_model():\n"
        "    class _Dummy:\n"
        "        def invoke(self, *args, **kwargs):\n"
        "            return None\n"
        "    return _Dummy()\n",
        encoding="utf-8",
    )

    state: dict = {
        "scaffold": {"path": str(project), "errors": [], "ok": False},
    }
    return state, project, agent_dir


def test_fix_empty_nodes_generates_fallback(tmp_path: Path) -> None:
    state, _project, agent_dir = _create_project(tmp_path)
    (agent_dir / "alpha.py").write_text("", encoding="utf-8")

    state["plan"] = {"goal": "Verify empty node repair"}
    state["architecture"] = {
        "graph_structure": [
            {"id": "alpha", "description": "Alpha node handles validation."}
        ]
    }
    state["scaffold"]["errors"].append("alpha: node file is empty after write")

    fix_empty_nodes(state)

    repaired_source = (agent_dir / "alpha.py").read_text(encoding="utf-8")
    assert "from .state import AppState" in repaired_source
    assert "from ..llm import client" in repaired_source

    repairs = state["scaffold"]["repairs"]["empty_nodes"]
    assert repairs["success"] is True
    assert "alpha" in repairs["fixed"]
    assert repairs["errors"] == []
    assert not state["scaffold"].get("errors")
    assert state["scaffold"]["ok"] is True


def test_fix_import_errors_inserts_missing_imports(tmp_path: Path) -> None:
    state, _project, agent_dir = _create_project(tmp_path)
    (agent_dir / "beta.py").write_text(
        "from __future__ import annotations\n\n"
        "def beta(state):\n"
        "    return state\n",
        encoding="utf-8",
    )

    state["scaffold"]["errors"].append(
        "beta: missing required imports: from .state import AppState, from ..llm import client"
    )

    fix_import_errors(state)

    repaired_source = (agent_dir / "beta.py").read_text(encoding="utf-8")
    assert "from .state import AppState" in repaired_source
    assert "from ..llm import client" in repaired_source

    repairs = state["scaffold"]["repairs"]["import_errors"]
    assert repairs["success"] is True
    assert "beta" in repairs["fixed"]
    assert repairs["errors"] == []
    assert not state["scaffold"].get("errors")
    assert state["scaffold"]["ok"] is True


def test_fix_graph_compilation_rewrites_graph(tmp_path: Path) -> None:
    state, _project, agent_dir = _create_project(tmp_path)

    (agent_dir / "gamma.py").write_text(
        "from __future__ import annotations\n\n"
        "from .state import AppState\n\n"
        "def gamma(state: AppState) -> AppState:\n"
        "    return state\n",
        encoding="utf-8",
    )
    graph_path = agent_dir / "graph.py"
    graph_path.write_text("raise RuntimeError('broken graph')\n", encoding="utf-8")

    state["architecture"] = {"plan": {"nodes": [{"id": "gamma"}], "edges": []}}
    state["scaffold"]["errors"].append("graph: generate_dynamic_workflow failed: boom")

    fix_graph_compilation(state)

    repaired_source = graph_path.read_text(encoding="utf-8")
    assert "ARCHITECTURE_STATE" in repaired_source
    assert "def generate_dynamic_workflow" in repaired_source

    repairs = state["scaffold"]["repairs"]["graph_compilation"]
    assert repairs["success"] is True
    assert "graph.py" in repairs["fixed"]
    assert repairs["errors"] == []
    assert not state["scaffold"].get("errors")
    assert state["scaffold"]["ok"] is True
