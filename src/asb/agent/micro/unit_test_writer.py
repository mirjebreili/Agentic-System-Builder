"""Generate smoke tests for each generated node module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import re

from asb.utils.fileops import atomic_write, ensure_dir


TEST_TEMPLATE = """from __future__ import annotations

import importlib
from typing import Callable

from src.agent.state import AppState

NODE_MODULES = {modules}


def _make_state() -> AppState:
    return AppState(messages=[])


def _load(name: str) -> Callable[[AppState], AppState]:
    module = importlib.import_module(f"src.agent.{name}")
    return getattr(module, f"node_{name}")

{tests}
"""


CONFTEST_TEMPLATE = """import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
"""


PYTEST_INI_TEMPLATE = """[pytest]
pythonpath = src
"""


def _determine_project_root(state: Dict[str, Any]) -> Path | None:
    scaffold = state.get("scaffold")
    if isinstance(scaffold, dict):
        path = scaffold.get("path")
        if path:
            return Path(path)
    candidate = state.get("project_root")
    if candidate:
        return Path(str(candidate))
    return None


def _norm_node(data: Dict[str, Any]) -> Dict[str, Any]:
    name = (data.get("name") or data.get("node_id") or data.get("id") or "").strip()
    if not name:
        base = (data.get("responsibility") or "node").split()[0]
        name = re.sub(r"[^A-Za-z0-9_]+", "_", base).strip("_") or "node"
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    data["name"] = name
    return data


def _iter_plan_modules(state: Dict[str, Any]) -> List[str]:
    plan = state.get("plan")
    modules: List[str] = []
    if isinstance(plan, dict):
        nodes = plan.get("nodes")
        if isinstance(nodes, list):
            seen: set[str] = set()
            for entry in nodes:
                if not isinstance(entry, dict):
                    continue
                normalized = _norm_node(dict(entry))
                name = normalized.get("name")
                if isinstance(name, str) and name and name not in seen:
                    modules.append(name)
                    seen.add(name)
    return modules


def unit_test_writer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Write pytest modules that import and execute each node once."""

    working_state: Dict[str, Any] = dict(state or {})
    project_root = _determine_project_root(working_state)
    if project_root is None:
        return working_state

    modules = _iter_plan_modules(working_state)
    if not modules:
        return working_state

    tests_dir = project_root / "tests"
    ensure_dir(tests_dir)

    test_blocks: List[str] = []
    for module in modules:
        block = (
            f"def test_node_{module}_executes() -> None:\n"
            f"    node = _load('{module}')\n"
            "    state = _make_state()\n"
            "    result = node(state)\n"
            "    assert 'messages' in result\n"
            "    assert isinstance(result['messages'], list)\n"
            "\n"
        )
        test_blocks.append(block)

    rendered = TEST_TEMPLATE.format(
        modules=json.dumps(modules),
        tests="".join(test_blocks),
    )

    target = tests_dir / "test_nodes.py"
    atomic_write(target, rendered)

    conftest_path = tests_dir / "conftest.py"
    wrote_conftest = False
    if not conftest_path.exists():
        atomic_write(conftest_path, CONFTEST_TEMPLATE)
        wrote_conftest = True

    pytest_ini_path = project_root / "pytest.ini"
    wrote_pytest_ini = False
    if not pytest_ini_path.exists():
        atomic_write(pytest_ini_path, PYTEST_INI_TEMPLATE)
        wrote_pytest_ini = True

    scratch = dict(working_state.get("scratch") or {})
    artifacts = scratch.setdefault("artifacts", {})
    tests_artifacts: Dict[str, Any] = {
        "module": str(target),
    }
    if wrote_conftest or conftest_path.exists():
        tests_artifacts["conftest"] = str(conftest_path)
    if wrote_pytest_ini or pytest_ini_path.exists():
        tests_artifacts["pytest_ini"] = str(pytest_ini_path)
    artifacts["tests"] = tests_artifacts
    working_state["scratch"] = scratch
    return working_state


__all__ = ["unit_test_writer_node"]
