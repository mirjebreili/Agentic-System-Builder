from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[3]

SCAFFOLD_BASE_PATH_KEY = "_scaffold_base_path"
SCAFFOLD_ROOT_KEY = "_scaffold_root_path"


def _get_base_path(state: Dict[str, Any]) -> Path:
    base_path = state.get(SCAFFOLD_BASE_PATH_KEY)
    if base_path is None:
        raise ValueError("Scaffold base path has not been initialized.")
    if isinstance(base_path, Path):
        return base_path
    return Path(str(base_path))


def _get_root(state: Dict[str, Any]) -> Path:
    root_override = state.get(SCAFFOLD_ROOT_KEY)
    if root_override is None:
        return ROOT
    if isinstance(root_override, Path):
        return root_override
    return Path(str(root_override))


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f"{path.name}.tmp"
    if tmp_path.exists():
        tmp_path.unlink()
    with tmp_path.open("wb") as handle:
        handle.write(data)
    os.replace(tmp_path, path)


def _atomic_write_text(path: Path, contents: str) -> None:
    _atomic_write(path, contents.encode("utf-8"))


def _atomic_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.parent / f"{dest.name}.tmp"
    if tmp_path.exists():
        tmp_path.unlink()
    with src.open("rb") as source, tmp_path.open("wb") as target:
        shutil.copyfileobj(source, target)
    os.replace(tmp_path, dest)


def init_project_structure(state: Dict[str, Any]) -> Path:
    base_path = _get_base_path(state)
    directories = [
        base_path,
        base_path / "prompts",
        base_path / "src" / "agent",
        base_path / "src" / "config",
        base_path / "src" / "llm",
        base_path / "tests",
        base_path / "reports",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    return base_path


def copy_base_files(state: Dict[str, Any]) -> List[str]:
    base_path = _get_base_path(state)
    root_path = _get_root(state)
    files = {
        "src/config/settings.py": "src/config/settings.py",
        "src/asb/llm/client.py": "src/llm/client.py",
        "src/asb/agent/prompts_util.py": "src/agent/prompts_util.py",
    }

    missing_files: List[str] = []
    for src_rel, dest_rel in files.items():
        destination = base_path / dest_rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        src_path = root_path / src_rel
        if src_path.exists():
            _atomic_copy(src_path, destination)
        else:
            missing_files.append(str(src_path))
            print(f"Template file missing, skipping: {src_path}")

    env_example = root_path / ".env.example"
    if env_example.exists():
        _atomic_copy(env_example, base_path / ".env.example")

    return missing_files


def write_config_files(state: Dict[str, Any]) -> None:
    base_path = _get_base_path(state)

    langgraph_path = base_path / "langgraph.json"
    langgraph_contents = json.dumps(
        {
            "graphs": {"agent": "src.agent.graph:graph"},
            "dependencies": ["."],
            "env": "./.env",
        },
        indent=2,
    )
    _atomic_write_text(langgraph_path, langgraph_contents)

    pyproject_path = base_path / "pyproject.toml"
    pyproject_contents = """[project]
name = \"generated-agent\"
version = \"0.1.0\"
requires-python = \">=3.11\"
dependencies = [
  \"langgraph>=0.6,<0.7\",
  \"langchain-core>=0.3,<0.4\",
  \"langchain-openai>=0.3,<0.4\",
  \"pydantic>=2.7,<3\",
  \"langgraph-checkpoint-sqlite>=2.0.0\",
  \"aiosqlite>=0.17.0\",
  \"pytest>=7.0.0\",
  \"langgraph-cli[inmem]>=0.1.0\",
  \"requests>=2.25.0\",
  \"black>=22.0.0\",
  \"isort>=5.0.0\",
  \"mypy>=1.0.0\",
  \"bandit[toml]>=1.7.0\",
]
[build-system]
requires = [\"setuptools\",\"wheel\"]
build-backend = \"setuptools.build_meta\"
[tool.setuptools.packages.find]
where = [\"src\"]
"""
    _atomic_write_text(pyproject_path, pyproject_contents)

