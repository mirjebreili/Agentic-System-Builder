from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple


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


def _normalize_generated_key(value: str) -> str:
    normalized = value.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _get_normalized_generated_files(state: Dict[str, Any]) -> Dict[str, str]:
    cached = state.get("_scaffold_generated")
    if isinstance(cached, dict):
        return cached
    raw = state.get("generated_files") or {}
    normalized = {_normalize_generated_key(key): value for key, value in raw.items()}
    state["_scaffold_generated"] = normalized
    return normalized


def _get_generated_content(state: Dict[str, Any], *candidates: str) -> str | None:
    generated = _get_normalized_generated_files(state)
    for candidate in candidates:
        normalized = _normalize_generated_key(candidate)
        if normalized in generated:
            return generated[normalized]
    return None


def write_state_schema(state: Dict[str, Any]) -> None:
    from asb.agent.scaffold import generate_enhanced_state_schema

    state_path = _get_base_path(state) / "src" / "agent" / "state.py"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    contents = _get_generated_content(state, "state.py", "src/agent/state.py", "agent/state.py")
    if contents is None:
        plan = state.get("_scaffold_architecture_plan")
        contents = generate_enhanced_state_schema(plan if isinstance(plan, dict) else {})
    _atomic_write_text(state_path, contents)


def _build_plan_node_specs(architecture_plan: Dict[str, Any]) -> List[Tuple[str, str, List[str], Dict[str, Any]]]:
    from asb.agent import scaffold as scaffold_module

    specs: List[Tuple[str, str, List[str], Dict[str, Any]]] = []
    if not isinstance(architecture_plan, dict):
        return specs
    nodes = architecture_plan.get("nodes") or []
    seen: set[str] = set()
    for entry in nodes:
        if not isinstance(entry, dict):
            continue
        raw_name = entry.get("name") or entry.get("id") or entry.get("label")
        if raw_name is None:
            continue
        node_id = str(raw_name).strip()
        if not node_id or node_id in seen:
            continue
        module_name = scaffold_module._sanitize_identifier(node_id)
        hints = scaffold_module._candidate_call_hints(node_id, module_name)
        metadata = dict(entry)
        metadata.setdefault("id", node_id)
        specs.append((node_id, module_name, hints, metadata))
        seen.add(node_id)
    return specs


def _write_generated_nodes(
    agent_dir: Path,
    generated_nodes: Dict[str, str],
    node_specs: List[Tuple[str, str, List[str], Dict[str, Any]]],
    state: Dict[str, Any],
) -> None:
    from asb.agent import scaffold as scaffold_module

    module_lookup = {module: node_id for node_id, module, _, _ in node_specs}
    user_goal = state.get("_scaffold_user_goal") or ""
    errors = state.get("_scaffold_errors", [])
    for filename, source in generated_nodes.items():
        module_path = agent_dir / filename
        _atomic_write_text(module_path, source)
        module_name = module_path.stem
        node_id = module_lookup.get(module_name, module_name)
        scaffold_module._validate_node_module(
            module_path,
            node_id,
            module_name,
            user_goal,
            errors,
            allow_regenerate=True,
        )


def _write_existing_node_modules(
    base: Path,
    node_specs: List[Tuple[str, str, List[str], Dict[str, Any]]],
    state: Dict[str, Any],
    user_goal: str,
) -> None:
    from asb.agent import scaffold as scaffold_module

    scaffold_module._write_node_modules(
        base,
        node_specs,
        _get_normalized_generated_files(state),
        state.get("_scaffold_missing_files", []),
        user_goal,
        state.get("_scaffold_errors", []),
    )


def _build_node_definitions(
    agent_dir: Path,
    node_specs: List[Tuple[str, str, List[str], Dict[str, Any]]],
) -> List[Dict[str, str]]:
    from asb.agent import scaffold as scaffold_module

    definitions: List[Dict[str, str]] = []
    for node_id, module_name, hints, _ in node_specs:
        module_path = agent_dir / f"{module_name}.py"
        callable_name = scaffold_module._detect_node_callable(module_path, hints)
        definitions.append(
            {
                "id": node_id,
                "module": module_name,
                "callable": callable_name,
                "alias": module_name,
            }
        )
    return definitions


def write_node_modules(
    state: Dict[str, Any]
) -> Tuple[List[Dict[str, str]], List[Tuple[str, str, List[str], Dict[str, Any]]]]:
    from asb.agent import scaffold as scaffold_module

    base = _get_base_path(state)
    agent_dir = base / "src" / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    architecture_plan = state.get("_scaffold_architecture_plan") or {}
    user_goal = state.get("_scaffold_user_goal") or ""
    generated_nodes = scaffold_module.generate_nodes_from_architecture(architecture_plan, user_goal)
    node_specs = (
        _build_plan_node_specs(architecture_plan)
        if generated_nodes
        else scaffold_module._collect_architecture_nodes(state.get("architecture") or {})
    )
    if generated_nodes:
        _write_generated_nodes(agent_dir, generated_nodes, node_specs, state)
    elif node_specs:
        _write_existing_node_modules(base, node_specs, state, user_goal)
    node_definitions = _build_node_definitions(agent_dir, node_specs)
    return node_definitions, node_specs


def write_graph_module(state: Dict[str, Any]) -> None:
    from asb.agent import scaffold as scaffold_module

    agent_dir = _get_base_path(state) / "src" / "agent"
    executor_source = _get_generated_content(
        state,
        "executor.py",
        "src/agent/executor.py",
        "agent/executor.py",
    )
    node_definitions = state.get("_scaffold_node_definitions") or []
    if executor_source is None:
        executor_source = scaffold_module._render_executor_module(node_definitions)
    _atomic_write_text(agent_dir / "executor.py", executor_source)
    graph_source = _get_generated_content(
        state,
        "graph.py",
        "src/agent/graph.py",
        "agent/graph.py",
    )
    plan = state.get("_scaffold_architecture_plan")
    graph_plan = dict(plan) if isinstance(plan, dict) else {}
    graph_plan["_node_definitions"] = node_definitions
    if graph_source is None:
        graph_source = scaffold_module.generate_dynamic_workflow_module(graph_plan)
    _atomic_write_text(agent_dir / "graph.py", graph_source)

