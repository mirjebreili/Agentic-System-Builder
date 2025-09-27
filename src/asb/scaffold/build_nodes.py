from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

from asb.utils.fileops import atomic_write, ensure_dir


ROOT = Path(__file__).resolve().parents[3]

logger = logging.getLogger(__name__)

SCAFFOLD_BASE_PATH_KEY = "_scaffold_base_path"
SCAFFOLD_ROOT_KEY = "_scaffold_root_path"


HEADLINE_NODE_SOURCE = """from __future__ import annotations

from typing import Any, Dict, List


def _extract_text(state: Dict[str, Any]) -> str:
    if not isinstance(state, dict):
        return ""
    candidates: List[str] = []
    for key in ("text", "input", "content", "prompt", "body"):
        value = state.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    candidates.append(item.strip())
    messages = state.get("messages")
    if isinstance(messages, list):
        for item in messages:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str) and content.strip():
                    candidates.append(content.strip())
    return " ".join(candidates).strip()


def _slice_headlines(text: str) -> List[str]:
    import math
    import re

    cleaned = text.strip()
    if not cleaned:
        return []
    sentences = [segment.strip() for segment in re.split(r"[.!?\\n]+", cleaned) if segment.strip()]
    segments: List[str] = []
    for sentence in sentences:
        segments.append(sentence[:120])
        if len(segments) == 5:
            break
    if len(segments) >= 3:
        return segments[:5]
    words = cleaned.split()
    if not words:
        return segments
    target = max(3, min(5, math.ceil(len(words) / 8) or 3))
    chunk_size = max(1, math.ceil(len(words) / target))
    segments = []
    for start in range(0, len(words), chunk_size):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            segments.append(chunk[:120])
        if len(segments) == 5:
            break
    while segments and len(segments) < 3:
        segments.append(segments[-1])
    return segments[:5]


def node_headline_generator(state: Dict[str, Any] | None = None) -> Dict[str, Any]:
    working: Dict[str, Any] = dict(state or {})
    headlines = _slice_headlines(_extract_text(working))
    if not headlines:
        headlines = ["No content available"] * 3
    working["headlines"] = headlines[:5]
    messages = working.get("messages")
    if not isinstance(messages, list):
        working["messages"] = []
    return working


__all__ = ["node_headline_generator"]
"""


def _ensure_headline_generator_module(agent_dir: Path, *, overwrite: bool = False) -> Path:
    ensure_dir(agent_dir)
    target = agent_dir / "headline_generator.py"
    if overwrite or not target.exists():
        atomic_write(target, HEADLINE_NODE_SOURCE)
    return target


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


def _update_build_report(
    state: Dict[str, Any],
    node: str,
    *,
    success: bool,
    errors: List[str] | None = None,
    details: Dict[str, Any] | None = None,
) -> None:
    scaffold = state.setdefault("scaffold", {})
    build_report = scaffold.setdefault("build_report", {})
    build_report[f"{node}_status"] = "complete" if success else "failed"
    if errors:
        build_report[f"{node}_errors"] = list(errors)
    else:
        build_report.pop(f"{node}_errors", None)
    if details:
        build_report[f"{node}_details"] = details


def _atomic_copy(src: Path, dest: Path) -> None:
    ensure_dir(dest.parent)
    tmp_path = dest.parent / f"{dest.name}.tmp"
    if tmp_path.exists():
        tmp_path.unlink()
    with src.open("rb") as source, tmp_path.open("wb") as target:
        shutil.copyfileobj(source, target)
    tmp_path.replace(dest)


def init_project_structure(state: Dict[str, Any]) -> Path:
    node_name = "init_project_structure"
    directories: List[Path] = []
    try:
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
            ensure_dir(directory)
    except Exception as exc:
        _update_build_report(state, node_name, success=False, errors=[str(exc)])
        raise

    _update_build_report(
        state,
        node_name,
        success=True,
        details={
            "created_directories": [str(path) for path in directories],
        },
    )
    return base_path


def copy_base_files(state: Dict[str, Any]) -> List[str]:
    node_name = "copy_base_files"
    try:
        base_path = _get_base_path(state)
        root_path = _get_root(state)
    except Exception as exc:
        _update_build_report(state, node_name, success=False, errors=[str(exc)])
        raise

    files = {
        "src/config/settings.py": "src/config/settings.py",
        "src/asb/llm/client.py": "src/llm/client.py",
        "src/asb/agent/prompts_util.py": "src/agent/prompts_util.py",
    }

    missing_files: List[str] = []
    copied: List[str] = []

    try:
        for src_rel, dest_rel in files.items():
            destination = base_path / dest_rel
            ensure_dir(destination.parent)
            src_path = root_path / src_rel
            if src_path.exists():
                _atomic_copy(src_path, destination)
                copied.append(dest_rel)
            else:
                missing_files.append(str(src_path))
                logger.warning("Template file missing, skipping: %s", src_path)

        env_example = root_path / ".env.example"
        if env_example.exists():
            ensure_dir((base_path / ".env.example").parent)
            _atomic_copy(env_example, base_path / ".env.example")
            copied.append(".env.example")
    except Exception as exc:
        _update_build_report(state, node_name, success=False, errors=[str(exc)])
        raise

    details = {"copied": copied, "missing": missing_files}
    errors = missing_files if missing_files else None
    _update_build_report(state, node_name, success=not errors, errors=errors, details=details)
    return missing_files


def write_config_files(state: Dict[str, Any]) -> None:
    node_name = "write_config_files"
    try:
        base_path = _get_base_path(state)
    except Exception as exc:
        _update_build_report(state, node_name, success=False, errors=[str(exc)])
        raise

    langgraph_path = base_path / "langgraph.json"
    langgraph_contents = json.dumps(
        {
            "graphs": {"agent": "src.agent.graph:graph"},
            "dependencies": ["."],
            "env": "./.env",
        },
        indent=2,
    )

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
  \"pytest>=7.0.0\",
  \"langgraph-cli[inmem]>=0.1.0\",
  \"requests>=2.25.0\",
  \"black>=22.0.0\",
  \"isort>=5.0.0\",
  \"mypy>=1.0.0\",
  \"bandit[toml]>=1.7.0\",
]
[project.optional-dependencies]
standalone = [
  \"langgraph-checkpoint-sqlite>=2.0.0\",
  \"aiosqlite>=0.17.0\",
]
[build-system]
requires = [\"setuptools\",\"wheel\"]
build-backend = \"setuptools.build_meta\"
[tool.setuptools.packages.find]
where = [\"src\"]
"""
    try:
        ensure_dir(langgraph_path.parent)
        atomic_write(langgraph_path, langgraph_contents)
        ensure_dir(pyproject_path.parent)
        atomic_write(pyproject_path, pyproject_contents)
    except Exception as exc:
        _update_build_report(state, node_name, success=False, errors=[str(exc)])
        raise

    _update_build_report(
        state,
        node_name,
        success=True,
        details={
            "langgraph_path": str(langgraph_path),
            "pyproject_path": str(pyproject_path),
        },
    )


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

    node_name = "write_state_schema"
    try:
        state_path = _get_base_path(state) / "src" / "agent" / "state.py"
        ensure_dir(state_path.parent)
        contents = _get_generated_content(
            state, "state.py", "src/agent/state.py", "agent/state.py"
        )
        if contents is None:
            plan = state.get("_scaffold_architecture_plan")
            contents = generate_enhanced_state_schema(plan if isinstance(plan, dict) else {})
        atomic_write(state_path, contents)
    except Exception as exc:
        _update_build_report(state, node_name, success=False, errors=[str(exc)])
        raise

    _update_build_report(
        state,
        node_name,
        success=True,
        details={"state_path": str(state_path)},
    )


def _build_plan_node_specs(architecture_plan: Dict[str, Any]) -> List[Tuple[str, str, List[str], Dict[str, Any]]]:
    from asb.agent import scaffold as scaffold_module

    specs: List[Tuple[str, str, List[str], Dict[str, Any]]] = []
    if not isinstance(architecture_plan, dict):
        return specs
    nodes = architecture_plan.get("nodes") or architecture_plan.get("graph_structure") or []
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
        ensure_dir(module_path.parent)
        atomic_write(module_path, source)
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

    node_name = "write_node_modules"
    try:
        base = _get_base_path(state)
        agent_dir = base / "src" / "agent"
        ensure_dir(agent_dir)

        architecture_plan = state.get("_scaffold_architecture_plan") or {}
        if not isinstance(architecture_plan, dict):
            architecture_plan = {}
        ensure_headline_fallback = False
        if not architecture_plan.get("graph_structure"):
            architecture_plan = dict(architecture_plan)
            architecture_plan["graph_structure"] = [
                {"name": "headline_generator", "responsibility": "Generate headlines"}
            ]
            state["_scaffold_architecture_plan"] = architecture_plan
            ensure_headline_fallback = True
        user_goal = state.get("_scaffold_user_goal") or ""
        requires_self_correction = False
        if isinstance(architecture_plan, dict):
            requires_self_correction = scaffold_module._architecture_requires_self_correction(
                architecture_plan
            )

        logger.debug("Build debug - architecture plan: %s", architecture_plan)
        logger.debug("Build debug - user goal: %s", user_goal)

        node_specs: List[Tuple[str, str, List[str], Dict[str, Any]]] = []
        generated_nodes: Dict[str, str] = {}
        normalized_generated = _get_normalized_generated_files(state)

        module_lookup: Dict[str, str] = {}

        if isinstance(architecture_plan, dict):
            node_specs = _build_plan_node_specs(architecture_plan)
            if node_specs:
                logger.debug("Build debug - found %d nodes to generate", len(node_specs))
                module_lookup = {module: node_id for node_id, module, _, _ in node_specs}
                if not requires_self_correction:
                    goal_lower = user_goal.lower()
                    if "summariz" in goal_lower or "chat" in goal_lower:
                        for node_id, module_name, _hints, metadata in node_specs:
                            template = generate_node_template(
                                node_id, module_name, user_goal, metadata
                            )
                            generated_nodes[f"{module_name}.py"] = template
                    else:
                        generated_nodes = scaffold_module.generate_nodes_from_architecture(
                            architecture_plan, user_goal
                        ) or {}
                    if generated_nodes:
                        for filename, source in generated_nodes.items():
                            module_name = Path(filename).stem
                            node_id = module_lookup.get(module_name, module_name)
                            normalized_tot = scaffold_module._normalize_tot_node_id(node_id)
                            if normalized_tot and normalized_tot in scaffold_module._TOT_RENDERERS:
                                continue
                            candidate = f"src/agent/{filename}"
                            normalized_key = scaffold_module._normalize_generated_key(
                                candidate
                            )
                            if normalized_key not in normalized_generated:
                                normalized_generated[normalized_key] = source

        if not node_specs:
            logger.info("No architecture plan found - using fallback generation")
            architecture_fallback = state.get("architecture") or {}
            node_specs = scaffold_module._collect_architecture_nodes(architecture_fallback)

        if node_specs:
            scaffold_module._write_node_modules(
                base,
                node_specs,
                normalized_generated,
                state.get("_scaffold_missing_files", []),
                user_goal,
                state.get("_scaffold_errors", []),
            )

        node_definitions = _build_node_definitions(agent_dir, node_specs)
        if ensure_headline_fallback:
            _ensure_headline_generator_module(agent_dir, overwrite=True)
            for definition in node_definitions:
                if definition.get("module") == "headline_generator":
                    definition["callable"] = "node_headline_generator"
    except Exception as exc:
        _update_build_report(state, node_name, success=False, errors=[str(exc)])
        raise

    _update_build_report(
        state,
        node_name,
        success=True,
        details={
            "node_count": len(node_definitions),
            "architecture_found": bool(architecture_plan),
            "user_goal": user_goal[:100] + "..." if len(user_goal) > 100 else user_goal,
        },
    )
    return node_definitions, node_specs


def generate_node_template(
    node_id: str,
    module_name: str,
    user_goal: str,
    metadata: Dict[str, Any] | None = None,
) -> str:
    """Generate a basic node template based on node ID and user goal."""

    func_name = module_name or node_id
    user_goal_str = str(user_goal or "")
    goal_lower = user_goal_str.lower()
    if "summariz" in goal_lower:
        return generate_summarizer_node_template(node_id, func_name, user_goal_str, metadata)
    if "chat" in goal_lower:
        return generate_chat_node_template(node_id, func_name, user_goal_str, metadata)
    return generate_generic_node_template(node_id, func_name, user_goal_str, metadata)


def generate_summarizer_node_template(
    node_id: str, func_name: str, user_goal: str, metadata: Dict[str, Any] | None
) -> str:
    """Generate a summarizer-specific node."""

    title = node_id.replace("_", " ").title()
    goal_for_doc = user_goal.replace("\"", "\\\"").replace("\n", " ")
    purpose = ""
    if metadata:
        raw_purpose = metadata.get("purpose") or metadata.get("description")
        if raw_purpose:
            purpose = str(raw_purpose).replace("\"", "\\\"").replace("\n", " ")

    lines = [
        f'"""Generated node: {node_id}"""',
        "",
        "from typing import Dict, Any",
        "from langchain_core.messages import AIMessage, HumanMessage, SystemMessage",
        "from ..llm import client",
        "from ..prompts_util import ROLE_RESPONSE_GUIDELINES, ROLE_SYSTEM_PROMPTS",
        "from asb.utils.message_utils import extract_last_message_content",
        "from .state import AppState",
        "",
        f"def {func_name}(state: AppState) -> AppState:",
        '    """',
        f"    {title} node for summarization workflow.",
        f"    Goal: {goal_for_doc}",
    ]
    if purpose:
        lines.append(f"    Purpose: {purpose}")
    lines.extend(
        [
            '    """',
            "",
            "    llm = client.get_chat_model()",
            "    base_prompt = ROLE_SYSTEM_PROMPTS.get(\"default\", \"\")",
            "    if base_prompt:",
            "        base_prompt = f\"{base_prompt}\\n\\n\"",
            "    response_guidelines = ROLE_RESPONSE_GUIDELINES.get(\"default\", \"\")",
            "    prompt_prefix = base_prompt",
            "    if response_guidelines:",
            "        prompt_prefix = (prompt_prefix + response_guidelines + \"\\n\\n\") if prompt_prefix else response_guidelines + \"\\n\\n\"",
            "",
            "",
            '    messages = state.get("messages", [])',
            '    user_input = extract_last_message_content(messages, "")',
            '    if not user_input:',
            '        user_input = state.get("input_text", "")',
            '    scratchpad = dict(state.get("scratch") or {})',
            "",
            f'    if "{func_name}" == "plan":',
            '        response = llm.invoke([',
            '            SystemMessage(prompt_prefix + "You are a summarization planner. Create a strategy for summarizing the given text."),',
            '            HumanMessage(f"Create a summary plan for: {{user_input}}"),',
            '        ])',
            '        plan_text = getattr(response, "content", response)',
            '        if not isinstance(plan_text, str):',
            '            plan_text = str(plan_text)',
            '        return {',
            '            "scratch": {"summary_plan": plan_text},',
            '            "messages": [AIMessage(content=f"[plan] {plan_text}")],',
            '        }',
            "",
            f'    if "{func_name}" == "do":',
            '        summary_plan = scratchpad.get("summary_plan", "Create 3-5 bullet points")',
            '        response = llm.invoke([',
            '            SystemMessage(prompt_prefix + "You are an expert summarizer. Follow the plan to create a clear, concise summary."),',
            '            HumanMessage(f"Plan: {{summary_plan}}\\n\\nText to summarize: {{user_input}}"),',
            '        ])',
            '        summary_text = getattr(response, "content", response)',
            '        if not isinstance(summary_text, str):',
            '            summary_text = str(summary_text)',
            '        return {',
            '            "result": summary_text,',
            '            "scratch": {"summary_result": summary_text},',
            '            "messages": [AIMessage(content=f"[summary] {summary_text}")],',
            '        }',
            "",
            f'    if "{func_name}" == "finish":',
            '        summary_result = scratchpad.get("summary_result", "")',
            '        formatted_output = f"## Summary\\n\\n{summary_result}"',
            '        return {',
            '            "final_output": formatted_output,',
            '            "result": summary_result,',
            '            "messages": [AIMessage(content=f"[finish] {formatted_output}")],',
            '            "scratch": {"completed": True},',
            '        }',
            "",
            '    response = llm.invoke([',
            '        SystemMessage(prompt_prefix + "You are a helpful assistant working on a summarization task."),',
            '        HumanMessage(f"Process this step: {{user_input}}"),',
            '    ])',
            '    raw_content = getattr(response, "content", response)',
            '    if not isinstance(raw_content, str):',
            '        raw_content = str(raw_content)',
            "    return {",
            f"        \"result\": raw_content,",
            f"        \"messages\": [AIMessage(content=f\"[{func_name}] {{raw_content}}\")],",
            f"        \"scratch\": {{\"{func_name}_result\": raw_content}},",
            "    }",
        ]
    )
    return "\n".join(lines)


def generate_chat_node_template(
    node_id: str,
    func_name: str,
    user_goal: str,
    metadata: Dict[str, Any] | None,
) -> str:
    """Generate a chat-specific node."""

    title = node_id.replace("_", " ").title()
    goal_for_doc = user_goal.replace("\"", "\\\"").replace("\n", " ")
    purpose = ""
    if metadata:
        raw_purpose = metadata.get("purpose") or metadata.get("description")
        if raw_purpose:
            purpose = str(raw_purpose).replace("\"", "\\\"").replace("\n", " ")

    lines = [
        f'"""Generated node: {node_id}"""',
        "",
        "from typing import Dict, Any",
        "from langchain_core.messages import AIMessage, HumanMessage, SystemMessage",
        "from ..llm import client",
        "from ..prompts_util import ROLE_RESPONSE_GUIDELINES, ROLE_SYSTEM_PROMPTS",
        "from asb.utils.message_utils import extract_last_message_content",
        "from .state import AppState",
        "",
        f"def {func_name}(state: AppState) -> AppState:",
        '    """',
        f"    {title} node for chat workflow.",
        f"    Goal: {goal_for_doc}",
    ]
    if purpose:
        lines.append(f"    Purpose: {purpose}")
    lines.extend(
        [
            '    """',
            "",
            "    llm = client.get_chat_model()",
            "    base_prompt = ROLE_SYSTEM_PROMPTS.get(\"default\", \"\")",
            "    if base_prompt:",
            "        base_prompt = f\"{base_prompt}\\n\\n\"",
            "    response_guidelines = ROLE_RESPONSE_GUIDELINES.get(\"default\", \"\")",
            "    prompt_prefix = base_prompt",
            "    if response_guidelines:",
            "        prompt_prefix = (prompt_prefix + response_guidelines + \"\\n\\n\") if prompt_prefix else response_guidelines + \"\\n\\n\"",
            "",
            '    messages = state.get("messages", [])',
            '    user_input = extract_last_message_content(messages, "")',
            '    if not user_input:',
            '        user_input = state.get("input_text", "")',
            "",
            '    response = llm.invoke([',
            '        SystemMessage(prompt_prefix + "You are a collaborative AI assistant in a chat application."),',
            f'        HumanMessage(f"Handle the \'{node_id}\' step with input: {{user_input}}"),',
            '    ])',
            '    raw_content = getattr(response, "content", response)',
            '    if not isinstance(raw_content, str):',
            '        raw_content = str(raw_content)',
            "",
            "    return {",
            f"        \"result\": raw_content,",
            f"        \"messages\": [AIMessage(content=f\"[{func_name}] {{raw_content}}\")],",
            f"        \"scratch\": {{\"{func_name}_result\": raw_content}},",
            "    }",
        ]
    )
    return "\n".join(lines)


def generate_generic_node_template(
    node_id: str,
    func_name: str,
    user_goal: str,
    metadata: Dict[str, Any] | None,
) -> str:
    """Generate a generic node template."""

    title = node_id.replace("_", " ").title()
    goal_for_doc = user_goal.replace("\"", "\\\"").replace("\n", " ")
    purpose = ""
    if metadata:
        raw_purpose = metadata.get("purpose") or metadata.get("description")
        if raw_purpose:
            purpose = str(raw_purpose).replace("\"", "\\\"").replace("\n", " ")
    purpose_for_prompt = purpose or title.lower()

    lines = [
        f'"""Generated node: {node_id}"""',
        "",
        "from typing import Dict, Any",
        "from langchain_core.messages import AIMessage, HumanMessage, SystemMessage",
        "from ..llm import client",
        "from ..prompts_util import ROLE_RESPONSE_GUIDELINES, ROLE_SYSTEM_PROMPTS",
        "from asb.utils.message_utils import extract_last_message_content",
        "from .state import AppState",
        "",
        f"def {func_name}(state: AppState) -> AppState:",
        '    """',
        f"    {title} node for workflow.",
        f"    Goal: {goal_for_doc}",
    ]
    if purpose:
        lines.append(f"    Purpose: {purpose}")
    lines.extend(
        [
            '    """',
            "",
            "    llm = client.get_chat_model()",
            "    base_prompt = ROLE_SYSTEM_PROMPTS.get(\"default\", \"\")",
            "    if base_prompt:",
            "        base_prompt = f\"{base_prompt}\\n\\n\"",
            "    response_guidelines = ROLE_RESPONSE_GUIDELINES.get(\"default\", \"\")",
            "    prompt_prefix = base_prompt",
            "    if response_guidelines:",
            "        prompt_prefix = (prompt_prefix + response_guidelines + \"\\n\\n\") if prompt_prefix else response_guidelines + \"\\n\\n\"",
            f"    adaptive_prompt = \"Adaptive implementation for the {purpose_for_prompt} step.\"",
            "",
            '    messages = state.get("messages", [])',
            '    user_input = extract_last_message_content(messages, "")',
            '    if not user_input:',
            '        user_input = state.get("input_text", "")',
            "",
            '    response = llm.invoke([',
            f'        SystemMessage(prompt_prefix + adaptive_prompt),',
            '        HumanMessage(f"Input: {{user_input}}"),',
            '    ])',
            '    raw_content = getattr(response, "content", response)',
            '    if not isinstance(raw_content, str):',
            '        raw_content = str(raw_content)',
            "",
            "    return {",
            f"        \"result\": raw_content,",
            f"        \"messages\": [AIMessage(content=f\"[{func_name}] {{raw_content}}\")],",
            f"        \"scratch\": {{\"{func_name}_result\": raw_content}},",
            "    }",
        ]
    )
    return "\n".join(lines)

def write_graph_module(state: Dict[str, Any]) -> None:
    from asb.agent import scaffold as scaffold_module

    node_name = "write_graph_module"
    try:
        agent_dir = _get_base_path(state) / "src" / "agent"
        ensure_dir(agent_dir)
        executor_source = _get_generated_content(
            state,
            "executor.py",
            "src/agent/executor.py",
            "agent/executor.py",
        )
        node_definitions = state.get("_scaffold_node_definitions") or []
        if executor_source is None:
            executor_source = scaffold_module._render_executor_module(node_definitions)
        executor_path = agent_dir / "executor.py"
        ensure_dir(executor_path.parent)
        atomic_write(executor_path, executor_source)
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
        graph_path = agent_dir / "graph.py"
        ensure_dir(graph_path.parent)
        atomic_write(graph_path, graph_source)
    except Exception as exc:
        _update_build_report(state, node_name, success=False, errors=[str(exc)])
        raise

    _update_build_report(
        state,
        node_name,
        success=True,
        details={
            "executor_path": str(agent_dir / "executor.py"),
            "graph_path": str(agent_dir / "graph.py"),
        },
    )

