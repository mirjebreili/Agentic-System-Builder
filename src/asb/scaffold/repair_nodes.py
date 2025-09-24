"""Heuristics for repairing scaffolded LangGraph node modules."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Tuple

from asb.agent import scaffold as scaffold_module
from asb.agent.scaffold import generate_generic_node_template
from asb.utils.fileops import atomic_write, ensure_dir

from .build_nodes import _build_node_definitions, _build_plan_node_specs


_EXCLUDED_NODE_FILES = {
    "__init__.py",
    "executor.py",
    "graph.py",
    "planner.py",
    "prompts_util.py",
    "state.py",
}


def _resolve_project_path(state: Mapping[str, Any]) -> Path:
    scaffold = state.get("scaffold") if isinstance(state, Mapping) else None
    if not isinstance(scaffold, Mapping):
        raise ValueError("Scaffold metadata is missing from the state.")
    project_path = scaffold.get("path")
    if not project_path:
        raise ValueError("Scaffold project path is not available.")
    return Path(str(project_path))


def _sanitize_identifier(value: str) -> str:
    return re.sub(r"\W+", "_", value).strip("_") or "node"


def _iter_architecture_nodes(state: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
    architecture = state.get("architecture") if isinstance(state, Mapping) else None
    if not isinstance(architecture, Mapping):
        return []

    def _maybe_extend(container: Any) -> Iterable[Dict[str, Any]]:
        if isinstance(container, Sequence):
            for item in container:
                if isinstance(item, Mapping):
                    yield dict(item)
        return []

    nodes: List[Dict[str, Any]] = []

    plan = architecture.get("plan")
    if isinstance(plan, Mapping):
        nodes.extend(list(_maybe_extend(plan.get("nodes"))))
        nodes.extend(list(_maybe_extend(plan.get("graph_structure"))))

    nodes.extend(list(_maybe_extend(architecture.get("nodes"))))
    nodes.extend(list(_maybe_extend(architecture.get("graph_structure"))))
    return nodes


def _match_node_metadata(module_name: str, state: Mapping[str, Any]) -> Tuple[str, str]:
    fallback_name = module_name
    fallback_purpose = f"Carry out the {module_name} step."

    for entry in _iter_architecture_nodes(state):
        identifier = ""
        for key in ("id", "name", "node", "label"):
            value = entry.get(key)
            if value is None:
                continue
            candidate = str(value).strip()
            if not candidate:
                continue
            sanitized = _sanitize_identifier(candidate)
            if sanitized == module_name or candidate == module_name:
                identifier = candidate
                break
        if not identifier:
            continue

        description = ""
        for key in ("description", "purpose", "summary", "prompt"):
            value = entry.get(key)
            if value:
                description = str(value).strip()
                if description:
                    break
        return identifier, description or fallback_purpose

    return fallback_name, fallback_purpose


def _infer_user_goal(state: Mapping[str, Any]) -> str:
    candidates: List[Any] = []
    if isinstance(state, Mapping):
        plan = state.get("plan")
        if isinstance(plan, Mapping):
            candidates.append(plan.get("goal"))
        architecture = state.get("architecture")
        if isinstance(architecture, Mapping):
            candidates.append(architecture.get("goal"))
            plan = architecture.get("plan")
            if isinstance(plan, Mapping):
                candidates.append(plan.get("goal"))
        candidates.append(state.get("goal"))
        messages = state.get("messages")
        if isinstance(messages, Sequence):
            for message in reversed(messages):
                if isinstance(message, Mapping):
                    content = message.get("content")
                else:
                    content = getattr(message, "content", None)
                if content:
                    candidates.append(content)
                    break
    for candidate in candidates:
        if candidate:
            return str(candidate)
    return "Complete the requested task."


def _collect_scaffold_errors(state: Mapping[str, Any]) -> List[str]:
    scaffold = state.get("scaffold") if isinstance(state, Mapping) else None
    if isinstance(scaffold, Mapping):
        errors = scaffold.get("errors")
        if isinstance(errors, list):
            return errors
    return []


def _update_build_report(
    state: MutableMapping[str, Any],
    node: str,
    *,
    success: bool,
    errors: Iterable[str] | None = None,
    details: Mapping[str, Any] | None = None,
) -> None:
    scaffold = state.setdefault("scaffold", {})
    if not isinstance(scaffold, MutableMapping):  # pragma: no cover - defensive
        scaffold = {}
        state["scaffold"] = scaffold
    build_report = scaffold.setdefault("build_report", {})
    build_report[f"{node}_status"] = "complete" if success else "failed"
    if errors:
        build_report[f"{node}_errors"] = list(errors)
    else:
        build_report.pop(f"{node}_errors", None)
    if details is not None:
        build_report[f"{node}_details"] = dict(details)


def _record_repair_result(
    state: MutableMapping[str, Any],
    name: str,
    *,
    success: bool,
    fixed: Iterable[str],
    errors: Iterable[str],
    details: Mapping[str, Any] | None = None,
) -> None:
    scaffold = state.setdefault("scaffold", {})
    if not isinstance(scaffold, MutableMapping):  # pragma: no cover - defensive
        scaffold = {}
        state["scaffold"] = scaffold
    repairs = scaffold.setdefault("repairs", {})
    repairs[name] = {
        "success": bool(success),
        "fixed": list(fixed),
        "errors": list(errors),
        "details": dict(details) if details is not None else {},
    }


def _remove_scaffold_messages(state: MutableMapping[str, Any], messages: Iterable[str]) -> None:
    scaffold = state.get("scaffold")
    if not isinstance(scaffold, MutableMapping):
        return
    existing = scaffold.get("errors")
    if not isinstance(existing, list):
        return
    to_remove = set(messages)
    filtered = [msg for msg in existing if msg not in to_remove]
    if filtered:
        scaffold["errors"] = filtered
    else:
        scaffold.pop("errors", None)
    scaffold["ok"] = not bool(scaffold.get("errors"))


def _parse_error_entries(
    state: Mapping[str, Any],
    predicate,
) -> List[Tuple[str, str, str]]:
    matches: List[Tuple[str, str, str]] = []
    for message in _collect_scaffold_errors(state):
        if not isinstance(message, str):
            continue
        module, sep, detail = message.partition(":")
        module = module.strip()
        detail_text = detail.strip() if sep else ""
        if predicate(module, detail_text, message):
            matches.append((module, detail_text, message))
    return matches


def _render_repaired_node(module_name: str, state: Mapping[str, Any]) -> str:
    node_label, node_purpose = _match_node_metadata(module_name, state)
    user_goal = _infer_user_goal(state)
    return generate_generic_node_template(node_label, node_purpose, user_goal)


def fix_empty_nodes(state: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Rewrite empty node modules with a generic fallback implementation."""

    node_name = "fix_empty_nodes"
    matches = _parse_error_entries(
        state, lambda module, detail, _msg: detail.startswith("node file is empty")
    )
    if not matches:
        details = {"reason": "no empty node errors detected"}
        _record_repair_result(
            state,
            "empty_nodes",
            success=True,
            fixed=[],
            errors=[],
            details=details,
        )
        _update_build_report(state, node_name, success=True, details={"fixed": [], "failures": [], **details})
        return state

    fixed: List[str] = []
    failures: List[str] = []

    try:
        project_path = _resolve_project_path(state)
    except ValueError as exc:
        error_message = str(exc)
        _record_repair_result(state, "empty_nodes", success=False, fixed=[], errors=[error_message])
        _update_build_report(state, node_name, success=False, errors=[error_message])
        return state

    agent_dir = project_path / "src" / "agent"

    for module_name, _detail, message in matches:
        target_module = module_name or ""
        module_path = agent_dir / f"{target_module}.py"
        if not module_path.exists():
            failures.append(f"{target_module}: module file not found at {module_path}")
            continue
        try:
            contents = module_path.read_text(encoding="utf-8")
        except Exception as exc:  # pragma: no cover - filesystem errors are rare
            failures.append(f"{target_module}: unable to read module contents ({exc})")
            continue
        if contents.strip():
            fixed.append(target_module)
            continue
        try:
            replacement = _render_repaired_node(target_module, state)
        except Exception as exc:  # pragma: no cover - defensive
            failures.append(f"{target_module}: failed to generate fallback implementation ({exc})")
            continue
        try:
            ensure_dir(module_path.parent)
            atomic_write(module_path, replacement)
        except Exception as exc:  # pragma: no cover - filesystem errors are rare
            failures.append(f"{target_module}: failed to write repaired source ({exc})")
            continue
        fixed.append(target_module)

    if fixed:
        resolved = [
            message
            for module, _detail, message in matches
            if (module or "") in fixed or f"{module}.py" in fixed
        ]
        if resolved:
            _remove_scaffold_messages(state, resolved)

    success = not failures
    _record_repair_result(state, "empty_nodes", success=success, fixed=fixed, errors=failures)
    _update_build_report(
        state,
        node_name,
        success=success,
        errors=failures,
        details={"fixed": fixed, "failures": failures},
    )
    return state


def _determine_missing_imports(detail: str) -> List[str]:
    missing_prefix = "missing required imports:"
    if missing_prefix not in detail:
        return []
    remainder = detail.split(missing_prefix, 1)[1]
    imports = [segment.strip() for segment in remainder.split(",")]
    return [imp for imp in imports if imp]


def _insert_missing_imports(source: str, imports: Sequence[str]) -> str:
    if not imports:
        return source

    lines = source.splitlines()
    insert_at = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if idx == 0 and stripped.startswith("#!/"):
            insert_at = 1
            continue
        if stripped.startswith("from __future__ import"):
            insert_at = idx + 1
            continue
        if not stripped:
            insert_at = idx + 1
            continue
        break

    new_lines = lines[:insert_at] + list(imports)
    if insert_at < len(lines) and lines[insert_at].strip():
        new_lines.append("")
    new_lines.extend(lines[insert_at:])
    result = "\n".join(new_lines)
    if source.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result


def fix_import_errors(state: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Inject missing AppState/client imports into node modules."""

    node_name = "fix_import_errors"
    matches = _parse_error_entries(
        state, lambda _module, detail, _msg: "missing required imports" in detail
    )
    if not matches:
        details = {"reason": "no import errors detected"}
        _record_repair_result(
            state,
            "import_errors",
            success=True,
            fixed=[],
            errors=[],
            details=details,
        )
        _update_build_report(
            state,
            node_name,
            success=True,
            details={"fixed": [], "failures": [], **details},
        )
        return state

    fixed: List[str] = []
    failures: List[str] = []

    try:
        project_path = _resolve_project_path(state)
    except ValueError as exc:
        error_message = str(exc)
        _record_repair_result(state, "import_errors", success=False, fixed=[], errors=[error_message])
        _update_build_report(state, node_name, success=False, errors=[error_message])
        return state

    agent_dir = project_path / "src" / "agent"

    for module_name, detail, _message in matches:
        target_module = module_name or ""
        module_path = agent_dir / f"{target_module}.py"
        if not module_path.exists():
            failures.append(f"{target_module}: module file not found at {module_path}")
            continue
        missing_imports = _determine_missing_imports(detail)
        if not missing_imports:
            fixed.append(target_module)
            continue
        try:
            contents = module_path.read_text(encoding="utf-8")
        except Exception as exc:  # pragma: no cover - filesystem errors are rare
            failures.append(f"{target_module}: unable to read module contents ({exc})")
            continue
        additions = [imp for imp in missing_imports if imp not in contents]
        if not additions:
            fixed.append(target_module)
            continue
        updated = _insert_missing_imports(contents, additions)
        try:
            ensure_dir(module_path.parent)
            atomic_write(module_path, updated)
        except Exception as exc:  # pragma: no cover - filesystem errors are rare
            failures.append(f"{target_module}: failed to write repaired source ({exc})")
            continue
        fixed.append(target_module)

    if fixed:
        resolved = [
            message
            for module, _detail, message in matches
            if (module or "") in fixed or f"{module}.py" in fixed
        ]
        if resolved:
            _remove_scaffold_messages(state, resolved)

    success = not failures
    _record_repair_result(state, "import_errors", success=success, fixed=fixed, errors=failures)
    _update_build_report(
        state,
        node_name,
        success=success,
        errors=failures,
        details={"fixed": fixed, "failures": failures},
    )
    return state


def _fallback_node_specs(agent_dir: Path) -> List[Tuple[str, str, List[str], Dict[str, Any]]]:
    specs: List[Tuple[str, str, List[str], Dict[str, Any]]] = []
    for path in sorted(agent_dir.glob("*.py")):
        if path.name in _EXCLUDED_NODE_FILES:
            continue
        module_name = path.stem
        hints = scaffold_module._candidate_call_hints(module_name, module_name)
        specs.append((module_name, module_name, hints, {"id": module_name}))
    return specs


def _coalesce_architecture_plan(state: Mapping[str, Any]) -> Dict[str, Any]:
    architecture = state.get("architecture") if isinstance(state, Mapping) else None
    if not isinstance(architecture, Mapping):
        return {}
    plan = architecture.get("plan")
    if isinstance(plan, Mapping):
        return dict(plan)
    return dict(architecture)


def fix_graph_compilation(state: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Re-render the generated graph module when compilation fails."""

    def _is_graph_failure(_module: str, detail: str, message: str) -> bool:
        target = detail or message
        lowered = target.lower()
        return "generate_dynamic_workflow failed" in lowered or "unable to import generated graph module" in lowered

    node_name = "fix_graph_compilation"
    matches = _parse_error_entries(state, _is_graph_failure)
    if not matches:
        details = {"reason": "no graph compilation errors detected"}
        _record_repair_result(
            state,
            "graph_compilation",
            success=True,
            fixed=[],
            errors=[],
            details=details,
        )
        _update_build_report(
            state,
            node_name,
            success=True,
            details={"fixed": [], "failures": [], **details},
        )
        return state

    fixed: List[str] = []
    failures: List[str] = []

    try:
        project_path = _resolve_project_path(state)
    except ValueError as exc:
        error_message = str(exc)
        _record_repair_result(state, "graph_compilation", success=False, fixed=[], errors=[error_message])
        _update_build_report(state, node_name, success=False, errors=[error_message])
        return state

    agent_dir = project_path / "src" / "agent"
    graph_path = agent_dir / "graph.py"
    if not graph_path.exists():
        failures.append(f"graph: module file not found at {graph_path}")
        _record_repair_result(state, "graph_compilation", success=False, fixed=[], errors=failures)
        _update_build_report(state, node_name, success=False, errors=failures, details={"fixed": [], "failures": failures})
        return state

    architecture_plan = _coalesce_architecture_plan(state)

    try:
        node_specs = _build_plan_node_specs(architecture_plan)
    except Exception:  # pragma: no cover - defensive import failure
        node_specs = []

    if not node_specs:
        architecture = state.get("architecture") if isinstance(state, Mapping) else {}
        if isinstance(architecture, Mapping):
            try:
                node_specs = scaffold_module._collect_architecture_nodes(architecture)
            except Exception:  # pragma: no cover - defensive
                node_specs = []
    if not node_specs:
        node_specs = _fallback_node_specs(agent_dir)

    try:
        node_definitions = _build_node_definitions(agent_dir, node_specs)
    except Exception as exc:
        failures.append(f"graph: unable to determine node definitions ({exc})")
        _record_repair_result(state, "graph_compilation", success=False, fixed=[], errors=failures)
        _update_build_report(state, node_name, success=False, errors=failures, details={"fixed": [], "failures": failures})
        return state

    graph_plan = dict(architecture_plan)
    graph_plan["_node_definitions"] = node_definitions

    try:
        new_graph_source = scaffold_module.generate_dynamic_workflow_module(graph_plan)
    except Exception as exc:
        failures.append(f"graph: failed to render graph module ({exc})")
        _record_repair_result(state, "graph_compilation", success=False, fixed=[], errors=failures)
        _update_build_report(state, node_name, success=False, errors=failures, details={"fixed": [], "failures": failures})
        return state

    try:
        ensure_dir(graph_path.parent)
        atomic_write(graph_path, new_graph_source)
    except Exception as exc:  # pragma: no cover - filesystem errors are rare
        failures.append(f"graph: failed to write repaired graph module ({exc})")
        _record_repair_result(state, "graph_compilation", success=False, fixed=[], errors=failures)
        _update_build_report(state, node_name, success=False, errors=failures, details={"fixed": [], "failures": failures})
        return state

    fixed.append("graph.py")
    _remove_scaffold_messages(state, [match[2] for match in matches])
    _record_repair_result(state, "graph_compilation", success=True, fixed=fixed, errors=[])
    _update_build_report(
        state,
        node_name,
        success=True,
        details={"fixed": fixed, "failures": []},
    )
    return state
