"""Validation helpers for scaffolded LangGraph node modules."""

from __future__ import annotations

import ast
import importlib
import re
import sys
import time
from contextlib import suppress

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def validate_state_schema_safety(project_path: Path) -> List[str]:
    """Ensure generated state schema includes required aggregators."""

    issues: List[str] = []
    state_file = project_path / "src" / "agent" / "state.py"

    if not state_file.exists():
        issues.append("state.py missing")
        return issues

    try:
        state_content = state_file.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - filesystem errors are unexpected
        issues.append(f"Unable to read state.py: {exc}")
        return issues

    required_imports = [
        "from typing import Any, Dict, List, TypedDict, Annotated",
        "import operator",
        "from langgraph.graph import add_messages",
    ]

    for statement in required_imports:
        if statement not in state_content:
            issues.append(f"Missing required import in state.py: {statement}")

    if "Annotated[List[AnyMessage], add_messages]" not in state_content:
        issues.append("messages field missing add_messages aggregator")

    if "Annotated[Dict[str, Any], operator.or_]" not in state_content:
        issues.append("Dict fields missing operator.or_ aggregator")

    return issues


def _update_build_report(
    state: ScaffoldState,
    node: str,
    *,
    success: bool,
    errors: Iterable[str] | None = None,
    details: Mapping[str, Any] | None = None,
) -> None:
    scaffold = state.setdefault("scaffold", {})
    build_report = scaffold.setdefault("build_report", {})
    build_report[f"{node}_status"] = "complete" if success else "failed"
    if errors:
        build_report[f"{node}_errors"] = list(errors)
    else:
        build_report.pop(f"{node}_errors", None)
    if details is not None:
        build_report[f"{node}_details"] = dict(details)

ScaffoldState = Dict[str, Any]

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
    candidate: Any = None
    if isinstance(scaffold, Mapping):
        candidate = scaffold.get("path")
    if candidate is None:
        candidate = state.get("_scaffold_base_path") if isinstance(state, Mapping) else None
    if candidate is None:
        raise ValueError("Scaffold project path is not available in the state.")
    return Path(str(candidate))


def _collect_node_module_paths(project_path: Path) -> Dict[str, Path]:
    agent_dir = project_path / "src" / "agent"
    modules: Dict[str, Path] = {}
    if not agent_dir.exists():
        return modules

    executor_path = agent_dir / "executor.py"
    module_names: List[str] = []
    if executor_path.exists():
        try:
            tree = ast.parse(executor_path.read_text(encoding="utf-8"), filename=str(executor_path))
        except Exception:
            module_names = []
        else:
            seen: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                    name = node.module.strip()
                    if name and name not in seen:
                        module_names.append(name)
                        seen.add(name)

    for name in module_names:
        modules[name] = agent_dir / f"{name}.py"

    if modules:
        return modules

    for path in sorted(agent_dir.glob("*.py")):
        if path.name in _EXCLUDED_NODE_FILES:
            continue
        try:
            contents = path.read_text(encoding="utf-8")
        except Exception:
            continue
        normalized = contents.lower()
        if "appstate" in contents or "client.get_chat_model" in normalized or "role_system_prompts" in normalized:
            modules[path.stem] = path
    return modules


def _start_phase(state: ScaffoldState, name: str, description: str) -> tuple[Dict[str, Any], float]:
    started = time.time()
    phase: Dict[str, Any] = {
        "name": name,
        "description": description,
        "status": "in_progress",
        "started_at": started,
    }
    state["scaffold_phase"] = phase
    return phase, started


def _finish_phase(
    state: ScaffoldState,
    phase: Dict[str, Any],
    started: float,
    *,
    success: bool,
    summary: str,
    details: Dict[str, Any] | None = None,
    errors: Iterable[str] | None = None,
) -> None:
    completed = time.time()
    phase["completed_at"] = completed
    phase["duration"] = max(0.0, completed - started)
    phase["status"] = "complete" if success else "failed"
    if summary:
        phase["summary"] = summary
    if details is not None:
        phase["details"] = dict(details)
    error_messages = list(errors or [])
    if error_messages:
        phase["error"] = error_messages[0]
        if details is not None:
            phase.setdefault("details", {}).setdefault("errors", error_messages)
    state["scaffold_phase"] = phase


def _record_validation_result(
    state: ScaffoldState,
    check_name: str,
    success: bool,
    errors: Iterable[str],
    details: Mapping[str, Any],
) -> None:
    scaffold = state.setdefault("scaffold", {})
    validation = scaffold.setdefault("validation", {})
    validation[check_name] = {
        "ok": bool(success),
        "errors": list(errors),
        "details": dict(details),
    }
    if success:
        if "ok" not in scaffold:
            scaffold["ok"] = True
    else:
        scaffold.setdefault("errors", [])
        for message in errors:
            if message and message not in scaffold["errors"]:
                scaffold["errors"].append(message)
        scaffold["ok"] = False


def validate_non_empty_generation(state: ScaffoldState) -> ScaffoldState:
    """Ensure the scaffold produced non-empty executor and graph modules."""

    errors: List[str] = []
    details: Dict[str, Any] = {}

    try:
        project_path = _resolve_project_path(state)
    except Exception as exc:
        message = f"Unable to resolve scaffold project path: {exc}"
        errors.append(message)
        _record_validation_result(state, "non_empty_generation", False, errors, details)
        _update_build_report(
            state,
            "validate_non_empty_generation",
            success=False,
            errors=errors,
            details=details,
        )
        return state

    project_details: Dict[str, Any] = {}

    executor_path = project_path / "src" / "agent" / "executor.py"
    if executor_path.exists():
        project_details["executor_exists"] = True
        try:
            executor_contents = executor_path.read_text(encoding="utf-8")
        except Exception as exc:
            errors.append(f"Unable to read executor.py: {exc}")
        else:
            project_details["executor_checked"] = True
            if "NODE_IMPLEMENTATIONS: List[Tuple[str, Callable]] = []" in executor_contents:
                errors.append("executor.py has empty NODE_IMPLEMENTATIONS")
    else:
        project_details["executor_exists"] = False

    graph_path = project_path / "src" / "agent" / "graph.py"
    if graph_path.exists():
        project_details["graph_exists"] = True
        try:
            graph_contents = graph_path.read_text(encoding="utf-8")
        except Exception as exc:
            errors.append(f"Unable to read graph.py: {exc}")
        else:
            project_details["graph_checked"] = True
            if "ARCHITECTURE_STATE = {}" in graph_contents or "json.loads('{}'" in graph_contents:
                errors.append("graph.py has empty ARCHITECTURE_STATE")
    else:
        project_details["graph_exists"] = False

    details.update(project_details)
    success = not errors
    _record_validation_result(state, "non_empty_generation", success, errors, details)
    _update_build_report(
        state,
        "validate_non_empty_generation",
        success=success,
        errors=errors,
        details=details,
    )
    return state


def validate_syntax(state: ScaffoldState) -> ScaffoldState:
    """Validate that scaffolded node modules are syntactically correct."""

    node_name = "validate_syntax"
    phase, started = _start_phase(
        state,
        "validate_syntax",
        "Parse scaffolded node modules to ensure they contain valid Python syntax.",
    )

    errors: List[str] = []
    details: Dict[str, Any] = {}

    try:
        project_path = _resolve_project_path(state)
    except Exception as exc:
        message = f"Unable to resolve scaffold project path: {exc}"
        errors.append(message)
        summary = "Node syntax validation could not start."
        _finish_phase(state, phase, started, success=False, summary=summary, details=details, errors=errors)
        _record_validation_result(state, "syntax", False, errors, details)
        _update_build_report(state, node_name, success=False, errors=errors, details=details)
        return state

    node_modules = _collect_node_module_paths(project_path)
    missing_files: List[str] = []
    checked: List[str] = []

    if not node_modules:
        summary = "No scaffold node modules were discovered for syntax validation."
        details.update({"checked_modules": [], "missing_modules": []})
        _finish_phase(state, phase, started, success=True, summary=summary, details=details, errors=[])
        _record_validation_result(state, "syntax", True, [], details)
        _update_build_report(state, node_name, success=True, details=details)
        return state

    for name, path in node_modules.items():
        if not path.exists():
            missing_files.append(name)
            errors.append(f"{name}: module file not found at {path}")
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except Exception as exc:
            errors.append(f"{name}: unable to read source contents ({exc})")
            continue
        try:
            ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            location: List[str] = []
            if exc.lineno is not None:
                location.append(f"line {exc.lineno}")
            if exc.offset is not None:
                location.append(f"column {exc.offset}")
            suffix = f" ({', '.join(location)})" if location else ""
            errors.append(f"{name}: syntax error {exc.msg}{suffix}")
        except Exception as exc:
            errors.append(f"{name}: unexpected failure during syntax validation ({exc})")
        else:
            checked.append(name)

    details.update(
        {
            "checked_modules": sorted(checked),
            "missing_modules": sorted(missing_files),
        }
    )

    success = not errors
    summary = "Node syntax validation passed." if success else "Node syntax validation detected issues."
    _finish_phase(state, phase, started, success=success, summary=summary, details=details, errors=errors)
    _record_validation_result(state, "syntax", success, errors, details)
    _update_build_report(state, node_name, success=success, errors=errors, details=details)
    return state


def _has_required_imports(tree: ast.AST) -> tuple[bool, bool]:
    has_app_state = False
    has_client = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level >= 1:
                if module.endswith("state"):
                    if any(alias.name == "AppState" for alias in node.names):
                        has_app_state = True
                if module.endswith("llm"):
                    if any(alias.name == "client" for alias in node.names):
                        has_client = True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                target = alias.name
                if target.endswith(".state") and alias.asname == "AppState":
                    has_app_state = True
                if target.endswith(".llm.client") and (alias.asname == "client" or alias.name.endswith(".client")):
                    has_client = True
    return has_app_state, has_client


def validate_imports(state: ScaffoldState) -> ScaffoldState:
    """Ensure scaffolded node modules import expected dependencies."""

    node_name = "validate_imports"
    phase, started = _start_phase(
        state,
        "validate_imports",
        "Check scaffolded node modules for required AppState and client imports.",
    )

    errors: List[str] = []
    details: Dict[str, Any] = {}

    try:
        project_path = _resolve_project_path(state)
    except Exception as exc:
        message = f"Unable to resolve scaffold project path: {exc}"
        errors.append(message)
        summary = "Node import validation could not start."
        _finish_phase(state, phase, started, success=False, summary=summary, details=details, errors=errors)
        _record_validation_result(state, "imports", False, errors, details)
        _update_build_report(state, node_name, success=False, errors=errors, details=details)
        return state

    node_modules = _collect_node_module_paths(project_path)
    checked: List[str] = []

    if not node_modules:
        summary = "No scaffold node modules were discovered for import validation."
        details.update({"checked_modules": []})
        _finish_phase(state, phase, started, success=True, summary=summary, details=details, errors=[])
        _record_validation_result(state, "imports", True, [], details)
        _update_build_report(state, node_name, success=True, details=details)
        return state

    for name, path in node_modules.items():
        if not path.exists():
            errors.append(f"{name}: module file not found at {path}")
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except Exception as exc:
            errors.append(f"{name}: unable to read source contents ({exc})")
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except Exception as exc:
            errors.append(f"{name}: unable to parse module for import validation ({exc})")
            continue
        has_app_state, has_client = _has_required_imports(tree)
        missing: List[str] = []
        if not has_app_state:
            missing.append("from .state import AppState")
        if not has_client:
            missing.append("from ..llm import client")
        if missing:
            errors.append(f"{name}: missing required imports: {', '.join(missing)}")
        else:
            checked.append(name)

    details.update({"checked_modules": sorted(checked)})

    success = not errors
    summary = "Node import validation passed." if success else "Node import validation detected issues."
    _finish_phase(state, phase, started, success=success, summary=summary, details=details, errors=errors)
    _record_validation_result(state, "imports", success, errors, details)
    _update_build_report(state, node_name, success=success, errors=errors, details=details)
    return state


def _parse_app_state_fields(state_source: str) -> tuple[set[str], set[str]]:
    appstate_fields: set[str] = set()
    aggregator_fields: set[str] = set()
    match = re.search(
        r"class\s+AppState\(TypedDict,\s*total=False\):\n((?:    .+\n)+)",
        state_source,
    )
    if not match:
        return appstate_fields, aggregator_fields
    body = match.group(1)
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        name, annotation = stripped.split(":", 1)
        field_name = name.strip()
        appstate_fields.add(field_name)
        if "Annotated[" in annotation:
            aggregator_fields.add(field_name)
    return appstate_fields, aggregator_fields


def _collect_return_keys(module_tree: ast.AST) -> set[str]:
    keys: set[str] = set()
    for node in ast.walk(module_tree):
        if isinstance(node, ast.Return):
            value = node.value
            if isinstance(value, ast.Dict):
                for key in value.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        keys.add(key.value)
    return keys


def validate_concurrency_safety(state: ScaffoldState) -> ScaffoldState:
    """Validate that generated graphs update state keys safely."""

    node_name = "validate_concurrency_safety"
    phase, started = _start_phase(
        state,
        "validate_concurrency_safety",
        "Ensure generated graph/state definitions are concurrency safe.",
    )

    errors: List[str] = []
    details: Dict[str, Any] = {}

    try:
        project_path = _resolve_project_path(state)
    except Exception as exc:
        message = f"Unable to resolve scaffold project path: {exc}"
        errors.append(message)
        summary = "Concurrency safety validation could not start."
        _finish_phase(state, phase, started, success=False, summary=summary, details=details, errors=errors)
        _record_validation_result(state, "concurrency", False, errors, details)
        _update_build_report(state, node_name, success=False, errors=errors, details=details)
        return state

    graph_path = project_path / "src" / "agent" / "graph.py"
    state_path = project_path / "src" / "agent" / "state.py"

    graph_contents = ""
    if graph_path.exists():
        try:
            graph_contents = graph_path.read_text(encoding="utf-8")
        except Exception as exc:
            errors.append(f"Unable to read graph.py: {exc}")
    else:
        errors.append("graph.py is missing; unable to validate concurrency safety.")

    state_contents = ""
    if state_path.exists():
        try:
            state_contents = state_path.read_text(encoding="utf-8")
        except Exception as exc:
            errors.append(f"Unable to read state.py: {exc}")
    else:
        errors.append("state.py is missing; unable to validate concurrency safety.")

    appstate_fields, aggregator_fields = _parse_app_state_fields(state_contents)
    aggregator_fields_no_messages = {field for field in aggregator_fields if field != "messages"}

    start_edge_targets: set[str] = set()
    if graph_contents:
        for match in re.findall(r"add_edge\(\s*START\s*,\s*(['\"])([^'\"]+)\\1", graph_contents):
            target = match[1]
            if target != "END":
                start_edge_targets.add(target)
        if "StateGraph(AppState" not in graph_contents:
            errors.append("graph.py does not instantiate StateGraph(AppState).")

    if len(start_edge_targets) > 1 and not aggregator_fields_no_messages:
        errors.append(
            "Invalid concurrent graph update risk: multiple START fan-out with no per-key aggregators. Either serialize the start edges or annotate the keys."
        )

    details.update(
        {
            "start_edge_targets": sorted(start_edge_targets),
            "appstate_fields": sorted(appstate_fields),
            "aggregator_fields": sorted(aggregator_fields),
        }
    )

    node_modules = _collect_node_module_paths(project_path)
    undeclared_keys: set[str] = set()
    merge_offenders: List[str] = []

    for name, path in node_modules.items():
        if not path.exists():
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except Exception as exc:
            errors.append(f"{name}: unable to read source for concurrency validation ({exc})")
            continue
        if re.search(r"\{\s*\*\*state", source):
            merge_offenders.append(name)
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue
        except Exception as exc:
            errors.append(f"{name}: unable to parse module for concurrency validation ({exc})")
            continue
        undeclared_keys.update(
            key for key in _collect_return_keys(tree) if key not in appstate_fields and key != "scratch"
        )

    if merge_offenders:
        for offender in merge_offenders:
            errors.append(f"{offender}: detected state merge pattern (`{{**state ...}}`).")

    if undeclared_keys:
        errors.append(
            "Node modules return keys not declared in AppState: "
            + ", ".join(sorted(undeclared_keys))
        )
        details["undeclared_keys"] = sorted(undeclared_keys)

    if merge_offenders:
        details["state_merge_modules"] = sorted(merge_offenders)

    success = not errors
    summary = (
        "Concurrency safety validation passed."
        if success
        else "Concurrency safety validation detected issues."
    )
    _finish_phase(state, phase, started, success=success, summary=summary, details=details, errors=errors)
    _record_validation_result(state, "concurrency", success, errors, details)
    _update_build_report(state, node_name, success=success, errors=errors, details=details)
    return state


def validate_langgraph_compile(state: ScaffoldState) -> ScaffoldState:
    """Verify that the scaffolded LangGraph can be imported and compiled."""

    node_name = "validate_langgraph_compile"
    phase, started = _start_phase(
        state,
        "validate_langgraph_compile",
        "Import the generated LangGraph module and ensure compilation succeeds.",
    )

    errors: List[str] = []
    details: Dict[str, Any] = {}

    try:
        project_path = _resolve_project_path(state)
    except Exception as exc:
        message = f"Unable to resolve scaffold project path: {exc}"
        errors.append(message)
        summary = "LangGraph compilation validation could not start."
        _finish_phase(state, phase, started, success=False, summary=summary, details=details, errors=errors)
        _record_validation_result(state, "langgraph_compile", False, errors, details)
        _update_build_report(state, node_name, success=False, errors=errors, details=details)
        return state

    src_dir = project_path / "src"
    if not src_dir.exists():
        errors.append("Scaffolded project is missing the src directory required for imports.")
        summary = "LangGraph compilation validation failed."
        _finish_phase(state, phase, started, success=False, summary=summary, details=details, errors=errors)
        _record_validation_result(state, "langgraph_compile", False, errors, details)
        _update_build_report(state, node_name, success=False, errors=errors, details=details)
        return state

    sys_path_entry = str(project_path)
    added_path = False
    if sys_path_entry not in sys.path:
        sys.path.insert(0, sys_path_entry)
        added_path = True

    removed_modules: Dict[str, Any] = {}
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            removed_modules[name] = sys.modules.pop(name)

    try:
        graph_module = importlib.import_module("src.agent.graph")
        details["module_imported"] = True
        graph_attr = getattr(graph_module, "graph", None)
        details["graph_attribute_type"] = type(graph_attr).__name__ if graph_attr is not None else None
        with suppress(Exception):
            details["graph_repr"] = repr(graph_attr)
        try:
            compiled = graph_module.generate_dynamic_workflow({})
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("langgraph"):
                errors.append(f"LangGraph dependency is missing: {exc}")
            else:
                errors.append(f"Missing dependency during dynamic workflow generation: {exc}")
        except Exception as exc:
            errors.append(f"generate_dynamic_workflow failed: {exc}")
        else:
            details["compiled_graph_type"] = type(compiled).__name__
    except ModuleNotFoundError as exc:
        errors.append(f"Unable to import generated graph module: {exc}")
    except Exception as exc:
        errors.append(f"Error importing generated graph module: {exc}")
    finally:
        if added_path:
            with suppress(ValueError):
                sys.path.remove(sys_path_entry)
        current_src_modules = [name for name in list(sys.modules) if name == "src" or name.startswith("src.")]
        for name in current_src_modules:
            if name not in removed_modules:
                sys.modules.pop(name, None)
        for name, module in removed_modules.items():
            sys.modules[name] = module

    success = not errors
    summary = "LangGraph graph compiled successfully." if success else "LangGraph compilation validation detected issues."
    _finish_phase(state, phase, started, success=success, summary=summary, details=details, errors=errors)
    _record_validation_result(state, "langgraph_compile", success, errors, details)
    _update_build_report(state, node_name, success=success, errors=errors, details=details)
    return state
