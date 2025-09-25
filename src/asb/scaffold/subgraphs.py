"""LangGraph subgraphs for scaffold build, validation, and repair phases."""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

from .build_nodes import (
    copy_base_files,
    init_project_structure,
    write_config_files,
    write_graph_module,
    write_node_modules,
    write_state_schema,
)
from .repair_nodes import fix_empty_nodes, fix_graph_compilation, fix_import_errors
from .validate_nodes import (
    validate_concurrency_safety,
    validate_imports,
    validate_langgraph_compile,
    validate_non_empty_generation,
    validate_syntax,
)

ScaffoldState = Dict[str, Any]


def _ensure_scaffold_lists(state: MutableMapping[str, Any]) -> None:
    """Ensure scaffold bookkeeping lists exist so micro-nodes can populate them."""

    scaffold = state.setdefault("scaffold", {})
    if not isinstance(scaffold, MutableMapping):  # pragma: no cover - defensive
        scaffold = {}
        state["scaffold"] = scaffold
    scaffold.setdefault("errors", [])
    scaffold.setdefault("missing", [])


def _init_project_structure_node(state: ScaffoldState) -> ScaffoldState:
    init_project_structure(state)
    _ensure_scaffold_lists(state)
    return state


def _write_config_files_node(state: ScaffoldState) -> ScaffoldState:
    write_config_files(state)
    return state


def _copy_base_files_node(state: ScaffoldState) -> ScaffoldState:
    missing = copy_base_files(state)
    state["_scaffold_missing_files"] = list(missing)
    scaffold = state.get("scaffold")
    if isinstance(scaffold, MutableMapping):
        scaffold["missing"] = list(missing)
    return state


def _write_state_schema_node(state: ScaffoldState) -> ScaffoldState:
    write_state_schema(state)
    return state


def _write_node_modules_node(state: ScaffoldState) -> ScaffoldState:
    node_definitions, _ = write_node_modules(state)
    state["_scaffold_node_definitions"] = node_definitions
    return state


def _write_graph_module_node(state: ScaffoldState) -> ScaffoldState:
    write_graph_module(state)
    return state


def create_build_subgraph() -> Any:
    graph = StateGraph(dict)
    graph.add_node("init_project_structure", _init_project_structure_node)
    graph.add_node("write_config_files", _write_config_files_node)
    graph.add_node("copy_base_files", _copy_base_files_node)
    graph.add_node("write_state_schema", _write_state_schema_node)
    graph.add_node("write_node_modules", _write_node_modules_node)
    graph.add_node("write_graph_module", _write_graph_module_node)

    graph.add_edge(START, "init_project_structure")
    graph.add_edge("init_project_structure", "write_config_files")
    graph.add_edge("write_config_files", "copy_base_files")
    graph.add_edge("copy_base_files", "write_state_schema")
    graph.add_edge("write_state_schema", "write_node_modules")
    graph.add_edge("write_node_modules", "write_graph_module")
    graph.add_edge("write_graph_module", END)

    return graph.compile()


def create_validate_subgraph() -> Any:
    graph = StateGraph(dict)
    graph.add_node("validate_non_empty_generation", validate_non_empty_generation)
    graph.add_node("validate_syntax", validate_syntax)
    graph.add_node("validate_imports", validate_imports)
    graph.add_node("validate_concurrency_safety", validate_concurrency_safety)
    graph.add_node("validate_langgraph_compile", validate_langgraph_compile)

    graph.add_edge(START, "validate_non_empty_generation")
    graph.add_edge("validate_non_empty_generation", "validate_syntax")
    graph.add_edge("validate_syntax", "validate_imports")
    graph.add_edge("validate_imports", "validate_concurrency_safety")
    graph.add_edge("validate_concurrency_safety", "validate_langgraph_compile")
    graph.add_edge("validate_langgraph_compile", END)

    return graph.compile()


def _collect_scaffold_errors(state: ScaffoldState) -> List[str]:
    scaffold = state.get("scaffold")
    if isinstance(scaffold, MutableMapping):
        errors = scaffold.get("errors")
        if isinstance(errors, Iterable):
            return [str(error) for error in errors if error]
    return []


def _needs_empty_node_repair(state: ScaffoldState) -> bool:
    return any("node file is empty" in error for error in _collect_scaffold_errors(state))


def _needs_import_repair(state: ScaffoldState) -> bool:
    return any("missing required imports" in error for error in _collect_scaffold_errors(state))


def _needs_graph_repair(state: ScaffoldState) -> bool:
    for error in _collect_scaffold_errors(state):
        lowered = error.lower()
        if "generate_dynamic_workflow failed" in lowered:
            return True
        if "unable to import generated graph module" in lowered:
            return True
    return False


def _route_after_empty_nodes(state: ScaffoldState) -> str:
    if _needs_import_repair(state):
        return "fix_import_errors"
    if _needs_graph_repair(state):
        return "fix_graph_compilation"
    return "complete"


def _route_after_import_repairs(state: ScaffoldState) -> str:
    if _needs_graph_repair(state):
        return "fix_graph_compilation"
    return "complete"


def create_repair_subgraph() -> Any:
    graph = StateGraph(dict)
    graph.add_node("fix_empty_nodes", fix_empty_nodes)
    graph.add_node("fix_import_errors", fix_import_errors)
    graph.add_node("fix_graph_compilation", fix_graph_compilation)

    graph.add_edge(START, "fix_empty_nodes")
    graph.add_conditional_edges(
        "fix_empty_nodes",
        _route_after_empty_nodes,
        {
            "fix_import_errors": "fix_import_errors",
            "fix_graph_compilation": "fix_graph_compilation",
            "complete": END,
        },
    )
    graph.add_conditional_edges(
        "fix_import_errors",
        _route_after_import_repairs,
        {
            "fix_graph_compilation": "fix_graph_compilation",
            "complete": END,
        },
    )
    graph.add_edge("fix_graph_compilation", END)

    return graph.compile()
