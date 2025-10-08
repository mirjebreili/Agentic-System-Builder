from __future__ import annotations

"""Runtime plan execution utilities.

This module inspects ``Plan`` instances produced by the planner and builds a
mapping of concrete Python callables that can execute each plan node. The
resulting registry is consumed by :func:`execute` (and the compatibility helper
``execute_deep``) to run nodes sequentially.
"""

from typing import Any, Callable, Dict, Iterable, List, Tuple

from agents.planner import Plan, PlanNode
from agents.state import AppState
from llm.client import get_chat_model, run_llm


# ``NODE_IMPLEMENTATIONS`` is populated at runtime via
# :func:`update_node_implementations`. Each entry contains the node identifier
# alongside the callable that should process that node when executing the plan.
NODE_IMPLEMENTATIONS: List[Tuple[str, Callable[[AppState], AppState]]] = []


# Mapping of plan node types to handler factories. Each handler factory receives
# the :class:`PlanNode` instance and returns a callable that accepts and returns
# the execution state.
_NODE_TYPE_HANDLERS: Dict[
    str, Callable[[PlanNode], Callable[[AppState], AppState]]
] = {}


def _record_result(state: AppState, node_id: str, result: Any) -> None:
    """Helper to store execution results in ``state``."""

    results: Dict[str, Any] = dict(state.get("results") or {})
    results[node_id] = result
    state["results"] = results
    state["last_node"] = node_id


def _llm_handler(node: PlanNode) -> Callable[[AppState], AppState]:
    prompt = node.prompt or ""

    def _fn(state: AppState) -> AppState:
        result = run_llm(prompt=prompt, state=state)
        _record_result(state, node.id, result)
        output_text = result.get("output_text") if isinstance(result, dict) else None
        if output_text:
            tagged_output = f"[{node.id}]\n{output_text}"
            messages = list(state.get("messages") or [])
            messages.append({"role": "assistant", "content": tagged_output})
            state["messages"] = messages
        return state

    return _fn


def _python_handler(node: PlanNode) -> Callable[[AppState], AppState]:
    target = (node.tool or "").strip()
    if not target or ":" not in target:

        def _noop(state: AppState) -> AppState:
            warnings = list(state.get("warnings") or [])
            warnings.append(f"No python tool for node {node.id}")
            state["warnings"] = warnings
            return state

        return _noop

    module_path, func_name = target.split(":", 1)
    module_path = module_path.strip()
    func_name = func_name.strip()

    def _fn(state: AppState) -> AppState:
        import importlib

        try:
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
        except Exception as exc:  # pragma: no cover - defensive guard
            errors = list(state.get("errors") or [])
            errors.append(
                f"Failed to import tool '{target}' for node {node.id}: {exc}"
            )
            state["errors"] = errors
            return state

        try:
            result = func(state)
        except Exception as exc:  # pragma: no cover - defensive guard
            errors = list(state.get("errors") or [])
            errors.append(f"Error executing node {node.id}: {exc}")
            state["errors"] = errors
            return state

        _record_result(state, node.id, result)
        return state

    return _fn


def _tool_handler(node: PlanNode) -> Callable[[AppState], AppState]:
    # ``tool`` nodes share the same implementation semantics as ``python``
    # nodes; this alias exists for clarity and potential future divergence.
    return _python_handler(node)


_NODE_TYPE_HANDLERS.update(
    {
        "llm": _llm_handler,
        "python": _python_handler,
        "tool": _tool_handler,
    }
)


def update_node_implementations(plan: Plan | Dict[str, Any]) -> None:
    """Populate :data:`NODE_IMPLEMENTATIONS` for the supplied plan."""

    global NODE_IMPLEMENTATIONS
    if isinstance(plan, Plan):
        plan_model = plan
    else:
        plan_model = Plan.model_validate(plan)

    NODE_IMPLEMENTATIONS = []
    for node in plan_model.nodes:
        handler_factory = _NODE_TYPE_HANDLERS.get(node.type)
        if handler_factory is None:

            def _unknown(state: AppState, *, _nid=node.id, _type=node.type) -> AppState:
                errors = list(state.get("errors") or [])
                errors.append(f"Unknown node type {_type} for {_nid}")
                state["errors"] = errors
                return state

            NODE_IMPLEMENTATIONS.append((node.id, _unknown))
            continue

        NODE_IMPLEMENTATIONS.append((node.id, handler_factory(node)))


def iter_node_callables() -> Iterable[Tuple[str, Callable[[AppState], AppState]]]:
    """Yield the currently registered node callables in execution order."""

    yield from NODE_IMPLEMENTATIONS


def execute(state: AppState) -> AppState:
    """Execute nodes sequentially using the populated registry."""

    current_state = state
    for _, node_callable in iter_node_callables():
        current_state = node_callable(current_state)
    return current_state


def execute_deep(state: Dict[str, Any]) -> Dict[str, Any]:
    """Compatibility helper that prepares node callables and executes them."""

    plan_data = state.get("plan")
    if not plan_data:
        errors = list(state.get("errors") or [])
        errors.append("No plan available for execution.")
        state["errors"] = errors
        return state

    update_node_implementations(plan_data)
    return execute(state)

