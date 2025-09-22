from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from asb.llm.client import get_chat_model

logger = logging.getLogger(__name__)

_PYTHON_BLOCK = re.compile(r"```python\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_DEFAULT_STUB_TEMPLATE = (
    "def {function_name}(state: Dict[str, Any]) -> Dict[str, Any]:\n"
    "    \"\"\"Auto-generated stub for node '{node_id}' of type '{node_type}'.\"\"\"\n"
    "    return state\n"
)


def _iter_graph_nodes(state: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    architecture = state.get("architecture") or {}
    nodes = architecture.get("graph_structure") or []
    if isinstance(nodes, dict):
        for node_id, node in nodes.items():
            if isinstance(node, dict):
                merged = dict(node)
                merged.setdefault("id", node_id)
                yield merged
            else:
                yield {"id": str(node_id), "description": node}
        return
    if not isinstance(nodes, list):
        return
    for node in nodes:
        if isinstance(node, dict):
            yield node


def _sanitize_identifier(node_id: str) -> str:
    sanitized = re.sub(r"\W+", "_", node_id).strip("_")
    return sanitized or "node"


def _node_filename(node_id: str) -> str:
    return f"{_sanitize_identifier(node_id)}.py"


def _select_next_unimplemented_node(
    state: Dict[str, Any]
) -> Tuple[str | None, Dict[str, Any] | None]:
    generated = state.get("generated_files") or {}
    for node in _iter_graph_nodes(state):
        node_id: str | None = None
        for key in ("id", "node", "name", "label"):
            value = node.get(key)
            if value is None:
                continue
            candidate = str(value).strip()
            if candidate:
                node_id = candidate
                break
        if not node_id:
            continue
        sanitized_id = _sanitize_identifier(node_id)
        filename = _node_filename(node_id)
        legacy_filename = f"{node_id}.py" if node_id != sanitized_id else filename
        if filename in generated or legacy_filename in generated:
            continue
        enriched_node = dict(node)
        enriched_node.setdefault("id", node_id)
        enriched_node["_sanitized_id"] = sanitized_id
        return node_id, enriched_node
    return None, None


def _build_prompts(node_id: str, node: Dict[str, Any], state: Dict[str, Any]) -> Tuple[str, str]:
    architecture = state.get("architecture") or {}
    node_type = node.get("type", "unspecified")
    description = node.get("description") or "No description provided."
    state_flow = architecture.get("state_flow") or {}
    node_flow = state_flow.get(node_id) or state_flow.get(str(node_id))
    conditional_edges = architecture.get("conditional_edges") or []
    relevant_edges = []
    for edge in conditional_edges:
        if not isinstance(edge, dict):
            continue
        if edge.get("from") == node_id or edge.get("to") == node_id:
            relevant_edges.append(edge)

    system_prompt = (
        "You implement Python LangGraph nodes.\n"
        "Return concise, production-quality code for the requested node only."
    )

    details = [
        f"Node id: {node_id}",
        f"Node type: {node_type}",
        f"Description: {description}",
    ]
    if node_flow:
        details.append(f"State flow guidance: {node_flow}")
    if relevant_edges:
        edges_lines = []
        for edge in relevant_edges:
            frm = edge.get("from")
            to = edge.get("to")
            condition = edge.get("condition") or edge.get("if")
            edges_lines.append(
                f"- {frm} -> {to}" + (f" when {condition}" if condition else "")
            )
        details.append("Conditional edges:\n" + "\n".join(edges_lines))

    user_prompt = (
        "Implement the LangGraph node using a function or class as appropriate.\n"
        + "\n".join(details)
        + "\nReturn only Python code in a single code block."
    )
    return system_prompt, user_prompt


def _extract_python_block(text: str) -> str:
    if not text:
        return ""
    match = _PYTHON_BLOCK.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _render_stub(node_id: str, node_type: str) -> str:
    function_name = _sanitize_identifier(node_id)
    return (
        "from typing import Any, Dict\n\n"
        + _DEFAULT_STUB_TEMPLATE.format(
            function_name=function_name,
            node_id=node_id,
            node_type=node_type or "unspecified",
        )
    )


def implement_single_node(state: Dict[str, Any]) -> Dict[str, Any]:
    node_id, node = _select_next_unimplemented_node(state)
    previous_files = state.get("generated_files") or {}

    updated_state = dict(state)
    updated_state["generated_files"] = dict(previous_files)
    updated_state["last_implemented_node"] = None

    if not node_id or node is None:
        logger.info("No remaining nodes to implement.")
        return updated_state

    node_type = node.get("type", "unspecified")
    system_prompt, user_prompt = _build_prompts(node_id, node, state)

    llm = get_chat_model()
    try:
        response = llm.invoke([
            SystemMessage(system_prompt),
            HumanMessage(user_prompt),
        ]).content
    except Exception:
        logger.exception("LLM call failed while implementing node %s.", node_id)
        response = None

    code = _extract_python_block(response or "")
    if not code:
        logger.warning("No code extracted for node %s; using stub.", node_id)
        code = _render_stub(node_id, node_type)
    else:
        logger.debug("Generated code for node %s: %s", node_id, code[:2000])

    sanitized_id = node.get("_sanitized_id") if node else None
    filename = _node_filename(sanitized_id or node_id)
    updated_state["generated_files"][filename] = code
    updated_state["last_implemented_node"] = node_id
    return updated_state


def node_implementor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    messages = list(state.get("messages") or [])
    previous_files = dict(state.get("generated_files") or {})

    try:
        updated_state = implement_single_node(state)
        new_files = updated_state.get("generated_files") or {}
        last_node = updated_state.get("last_implemented_node")

        if last_node:
            filename = _node_filename(last_node)
            previous_code = previous_files.get(filename)
            new_code = new_files.get(filename)
            produced = new_code is not None and new_code != previous_code
            status = "new code" if produced else "no new code"
            summary = f"[node-implementor]\nImplemented node {last_node} ({status})."
        else:
            summary = "[node-implementor]\nNo nodes remaining to implement."

        messages.append({"role": "assistant", "content": summary})
        updated_state["messages"] = messages
        return updated_state
    except Exception:
        logger.exception("Unhandled error within node implementor node.")
        messages.append({
            "role": "assistant",
            "content": "[node-implementor-error]\nFailed to implement node.",
        })
        fallback_state = dict(state)
        fallback_state["messages"] = messages
        fallback_state["generated_files"] = previous_files
        return fallback_state
