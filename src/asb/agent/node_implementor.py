from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from asb.llm.client import get_chat_model
from asb.utils.message_utils import (
    extract_last_message_content,
    extract_user_messages_content,
)

logger = logging.getLogger(__name__)

_PYTHON_BLOCK = re.compile(r"```python\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_DEFAULT_STUB_TEMPLATE = (
    "def {function_name}(state: Dict[str, Any]) -> Dict[str, Any]:\n"
    "    \"\"\"Auto-generated stub for node '{node_id}' of type '{node_type}'.\"\"\"\n"
    "    return state\n"
)


def personalize_prompts(state: Dict[str, Any], node: Dict[str, Any]) -> Dict[str, Any]:
    messages = list(state.get("messages") or [])

    user_messages = extract_user_messages_content(messages)
    latest_user_content = user_messages[-1] if user_messages else None
    if latest_user_content is None and messages:
        latest_user_content = extract_last_message_content(messages, "")

    plan_goal = None
    plan = state.get("plan")
    if isinstance(plan, dict):
        plan_goal = plan.get("goal")

    user_goal = (latest_user_content or plan_goal or "").strip()

    normalized_content = latest_user_content.lower() if latest_user_content else ""
    task_type = None
    format_hint = ""
    if any(keyword in normalized_content for keyword in ["bullet point", "bullet-point", "bulletpoint", "bullet list", "bulleted"]):
        task_type = "bullet_points"
        format_hint = (
            "Use bullet points (for example, '- item') in any explanatory text or comments so the user receives bullet points."
        )
    elif "headline" in normalized_content:
        task_type = "headline"
        format_hint = "Add a concise headline-style comment summarizing the implementation."
    elif "summary" in normalized_content:
        task_type = "summary"
        format_hint = "Include a short summary comment (2-3 sentences) explaining the implementation."

    goal_phrase = user_goal or "the requested LangGraph node"
    workflow_steps = [
        f"Generate candidate implementations that address {goal_phrase}.",
        "Evaluate the candidates for correctness, maintainability, and compatibility with the LangGraph architecture.",
        "Select the strongest candidate before finalizing the code.",
    ]
    final_instruction = "Final answer: deliver the selected implementation as polished Python code."
    if format_hint:
        final_instruction = (
            "Final answer: deliver the selected implementation as polished Python code while honoring this format guidance: "
            + format_hint
        )
    workflow_steps.append(final_instruction)

    system_fragment = "Workflow:\n" + "\n".join(
        f"{index + 1}. {line}" for index, line in enumerate(workflow_steps)
    )

    user_lines = []
    if user_goal:
        user_lines.append(f"User goal/context: {user_goal}")
    if format_hint:
        user_lines.append(f"Format guidance: {format_hint}")
    user_lines.append("Task steps:")
    user_lines.extend(f"{index + 1}. {line}" for index, line in enumerate(workflow_steps))
    user_fragment = "\n".join(user_lines)

    return {
        "user_goal": user_goal,
        "task_type": task_type,
        "format_hint": format_hint,
        "system_fragment": system_fragment,
        "user_fragment": user_fragment,
    }


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

    personalization = personalize_prompts(state, node)

    system_lines = [
        "You implement Python LangGraph nodes.",
        "Return concise, production-quality code for the requested node only.",
    ]
    system_fragment = personalization.get("system_fragment")
    if system_fragment:
        system_lines.extend(["", system_fragment])
    system_prompt = "\n".join(system_lines)

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

    user_lines = [
        "Implement the LangGraph node using a function or class as appropriate.",
        "",
    ]
    user_lines.extend(details)
    user_fragment = personalization.get("user_fragment")
    if user_fragment:
        user_lines.extend(["", user_fragment])
    final_instruction = "Return only Python code in a single code block."
    format_hint = personalization.get("format_hint")
    if format_hint:
        final_instruction = (
            final_instruction
            + " If you need to include explanatory text, follow the format guidance above."
        )
    user_lines.extend(["", final_instruction])
    user_prompt = "\n".join(line for line in user_lines if line is not None)
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
    previous_files = state.get("generated_files") or {}

    updated_state = dict(state)
    generated_files = dict(previous_files)
    updated_state["generated_files"] = generated_files
    updated_state["implemented_nodes"] = []
    updated_state["last_implemented_node"] = None

    implemented_nodes: List[Dict[str, Any]] = []
    llm = None

    while True:
        node_id, node = _select_next_unimplemented_node(updated_state)
        if not node_id or node is None:
            if not implemented_nodes:
                logger.info("No remaining nodes to implement.")
            break

        node_type = node.get("type", "unspecified")
        system_prompt, user_prompt = _build_prompts(node_id, node, updated_state)

        if llm is None:
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
        previous_code = previous_files.get(filename)
        generated_files[filename] = code

        implemented_nodes.append(
            {
                "node_id": node_id,
                "filename": filename,
                "new_code": previous_code != code,
            }
        )

        updated_state["last_implemented_node"] = node_id

    if not implemented_nodes:
        updated_state["generated_files"] = previous_files

    updated_state["implemented_nodes"] = implemented_nodes
    return updated_state


def node_implementor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    messages = list(state.get("messages") or [])
    previous_files = dict(state.get("generated_files") or {})

    try:
        updated_state = implement_single_node(state)
        implemented_nodes = updated_state.get("implemented_nodes") or []

        produced_any = any(node_info.get("new_code") for node_info in implemented_nodes)
        if not produced_any:
            updated_state["generated_files"] = previous_files

        if implemented_nodes:
            summary_lines = ["[node-implementor]", "Implemented nodes:"]
            for node_info in implemented_nodes:
                node_id = node_info.get("node_id", "unknown")
                filename = node_info.get("filename", "unknown")
                status = "new code" if node_info.get("new_code") else "no new code"
                summary_lines.append(f"- {node_id} ({status}) -> {filename}")
            summary = "\n".join(summary_lines)
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
