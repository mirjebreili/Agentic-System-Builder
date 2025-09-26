from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from asb.llm.client import get_chat_model

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"```json\s*(.*?)```", re.IGNORECASE | re.DOTALL)

_DEFAULT_ARCHITECTURE: Dict[str, Any] = {
    "graph_structure": [],
    "state_flow": {},
    "conditional_edges": [],
    "entry_exit_points": {"entry": [], "exit": []},
}

_SYSTEM_PROMPT = (
    "You are a LangGraph architecture designer. Given structured requirements, "
    "propose a concise topology.\n"
    "Return ONLY a JSON object with keys: graph_structure (array describing node "
    "identities, responsibilities, and ordering), state_flow (object describing "
    "how state fields evolve across nodes), conditional_edges (array of conditional "
    "transitions), and entry_exit_points (object with entry and exit arrays)."
)

_REPAIR_PROMPT = (
    "The previous response must be valid JSON for the architecture schema. "
    "Return ONLY the corrected JSON object with the same information."
)


def _extract_json_block(text: str) -> str:
    match = _JSON_BLOCK.search(text or "")
    if match:
        return match.group(1)
    return text or ""


def _coerce_list(value: Any) -> list[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for item in value:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    items.append(cleaned)
            elif item is not None:
                items.append(str(item))
        return items
    if isinstance(value, dict):
        items: list[str] = []
        for key, item in value.items():
            if item is None:
                continue
            if isinstance(item, str) and item.strip():
                items.append(item.strip())
            else:
                items.append(f"{key}: {item}")
        return items
    if value is None:
        return []
    return [str(value)]


def _coerce_list_of_dicts(value: Any) -> list[Dict[str, Any]]:
    if isinstance(value, list):
        nodes: list[Dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                node = {str(k): item[k] for k in item.keys()}
                nodes.append(node)
            elif isinstance(item, str):
                nodes.append({"description": item})
            elif item is not None:
                nodes.append({"description": str(item)})
        return nodes
    if isinstance(value, dict):
        return [{str(k): v} for k, v in value.items()]
    if value is None:
        return []
    return [{"description": str(value)}]


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    if isinstance(value, list):
        return {str(idx): item for idx, item in enumerate(value)}
    if value is None:
        return {}
    return {"description": value}


def _coerce_entry_exit(value: Any) -> Dict[str, list[str]]:
    result = {"entry": [], "exit": []}
    if isinstance(value, dict):
        for key in ("entry", "entries", "start", "starts"):
            if key in value:
                result["entry"] = _coerce_list(value.get(key))
                break
        else:
            result["entry"] = _coerce_list(value.get("entry") or value.get("entry_points"))

        for key in ("exit", "exits", "end", "ends"):
            if key in value:
                result["exit"] = _coerce_list(value.get(key))
                break
        else:
            result["exit"] = _coerce_list(value.get("exit") or value.get("exit_points"))
        return result
    if value is None:
        return result
    entries = _coerce_list(value)
    if entries:
        result["entry"] = entries
    return result


def _truncate(value: str, limit: int = 120) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def design_architecture(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a LangGraph architecture blueprint from requirements."""

    requirements = state.get("requirements", {}) or {}
    try:
        requirements_text = json.dumps(requirements, ensure_ascii=False, indent=2)
    except TypeError:
        requirements_text = str(requirements)

    logger.info(
        "Designing architecture for requirements snippet: %s",
        _truncate(requirements_text.replace("\n", " ") if requirements_text else "<empty>")
    )

    llm = get_chat_model()
    sys_msg = SystemMessage(_SYSTEM_PROMPT)
    user_msg = HumanMessage(
        "Design a LangGraph architecture based on these requirements:\n" + requirements_text
    )

    try:
        response = llm.invoke([sys_msg, user_msg]).content
    except Exception:
        logger.exception("LLM call failed during architecture design.")
        updated_state = dict(state)
        updated_state["architecture"] = dict(_DEFAULT_ARCHITECTURE)
        return updated_state

    raw_json = _extract_json_block(response)
    logger.debug("Raw architecture LLM output: %s", response)

    parsed: Dict[str, Any] | None = None
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        logger.warning("Initial architecture response not valid JSON; attempting repair.")
        try:
            repair = llm.invoke(
                [SystemMessage(_REPAIR_PROMPT), HumanMessage(response)]
            ).content
            logger.debug("Architecture repair output: %s", repair)
            parsed = json.loads(_extract_json_block(repair))
        except Exception:
            logger.exception("Failed to repair architecture JSON output.")
            parsed = None

    if not isinstance(parsed, dict):
        parsed = {}

    architecture = dict(_DEFAULT_ARCHITECTURE)
    architecture["graph_structure"] = _coerce_list_of_dicts(parsed.get("graph_structure"))
    architecture["state_flow"] = _coerce_mapping(parsed.get("state_flow"))
    architecture["conditional_edges"] = _coerce_list_of_dicts(parsed.get("conditional_edges"))
    architecture["entry_exit_points"] = _coerce_entry_exit(parsed.get("entry_exit_points"))

    logger.info(
        "Architecture designed: nodes=%d, conditional_edges=%d",
        len(architecture["graph_structure"]),
        len(architecture["conditional_edges"]),
    )

    updated_state = dict(state)
    updated_state["architecture"] = architecture
    logger.debug("Architecture debug - processed architecture: %s", architecture)
    return updated_state


def architecture_designer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node wrapper for :func:`design_architecture`."""

    messages = list(state.get("messages") or [])

    try:
        updated_state = design_architecture(state)
        architecture = updated_state.get("architecture") or dict(_DEFAULT_ARCHITECTURE)
        summary_lines = ["[architecture]"]
        nodes = architecture.get("graph_structure") or []
        if nodes:
            node_summaries = []
            for node in nodes[:4]:
                node_id = str(node.get("id") or node.get("name") or "node")
                node_type = node.get("type")
                if node_type:
                    node_summaries.append(f"{node_id}({node_type})")
                else:
                    node_summaries.append(node_id)
            if len(nodes) > 4:
                node_summaries.append("…")
            summary_lines.append("Nodes: " + ", ".join(node_summaries))
        entry_exit = architecture.get("entry_exit_points") or {}
        entries = _coerce_list(entry_exit.get("entry"))
        exits = _coerce_list(entry_exit.get("exit"))
        if entries:
            summary_lines.append("Entry: " + ", ".join(entries))
        if exits:
            summary_lines.append("Exit: " + ", ".join(exits))
        cond_edges = architecture.get("conditional_edges") or []
        if cond_edges:
            summary_lines.append(f"Conditional edges: {len(cond_edges)}")
        flow_keys = list((architecture.get("state_flow") or {}).keys())
        if flow_keys:
            summary_lines.append("State keys: " + ", ".join(flow_keys[:4]))
        messages.append({"role": "assistant", "content": "\n".join(summary_lines)})
        updated_state["messages"] = messages
        return updated_state
    except Exception:
        logger.exception("Unhandled error within architecture designer node.")
        messages.append({
            "role": "assistant",
            "content": "[architecture-error]\nUnable to design architecture at this time.",
        })
        fallback_state = dict(state)
        fallback_state.setdefault("architecture", dict(_DEFAULT_ARCHITECTURE))
        fallback_state["messages"] = messages
        return fallback_state
