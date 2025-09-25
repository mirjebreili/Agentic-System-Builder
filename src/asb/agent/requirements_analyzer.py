from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from asb.llm.client import get_chat_model
from asb.utils.message_utils import extract_last_message_content, safe_message_access

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"```json\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_DEFAULT_REQUIREMENTS: Dict[str, Any] = {
    "nodes_needed": [],
    "state_schema": {},
    "tools_required": [],
    "dependencies": [],
    "complexity_level": "unknown",
}

_SYSTEM_PROMPT = (
    "You analyze product requirements for a LangGraph workflow.\n"
    "Return ONLY a JSON object with keys: nodes_needed (array of node descriptions),\n"
    "state_schema (object describing state fields), tools_required (array),\n"
    "dependencies (array describing ordering/relationships), and complexity_level\n"
    "(one of: low, medium, high, extreme). Provide concise yet concrete details."
)

_REPAIR_PROMPT = (
    "The previous response must be valid JSON matching the required schema. "
    "Return ONLY the corrected JSON object with the same information."
)


def _extract_json_block(text: str) -> str:
    match = _JSON_BLOCK.search(text or "")
    if match:
        return match.group(1)
    return text or ""


def _coerce_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Mapping):
        items: list[str] = []
        for key, item in value.items():
            if item is None:
                continue
            if isinstance(item, str) and item.strip():
                items.append(item.strip())
            else:
                items.append(f"{key}: {item}")
        return items
    if isinstance(value, Iterable):
        items: list[str] = []
        for item in value:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    items.append(cleaned)
            elif item is not None:
                items.append(str(item))
        return items
    return []


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        result: Dict[str, Any] = {}
        for idx, item in enumerate(value):
            key = f"field_{idx}"
            result[key] = item
        return result
    if value is None:
        return {}
    return {"description": value}


def _truncate(value: str, limit: int = 120) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def analyze_requirements(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the latest user request and return structured LangGraph requirements."""

    messages = list(state.get("messages") or [])
    latest_user_prompt = ""
    for message in reversed(messages):
        role = safe_message_access(message, "role", "")
        if not role:
            role = str(safe_message_access(message, "type", "")).lower()
        else:
            role = str(role).lower()
        content = safe_message_access(message, "content", "")
        if not content and isinstance(message, str):
            content = message
        if role in {"user", "human"} and content:
            latest_user_prompt = str(content)
            break
    if not latest_user_prompt and messages:
        latest_user_prompt = extract_last_message_content(messages, "")

    logger.info(
        "Analyzing requirements for prompt snippet: %s",
        _truncate(latest_user_prompt or "<empty>")
    )

    llm = get_chat_model()
    sys_msg = SystemMessage(_SYSTEM_PROMPT)
    user_msg = HumanMessage(latest_user_prompt or "No explicit user request provided.")

    try:
        response = llm.invoke([sys_msg, user_msg]).content
    except Exception:
        logger.exception("LLM call failed during requirements analysis.")
        updated_state = dict(state)
        updated_state["requirements"] = dict(_DEFAULT_REQUIREMENTS)
        return updated_state

    raw_json = _extract_json_block(response)
    logger.debug("Raw requirements LLM output: %s", response)

    parsed: Dict[str, Any] | None = None
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        logger.warning("Initial requirements response not valid JSON; attempting repair.")
        try:
            repair = llm.invoke([SystemMessage(_REPAIR_PROMPT), HumanMessage(response)]).content
            logger.debug("Requirements repair output: %s", repair)
            parsed = json.loads(_extract_json_block(repair))
        except Exception:
            logger.exception("Failed to repair requirements JSON output.")
            parsed = None

    if not isinstance(parsed, dict):
        parsed = {}

    requirements = dict(_DEFAULT_REQUIREMENTS)
    requirements["nodes_needed"] = _coerce_list(parsed.get("nodes_needed"))
    requirements["state_schema"] = _coerce_mapping(parsed.get("state_schema"))
    requirements["tools_required"] = _coerce_list(parsed.get("tools_required"))
    requirements["dependencies"] = _coerce_list(parsed.get("dependencies"))
    complexity = parsed.get("complexity_level")
    if isinstance(complexity, str) and complexity.strip():
        requirements["complexity_level"] = complexity.strip().lower()

    logger.info(
        "Requirements extracted: nodes=%d, tools=%d, complexity=%s",
        len(requirements["nodes_needed"]),
        len(requirements["tools_required"]),
        requirements["complexity_level"],
    )

    updated_state = dict(state)
    updated_state["requirements"] = requirements
    return updated_state


def requirements_analyzer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node wrapper for :func:`analyze_requirements`."""

    messages = list(state.get("messages") or [])

    try:
        updated_state = analyze_requirements(state)
        requirements = updated_state.get("requirements") or dict(_DEFAULT_REQUIREMENTS)
        summary_lines = ["[requirements]"]
        if requirements.get("nodes_needed"):
            summary_lines.append("Nodes: " + ", ".join(requirements["nodes_needed"]))
        if requirements.get("tools_required"):
            summary_lines.append("Tools: " + ", ".join(requirements["tools_required"]))
        if requirements.get("dependencies"):
            summary_lines.append("Dependencies: " + ", ".join(requirements["dependencies"]))
        summary_lines.append(f"Complexity: {requirements.get('complexity_level', 'unknown')}")
        messages.append({"role": "assistant", "content": "\n".join(summary_lines)})
        updated_state["messages"] = messages
        return updated_state
    except Exception:
        logger.exception("Unhandled error within requirements analyzer node.")
        messages.append({
            "role": "assistant",
            "content": "[requirements-error]\nUnable to analyze requirements at this time."
        })
        fallback_state = dict(state)
        fallback_state.setdefault("requirements", dict(_DEFAULT_REQUIREMENTS))
        fallback_state["messages"] = messages
        return fallback_state
