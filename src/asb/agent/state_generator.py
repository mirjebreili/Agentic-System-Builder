"""Generate a TypedDict state schema module from architecture metadata."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Literal, TypedDict

from asb.agent.scaffold import generate_adaptive_state_schema

logger = logging.getLogger(__name__)


_BASE_FIELDS: List[tuple[str, str]] = [
    ("architecture", "Dict[str, Any]"),
    ("artifacts", "Dict[str, Any]"),
    ("build_attempts", "int"),
    ("code_fixes", "Dict[str, Any]"),
    ("code_validation", "Dict[str, Any]"),
    ("consecutive_failures", "int"),
    ("coordinator_decision", "str"),
    ("current_step", "Dict[str, bool]"),
    ("debug", "Dict[str, Any]"),
    ("error", "str"),
    ("evaluations", "List[Dict[str, Any]]"),
    ("fix_attempts", "int"),
    ("fix_strategy_used", "str | None"),
    ("flags", "Dict[str, bool]"),
    ("generated_files", "Dict[str, str]"),
    ("goal", "str"),
    ("implemented_nodes", "List[Dict[str, Any]]"),
    ("input_text", "str"),
    ("last_implemented_node", "str | None"),
    ("last_user_input", "str"),
    ("messages", "List[ChatMessage]"),
    ("metrics", "Dict[str, Any]"),
    ("next_action", "str"),
    ("passed", "bool"),
    ("plan", "Dict[str, Any]"),
    ("replan", "bool"),
    ("repair_start_time", "float"),
    ("report", "Dict[str, Any]"),
    ("requirements", "Dict[str, Any]"),
    ("review", "Dict[str, Any]"),
    ("sandbox", "Dict[str, Any]"),
    ("scaffold", "Dict[str, Any]"),
    ("selected_thought", "Dict[str, Any]"),
    ("syntax_validation", "Dict[str, Any]"),
    ("tests", "Dict[str, Any]"),
    ("thoughts", "List[str]"),
    ("tot", "Dict[str, Any]"),
    ("validation_errors", "List[str]"),
]


def _infer_collection_type(key: str, value: Any | None = None) -> str:
    """Infer a reasonable type hint for an application state field."""

    normalized = key.lower()

    base_lookup = {name: annotation for name, annotation in _BASE_FIELDS}
    if normalized in base_lookup:
        return base_lookup[normalized]

    list_indicators: Iterable[str] = (
        "messages",
        "items",
        "steps",
        "entries",
        "history",
        "logs",
        "tasks",
        "nodes",
    )
    dict_indicators: Iterable[str] = (
        "config",
        "settings",
        "data",
        "info",
        "details",
        "metadata",
        "context",
        "results",
        "state",
        "map",
        "graph",
        "architecture",
        "requirements",
        "report",
        "summary",
        "plan",
        "artifacts",
    )

    bool_indicators: Iterable[str] = (
        "replan",
        "passed",
        "ready",
        "complete",
        "completed",
        "approved",
        "finished",
        "done",
        "success",
    )

    if any(indicator in normalized for indicator in list_indicators):
        return "List[Any]"

    if any(indicator in normalized for indicator in dict_indicators):
        return "Dict[str, Any]"

    if normalized.startswith("has_") or normalized.startswith("is_"):
        return "bool"

    if normalized.endswith("_flag") or normalized in bool_indicators:
        return "bool"

    if isinstance(value, dict):
        return "Dict[str, Any]"

    if isinstance(value, list):
        return "List[Any]"

    return "Any"


def _parse_schema_fields(schema: str) -> Dict[str, str]:
    header = "class AppState(TypedDict, total=False):"
    start = schema.find(header)
    if start == -1:
        return {}

    body = schema[start + len(header) :]
    lines = body.splitlines()[1:]

    fields: Dict[str, str] = {}
    for line in lines:
        if not line.startswith("    "):
            break
        stripped = line.strip()
        if not stripped:
            continue
        if ":" not in stripped:
            continue
        name, annotation = stripped.split(":", 1)
        fields[name.strip()] = annotation.strip()
    return fields


def _append_fields_to_schema(
    schema: str, new_fields: List[tuple[str, str]]
) -> str:
    if not new_fields:
        return schema

    marker = "\n\ndef update_state_with_circuit_breaker"
    additions = "".join(f"    {name}: {annotation}\n" for name, annotation in new_fields)

    if marker in schema:
        return schema.replace(marker, f"{additions}{marker}", 1)

    if not schema.endswith("\n"):
        schema += "\n"
    return schema + additions


def generate_state_schema(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate the state schema module and store it in ``generated_files``."""

    architecture = state.get("architecture") or {}
    if not isinstance(architecture, dict):
        architecture = {}

    plan_metadata = architecture.get("plan")
    if not isinstance(plan_metadata, dict):
        plan_metadata = architecture
        if not isinstance(plan_metadata, dict):
            plan_metadata = {}

    state_module = generate_adaptive_state_schema(plan_metadata)
    existing_fields = _parse_schema_fields(state_module)
    existing_field_names = set(existing_fields)

    state_flow = architecture.get("state_flow") or {}
    additional_fields: List[tuple[str, str]] = []

    if isinstance(state_flow, dict):
        for key, value in state_flow.items():
            normalized = str(key).strip()
            if not normalized or normalized in existing_field_names:
                continue
            annotation = _infer_collection_type(normalized, value)
            additional_fields.append((normalized, annotation))
            existing_field_names.add(normalized)

    state_module = _append_fields_to_schema(state_module, additional_fields)

    generated = dict(state.get("generated_files") or {})
    generated["state.py"] = state_module

    updated_state = dict(state)
    updated_state["generated_files"] = generated
    return updated_state


def _summarize_fields(fields: Iterable[str]) -> str:
    ordered = sorted(set(fields))
    if not ordered:
        return "no additional fields"
    if len(ordered) == 1:
        return ordered[0]
    return ", ".join(ordered[:-1]) + f" and {ordered[-1]}"


def state_generator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node that populates the generated state schema."""

    try:
        updated_state = generate_state_schema(state)
    except Exception as exc:  # pragma: no cover - defensive logging path
        logger.exception("Failed to generate state schema", exc_info=exc)
        new_state = dict(state)
        messages = list(new_state.get("messages") or [])
        messages.append(
            {
                "role": "assistant",
                "content": f"[state-schema-error] Unable to generate state schema: {exc}",
            }
        )
        new_state["messages"] = messages
        return new_state

    architecture = state.get("architecture", {}) or {}
    state_flow = architecture.get("state_flow") or {}
    generated_fields = list(state_flow.keys()) if isinstance(state_flow, dict) else []

    summary = _summarize_fields(generated_fields)
    message = f"[state-schema] Generated state.py with fields from {summary}."

    messages = list(updated_state.get("messages") or state.get("messages") or [])
    messages.append({"role": "assistant", "content": message})
    updated_state["messages"] = messages
    return updated_state


__all__ = ["generate_state_schema", "state_generator_node"]

