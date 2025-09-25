"""Generate a TypedDict state schema module from architecture metadata."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Literal, TypedDict

from asb.agent.scaffold import generate_enhanced_state_schema

logger = logging.getLogger(__name__)


_BASE_FIELDS: List[tuple[str, str]] = [
    ("messages", "Annotated[List[AnyMessage], add_messages]"),
    ("goal", "str"),
    ("input_text", "str"),
    ("plan", "Dict[str, Any]"),
    ("architecture", "Dict[str, Any]"),
    ("result", "str"),
    ("final_output", "str"),
    ("error", "str"),
    ("errors", "Annotated[List[str], operator.add]"),
    ("scratch", "Annotated[Dict[str, Any], operator.or_]"),
    ("scaffold", "Annotated[Dict[str, Any], operator.or_]"),
    ("self_correction", "Annotated[Dict[str, Any], operator.or_]"),
    ("tot", "Annotated[Dict[str, Any], operator.or_]"),
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
    """Append ``new_fields`` to the ``AppState`` definition within ``schema``."""

    if not new_fields:
        return schema

    lines = schema.splitlines()
    header = "class AppState(TypedDict, total=False):"

    # Fallback: if the header cannot be located, append the fields to the end of
    # the document so the caller can inspect or repair the template manually.
    if header not in lines:
        additions = "\n".join(
            f"    {name}: {annotation}" for name, annotation in new_fields
        )
        if schema and not schema.endswith("\n"):
            schema += "\n"
        return schema + additions + "\n"

    class_index = lines.index(header)

    # Identify the first position after the existing field declarations to
    # insert the new field definitions.
    insert_at = class_index + 1
    for idx in range(class_index + 1, len(lines)):
        line = lines[idx]
        if line.startswith("    ") or not line.strip():
            insert_at = idx + 1
            continue
        break

    insertion_lines = [f"    {name}: {annotation}" for name, annotation in new_fields]
    updated_lines = lines[:insert_at] + insertion_lines + lines[insert_at:]

    result = "\n".join(updated_lines)
    if not result.endswith("\n"):
        result += "\n"
    return result



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

    state_module = generate_enhanced_state_schema(plan_metadata)
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

