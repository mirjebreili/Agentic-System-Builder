from __future__ import annotations
import json, os, re, logging, math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from asb.scaffold.build_nodes import (
    SCAFFOLD_BASE_PATH_KEY,
    SCAFFOLD_ROOT_KEY,
    copy_base_files,
    init_project_structure,
    write_config_files,
    write_graph_module,
    write_node_modules,
    write_state_schema,
)
from asb.scaffold.validate_nodes import validate_state_schema_safety
from asb.utils.fileops import atomic_write, ensure_dir


_BASE_STATE_FIELDS: List[tuple[str, str]] = [
    ("messages", "Annotated[List[AnyMessage], add_messages]"),
    ("goal", "str"),
    ("input_text", "str"),
    ("plan", "Annotated[Dict[str, Any], operator.or_]"),
    ("architecture", "Annotated[Dict[str, Any], operator.or_]"),
    ("result", "str"),
    ("final_output", "str"),
    ("error", "str"),
    ("errors", "Annotated[List[str], operator.add]"),
    ("scratch", "Annotated[Dict[str, Any], operator.or_]"),
    ("scaffold", "Annotated[Dict[str, Any], operator.or_]"),
    ("self_correction", "Annotated[Dict[str, Any], operator.or_]"),
    ("tot", "Annotated[Dict[str, Any], operator.or_]"),
]

def _architecture_requires_self_correction(architecture_plan: Dict[str, Any] | None) -> bool:
    if not isinstance(architecture_plan, dict):
        return False

    def _matches(value: Any) -> bool:
        if isinstance(value, str):
            return "self_correcting_generation" in value.strip().lower()
        return False

    for key in ("workflow_pattern", "pattern", "default_pattern"):
        value = architecture_plan.get(key)
        if _matches(value):
            return True

    workflow = architecture_plan.get("workflow")
    if isinstance(workflow, dict):
        for key in ("pattern", "name", "type"):
            if _matches(workflow.get(key)):
                return True

    patterns = architecture_plan.get("patterns")
    if isinstance(patterns, list):
        for entry in patterns:
            if isinstance(entry, dict):
                for key in ("name", "pattern", "type"):
                    if _matches(entry.get(key)):
                        return True
            elif _matches(entry):
                return True

    for value in architecture_plan.values():
        if isinstance(value, dict):
            if _architecture_requires_self_correction(value):
                return True
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and _architecture_requires_self_correction(item):
                    return True
                if _matches(item):
                    return True

    return False


def generate_enhanced_state_schema(architecture_plan: Dict[str, Any] | None) -> str:
    """Render the default state.py template with safe aggregators."""

    _ = architecture_plan  # architecture metadata may inform extensions later

    lines = [
        "from __future__ import annotations",
        "",
        "from typing import Any, Dict, List, TypedDict, Annotated",
        "import operator",
        "",
        "from langchain_core.messages import AnyMessage",
        "from langgraph.graph import add_messages",
        "",
        "",
        "class AppState(TypedDict, total=False):",
        "    # Messages with proper aggregator",
        "    messages: Annotated[List[AnyMessage], add_messages]",
        "",
        "    # Core inputs - these are usually set once",
        "    goal: str",
        "    input_text: str",
        "",
        "    # Architecture data - merge safely with operator.or_",
        "    plan: Annotated[Dict[str, Any], operator.or_]",
        "    architecture: Annotated[Dict[str, Any], operator.or_]",
        "",
        "    # Execution outputs - last writer wins",
        "    result: str",
        "    final_output: str",
        "",
        "    # Error handling - merge lists with operator.add",
        "    error: str",
        "    errors: Annotated[List[str], operator.add]",
        "",
        "    # Flexible containers - merge with operator.or_",
        "    scratch: Annotated[Dict[str, Any], operator.or_]",
        "    scaffold: Annotated[Dict[str, Any], operator.or_]",
        "    self_correction: Annotated[Dict[str, Any], operator.or_]",
        "    tot: Annotated[Dict[str, Any], operator.or_]",
    ]

    return "\n".join(lines) + "\n"


STATE_TEMPLATE = generate_enhanced_state_schema({})

# repository root
ROOT = Path(__file__).resolve().parents[3]


logger = logging.getLogger(__name__)

LLM_USAGE_CUES: Tuple[str, ...] = (
    "client.get_chat_model",
    "client.get_completion_model",
    "client.invoke(",
    "client.ainvoke(",
)


def _append_scaffold_error(errors: List[str], module_name: str, message: str) -> None:
    entry = f"{module_name}: {message}" if module_name else message
    if entry not in errors:
        errors.append(entry)


def _validate_node_module(
    module_path: Path,
    node_id: str,
    module_name: str,
    user_goal: str,
    errors: List[str],
    *,
    allow_regenerate: bool,
) -> None:
    identifier = module_name or node_id or module_path.stem

    try:
        contents = module_path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - filesystem errors are rare
        _append_scaffold_error(errors, identifier, f"unable to read node file for validation: {exc}")
        return

    if not contents.strip():
        _append_scaffold_error(errors, identifier, "node file is empty after write")
        if allow_regenerate:
            regenerated = generate_generic_node_template(
                node_id or identifier,
                f"Carry out the {node_id or identifier} step.",
                user_goal,
            )
            module_path.write_text(regenerated, encoding="utf-8")
            _validate_node_module(
                module_path,
                node_id,
                module_name,
                user_goal,
                errors,
                allow_regenerate=False,
            )
        return

    required_imports = (
        "from .state import AppState",
        "from ..llm import client",
    )
    missing_imports = [imp for imp in required_imports if imp not in contents]
    if missing_imports:
        _append_scaffold_error(
            errors,
            identifier,
            f"missing required imports: {', '.join(missing_imports)}",
        )
        if allow_regenerate:
            regenerated = generate_generic_node_template(
                node_id or identifier,
                f"Carry out the {node_id or identifier} step.",
                user_goal,
            )
            module_path.write_text(regenerated, encoding="utf-8")
            _validate_node_module(
                module_path,
                node_id,
                module_name,
                user_goal,
                errors,
                allow_regenerate=False,
            )
            return

    if not any(cue in contents for cue in LLM_USAGE_CUES):
        cues_text = ", ".join(LLM_USAGE_CUES)
        _append_scaffold_error(
            errors,
            identifier,
            f"missing LLM client usage (expected one of: {cues_text})",
        )
def _slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+","-", s.strip())
    return s.strip("-").lower() or "project"


def _normalize_generated_key(key: str) -> str:
    """Normalize generated file keys for reliable lookups."""

    normalized = key.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _get_generated_content(
    generated: Dict[str, str],
    *candidates: str,
) -> str | None:
    for candidate in candidates:
        normalized = _normalize_generated_key(candidate)
        if normalized in generated:
            return generated[normalized]
    return None


def _sanitize_identifier(value: str) -> str:
    sanitized = re.sub(r"\W+", "_", value).strip("_")
    return sanitized or "node"


def _candidate_call_hints(node_id: str, module_name: str) -> List[str]:
    """Build an ordered list of attribute names likely to contain the node callable."""

    hints: List[str] = ["run", "execute"]
    variants = {module_name, module_name.lower()}
    normalized_id = re.sub(r"\W+", "_", node_id).strip("_")
    if normalized_id:
        variants.add(normalized_id)
        variants.add(normalized_id.lower())

    for variant in variants:
        if not variant:
            continue
        hints.append(variant)
        hints.append(f"run_{variant}")
        hints.append(f"{variant}_run")

    ordered: List[str] = []
    for hint in hints:
        if hint and hint not in ordered:
            ordered.append(hint)
    return ordered


def _extract_node_id(node: Dict[str, Any]) -> str | None:
    for key in ("id", "node", "name", "label"):
        value = node.get(key)
        if value is None:
            continue
        candidate = str(value).strip()
        if candidate:
            return candidate
    if len(node) == 1:
        only_key = next(iter(node))
        candidate = str(node[only_key]).strip()
        return candidate or None
    return None


def _collect_architecture_nodes(
    architecture: Dict[str, Any]
) -> List[Tuple[str, str, List[str], Dict[str, Any]]]:
    nodes = architecture.get("graph_structure")
    if not nodes:
        return []

    ordered_nodes: List[Tuple[str, str, List[str], Dict[str, Any]]] = []
    seen: set[str] = set()

    iterable: Iterable[Any]
    if isinstance(nodes, dict):
        iterable = nodes.items()
    else:
        iterable = nodes

    for item in iterable:
        node: Dict[str, Any]
        if isinstance(item, tuple) and len(item) == 2 and isinstance(nodes, dict):
            node_id_raw, details = item
            if isinstance(details, dict):
                node = dict(details)
                node.setdefault("id", node_id_raw)
            else:
                node = {"id": node_id_raw, "description": details}
        elif isinstance(item, dict):
            node = item
        else:
            continue

        node_id = _extract_node_id(node)
        if not node_id:
            continue
        if node_id in seen:
            continue

        sanitized = _sanitize_identifier(node_id)
        hints = _candidate_call_hints(node_id, sanitized)
        metadata: Dict[str, Any]
        if isinstance(node, dict):
            metadata = dict(node)
        else:
            metadata = {}
            if node is not None:
                metadata["description"] = str(node)
        metadata.setdefault("id", node_id)
        ordered_nodes.append((node_id, sanitized, hints, metadata))
        seen.add(node_id)

    return ordered_nodes


def _render_node_stub(node_id: str, sanitized: str) -> str:
    return (
        "from typing import Any, Dict\n\n"
        f"def {sanitized}(state: Dict[str, Any]) -> Dict[str, Any]:\n"
        f"    \"\"\"Placeholder implementation for node '{node_id}'.\"\"\"\n"
        "    return state\n"
    )


TOT_UTILS_TEMPLATE = """from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage


MESSAGE_PREFIX_PATTERN = re.compile(r"^(?:human|user|assistant|system)\\s*[:：-]\\s*", re.IGNORECASE)


def _message_to_dict(message: Any) -> Dict[str, Any]:
    if isinstance(message, dict):
        return {k: message[k] for k in ("role", "content") if k in message}

    data: Dict[str, Any] = {}
    if isinstance(message, BaseMessage):
        data["role"] = getattr(message, "type", None) or getattr(message, "role", None)
        data["content"] = getattr(message, "content", None)
        return {k: v for k, v in data.items() if v is not None}

    if hasattr(message, "role"):
        data["role"] = getattr(message, "role")
    if hasattr(message, "content"):
        data["content"] = getattr(message, "content")
    return {k: v for k, v in data.items() if v is not None}


def _coerce_text(value: Any) -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, str):
        text = value
    elif isinstance(value, (list, tuple)):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                candidate = item
            elif isinstance(item, dict):
                candidate = item.get("text") or item.get("content") or item.get("value")
            else:
                candidate = str(item)
            if candidate:
                parts.append(str(candidate))
        text = " ".join(part.strip() for part in parts if part)
    else:
        text = str(value)

    text = text.strip()
    if not text:
        return None

    cleaned = MESSAGE_PREFIX_PATTERN.sub("", text, count=1).strip()
    return cleaned or None


def extract_input_text(state: Dict[str, Any]) -> str:
    \"\"\"Return the most relevant user text while filtering role prefixes.

    Direct input fields are inspected first, followed by the message history.
    Known role prefixes such as "User:" or "Human:" are stripped from the
    resolved string. The sentinel "No input text provided" is returned when
    no meaningful text can be located.
    \"\"\"

    for key in ("input_text", "last_user_input"):
        direct = _coerce_text(state.get(key))
        if direct:
            return direct

    messages = state.get("messages") or []
    fallback: Optional[str] = None
    for message in reversed(list(messages)):
        payload = _message_to_dict(message)
        text = _coerce_text(payload.get("content"))
        if not text:
            continue

        role = (payload.get("role") or "").lower()
        if role in {"user", "human"}:
            return text
        if fallback is None:
            fallback = text

    if fallback:
        return fallback

    return "No input text provided"


def extract_goal(state: Dict[str, Any]) -> str:
    \"\"\"Derive a natural-language goal for the agent based on the state.

    Plan and requirements metadata are consulted before falling back to
    explicit goal fields or the inferred user input. If no goal can be
    determined, the function returns "Complete the requested task".
    \"\"\"

    plan = state.get("plan")
    if isinstance(plan, dict):
        for key in ("goal", "summary", "description"):
            candidate = _coerce_text(plan.get(key))
            if candidate:
                return candidate

    requirements = state.get("requirements")
    if isinstance(requirements, dict):
        for key in ("goal", "summary", "description", "problem"):
            candidate = _coerce_text(requirements.get(key))
            if candidate:
                return candidate

    direct_goal = _coerce_text(state.get("goal"))
    if direct_goal:
        return direct_goal

    inferred = extract_input_text(state)
    if inferred and inferred != "No input text provided":
        return inferred

    return "Complete the requested task"


def parse_approaches(text: str) -> List[str]:
    if not text:
        return []

    entries: List[str] = []
    current: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        bullet = re.match(r"^(?:\\d+[.)-]?|[-*+])\\s*(.+)$", stripped)
        if bullet:
            if current:
                entries.append(" ".join(current).strip())
                current = []
            current.append(bullet.group(1).strip())
            continue

        if current:
            current.append(stripped)
        else:
            current.append(stripped)

    if current:
        entries.append(" ".join(current).strip())

    cleaned = [entry for entry in entries if entry]
    if cleaned:
        return cleaned

    normalized = text.strip()
    return [normalized] if normalized else []


def _extract_score(text: str) -> float:
    match = re.search(r"([0-9]+(?:\\.[0-9]+)?)", text)
    if not match:
        return 0.0

    try:
        value = float(match.group(1))
    except (TypeError, ValueError):
        return 0.0

    if value > 10:
        return value / 100 if value > 100 else value / 10
    return value


def score_thoughts(raw: str, thoughts: List[str]) -> List[Dict[str, Any]]:
    if not raw:
        return []

    segments = parse_approaches(raw)
    evaluations: List[Dict[str, Any]] = []
    for index, thought in enumerate(thoughts, start=1):
        segment = segments[index - 1] if index - 1 < len(segments) else ""
        reasoning = segment.strip() or "No evaluation provided."
        score = _extract_score(segment)
        evaluations.append(
            {
                "index": index,
                "thought": thought,
                "score": score,
                "reasoning": reasoning,
            }
        )

    return evaluations


def get_thoughts(state: Dict[str, Any]) -> List[str]:
    container = state.get("tot") or {}
    thoughts = container.get("thoughts")
    if isinstance(thoughts, list):
        return list(thoughts)

    fallback = state.get("thoughts")
    if isinstance(fallback, list):
        return list(fallback)
    return []


def get_evaluations(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    container = state.get("tot") or {}
    evaluations = container.get("evaluations")
    if isinstance(evaluations, list):
        return [dict(item) for item in evaluations]

    fallback = state.get("evaluations")
    if isinstance(fallback, list):
        return [dict(item) for item in fallback]
    return []


def get_selected_thought(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    container = state.get("tot") or {}
    selected = container.get("selected_thought")
    if isinstance(selected, dict):
        return dict(selected)

    fallback = state.get("selected_thought")
    if isinstance(fallback, dict):
        return dict(fallback)
    return None


def select_top_evaluation(evaluations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_score = float("-inf")
    for evaluation in evaluations:
        candidate = dict(evaluation)
        score = candidate.get("score", 0.0)
        try:
            numeric = float(score)
        except (TypeError, ValueError):
            numeric = 0.0
        candidate["score"] = numeric
        if numeric > best_score:
            best_score = numeric
            best = candidate

    return best


def update_tot_state(state: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    container = dict(state.get("tot") or {})
    container.update(updates)
    return {"tot": container, "error": ""}


def capture_tot_error(state: Dict[str, Any], node_id: str, error: Exception) -> Dict[str, Any]:
    container = dict(state.get("tot") or {})
    errors = list(container.get("errors") or [])
    errors.append({"node": node_id, "message": str(error)})
    container["errors"] = errors
    return {"tot": container, "error": str(error)}


def analyze_task_type(goal: str) -> str:
    normalized = (goal or "").strip().lower()
    if not normalized:
        return "action"

    if any(
        keyword in normalized
        for keyword in ("plan", "outline", "strategy", "roadmap", "design")
    ):
        return "planning"
    if any(
        keyword in normalized
        for keyword in (
            "review",
            "evaluate",
            "critique",
            "check",
            "verify",
            "validate",
            "analysis",
            "assess",
        )
    ):
        return "evaluation"
    if any(
        keyword in normalized
        for keyword in (
            "summarize",
            "summary",
            "report",
            "communicate",
            "brief",
            "final",
            "update",
            "status",
        )
    ):
        return "synthesis"
    if any(
        keyword in normalized
        for keyword in (
            "research",
            "investigate",
            "analyze",
            "learn",
            "understand",
        )
    ):
        return "research"

    return "action"
"""


def generate_generic_node_template(
    node_name: str, node_purpose: str, user_goal: str
) -> str:
    sanitized = _sanitize_identifier(node_name)
    node_label = node_name.strip() or sanitized
    default_goal = (user_goal or "Complete the requested task.").strip() or "Complete the requested task."
    default_purpose = (node_purpose or f"Carry out the {node_label} step.").strip()
    purpose_literal = json.dumps(default_purpose)
    goal_literal = json.dumps(default_goal)
    label_literal = json.dumps(node_label)
    result_key_literal = json.dumps(f"{sanitized}_result")

    role_system_prompts = {
        "planner": (
            "You are an expert workflow planner operating as a LangGraph node."
            " Break complex goals into structured, achievable steps while keeping"
            " downstream execution in mind."
        ),
        "reviewer": (
            "You are a rigorous reviewer inside a LangGraph workflow."
            " Inspect prior work for risks, bugs, and gaps before the project"
            " moves forward."
        ),
        "communicator": (
            "You are a communication specialist summarizing progress for"
            " stakeholders. Translate outcomes into a clear narrative that"
            " highlights impact and next steps."
        ),
        "analyst": (
            "You are an investigative analyst responsible for researching,"
            " interpreting findings, and extracting actionable insights within"
            " the workflow."
        ),
        "executor": (
            "You are an autonomous LangGraph node focused on advancing the task."
            " Execute decisively, update artifacts when relevant, and document"
            " tangible outcomes."
        ),
        "default": (
            "You are a LangGraph node collaborating within a larger agent"
            " workflow. Reason carefully about the context and deliver the"
            " strongest next contribution."
        ),
    }

    role_response_guidelines = {
        "planner": (
            "Return an ordered plan that shows how the broader goal will be"
            " achieved. Call out dependencies or open questions explicitly."
        ),
        "reviewer": (
            "State whether the inspected work passes your checks. Provide a"
            " concise rationale, highlight any issues, and suggest corrective"
            " actions."
        ),
        "communicator": (
            "Summarize the progress so far with emphasis on outcomes, impact,"
            " and recommended next steps for stakeholders."
        ),
        "analyst": (
            "Document the insights you uncovered, the evidence supporting them,"
            " and how they inform future decisions."
        ),
        "executor": (
            "Describe the concrete work you performed, mention any artifacts"
            " touched, and outline immediate follow-up actions if needed."
        ),
        "default": (
            "Report on the progress you achieved and clarify what the team"
            " should do next."
        ),
    }

    def _dict_literal(name: str, mapping: Dict[str, str]) -> str:
        lines = [f"{name} = {{"]
        for key, value in mapping.items():
            lines.append(f"    {json.dumps(key)}: {json.dumps(value)},")
        lines.append("}")
        return "\n".join(lines)

    analyze_lines = [
        "def analyze_task_type(node_name: str, node_purpose: str, state: Dict[str, Any]) -> Dict[str, str]:",
        '    """Infer the node role and communication style from metadata and context."""',
        "    description_parts: List[str] = []",
        "    for candidate in (node_name, node_purpose):",
        "        if candidate:",
        "            description_parts.append(str(candidate))",
        "    plan_goal = None",
        "    plan = state.get(\"plan\") or {}",
        "    if isinstance(plan, dict):",
        "        plan_goal = plan.get(\"goal\")",
        "        if plan_goal:",
        "            description_parts.append(str(plan_goal))",
        "    latest_user = \"\"",
        "    for message in reversed(list(state.get(\"messages\") or [])):",
        "        role = getattr(message, \"type\", None) or getattr(message, \"role\", None)",
        "        content = getattr(message, \"content\", None)",
        "        if isinstance(message, dict):",
        "            role = message.get(\"role\") or role",
        "            if message.get(\"content\") is not None:",
        "                content = message.get(\"content\")",
        "        if not content:",
        "            continue",
        "        text = str(content).strip()",
        "        if not text:",
        "            continue",
        "        normalized_role = str(role or \"\").lower()",
        "        if normalized_role in {\"user\", \"human\"}:",
        "            latest_user = text",
        "            break",
        "        if not latest_user:",
        "            latest_user = text",
        "    if latest_user:",
        "        description_parts.append(latest_user)",
        "    combined = \" \".join(part.lower() for part in description_parts if part).strip()",
        "    role = \"executor\"",
        "    task_type = \"action\"",
        "    tone = \"Drive the task forward and produce tangible progress.\"",
        "    format_hint = \"\"",
        "    response_focus = \"Explain what you accomplished and surface any key artifacts or follow-up actions.\"",
        "    if any(keyword in combined for keyword in (\"plan\", \"outline\", \"strategy\", \"roadmap\", \"design\")):",
        "        role = \"planner\"",
        "        task_type = \"planning\"",
        "        tone = \"Synthesize context into a concrete plan with structured reasoning.\"",
        "        format_hint = \"Return numbered steps that chart the path forward.\"",
        "        response_focus = \"Summarize the plan and highlight dependencies or open questions.\"",
        "    elif any(keyword in combined for keyword in (\"review\", \"evaluate\", \"critique\", \"check\", \"verify\", \"validate\", \"analysis\", \"assess\")):",
        "        role = \"reviewer\"",
        "        task_type = \"evaluation\"",
        "        tone = \"Apply critical thinking to assess quality, risks, and completeness.\"",
        "        format_hint = \"Provide pass/fail judgement with supporting rationale.\"",
        "        response_focus = \"Call out issues discovered and recommended next steps.\"",
        "    elif any(keyword in combined for keyword in (\"summarize\", \"summary\", \"report\", \"communicate\", \"brief\", \"final\", \"update\", \"status\")):",
        "        role = \"communicator\"",
        "        task_type = \"synthesis\"",
        "        tone = \"Condense the progress into a clear summary for stakeholders.\"",
        "        format_hint = \"Use short paragraphs or bullet points focused on outcomes.\"",
        "        response_focus = \"Deliver a polished summary emphasizing results and implications.\"",
        "    elif any(keyword in combined for keyword in (\"research\", \"investigate\", \"analyze\", \"learn\", \"understand\")):",
        "        role = \"analyst\"",
        "        task_type = \"research\"",
        "        tone = \"Reason through the problem and document insights methodically.\"",
        "        format_hint = \"Capture findings and any outstanding unknowns.\"",
        "        response_focus = \"Detail insights gathered and how they inform next actions.\"",
        "    return {",
        "        \"role\": role,",
        "        \"task_type\": task_type,",
        "        \"tone\": tone,",
        "        \"format_hint\": format_hint,",
        "        \"response_focus\": response_focus,",
        "    }",
    ]

    node_lines = [
        f"def {sanitized}(state: AppState) -> AppState:",
        f"    \"\"\"Adaptive implementation for the '{node_label}' node.\"\"\"",
        "",
        "    llm = client.get_chat_model()",
        f"    analysis = analyze_task_type({label_literal}, {purpose_literal}, state)",
        "    role = analysis.get(\"role\", \"executor\")",
        "    task_type = analysis.get(\"task_type\", \"action\")",
        "    tone = analysis.get(\"tone\") or \"\"",
        "    format_hint = analysis.get(\"format_hint\") or \"\"",
        "    response_focus = analysis.get(\"response_focus\") or \"\"",
        f"    raw_goal = state.get(\"goal\") or (state.get(\"plan\") or {{}}).get(\"goal\") or {goal_literal}",
        "    goal = str(raw_goal).strip() or {goal_literal}",
        "    system_prompt = ROLE_SYSTEM_PROMPTS.get(role) or ROLE_SYSTEM_PROMPTS[\"default\"]",
        "    if tone:",
        "        system_prompt = f\"{system_prompt}\\n\\n{tone}\"",
        "    context_data: Dict[str, Any] = {}",
        "    for candidate in (",
        "        state.get(\"context\"),",
        "        (state.get(\"scaffold\") or {}).get(\"context\"),",
        "    ):",
        "        if isinstance(candidate, dict):",
        "            context_data.update(candidate)",
        "",
        "    context_lines: List[str] = []",
        "    if goal:",
        "        context_lines.append(f\"Overall goal: {goal}\")",
        f"    node_purpose = {purpose_literal}",
        "    if node_purpose:",
        "        context_lines.append(f\"Node purpose: {node_purpose}\")",
        "    if context_data:",
        "        context_lines.append(\"Shared context:\")",
        "        for key, value in context_data.items():",
        "            context_lines.append(f\"- {key}: {value}\")",
        "",
        "    messages = list(state.get(\"messages\") or [])",
        "    latest_user = \"\"",
        "    for message in reversed(messages):",
        "        role_name = getattr(message, \"type\", None) or getattr(message, \"role\", None)",
        "        content = getattr(message, \"content\", None)",
        "        if isinstance(message, dict):",
        "            role_name = message.get(\"role\") or role_name",
        "            if message.get(\"content\") is not None:",
        "                content = message.get(\"content\")",
        "        if not content:",
        "            continue",
        "        text = str(content).strip()",
        "        if not text:",
        "            continue",
        "        normalized = str(role_name or \"\").lower()",
        "        if normalized in {\"user\", \"human\"}:",
        "            latest_user = text",
        "            break",
        "        if not latest_user:",
        "            latest_user = text",
        "    if latest_user:",
        "        context_lines.append(f\"Latest user input: {latest_user}\")",
        "",
        "    context_snapshot = list(context_lines)",
        "    context_block = \"\\n\".join(context_lines) or \"No additional context available.\"",
        "    response_guidance = ROLE_RESPONSE_GUIDELINES.get(role) or ROLE_RESPONSE_GUIDELINES[\"default\"]",
        "    user_sections: List[str] = [",
        "        \"Context:\",",
        "        context_block,",
        "        \"\",",
        "        response_guidance,",
        "    ]",
        "    if response_focus:",
        "        user_sections.append(response_focus)",
        "    if format_hint:",
        "        user_sections.append(format_hint)",
        "    user_sections.append(\"Be specific about progress made and reference any artifacts updated.\")",
        "    user_prompt = \"\\n\".join(section for section in user_sections if section)",
        "",
        "    try:",
        "        response = llm.invoke(",
        "            [",
        "                SystemMessage(content=system_prompt),",
        "                HumanMessage(content=user_prompt),",
        "            ]",
        "        )",
        "        content = getattr(response, \"content\", response)",
        "        result_text = content if isinstance(content, str) else str(content)",
        "",
        "        result_snapshot: Dict[str, Any] = {",
        f"            \"node\": {label_literal},",
        "            \"role\": role,",
        "            \"task_type\": task_type,",
        "            \"content\": result_text,",
        "        }",
        "        if context_snapshot:",
        "            result_snapshot[\"context\"] = context_snapshot",
        "        ai_message = AIMessage(",
        "            content=result_text,",
        "            additional_kwargs={\"node\": {label_literal}, \"node_role\": role},",
        "            response_metadata={\"task_type\": task_type},",
        "        )",
        f"        scratch_update = {{{result_key_literal}: result_snapshot}}",
        "        return {",
        "            \"result\": result_text,",
        "            \"messages\": [ai_message],",
        "            \"scratch\": scratch_update,",
        "            \"error\": \"\",",
        "        }",
        "    except Exception as exc:",
        "        return {\"error\": str(exc)}",
    ]

    template_lines = [
        "from __future__ import annotations",
        "",
        "from typing import Any, Dict, List",
        "",
        "from langchain_core.messages import AIMessage, HumanMessage, SystemMessage",
        "",
        "from ..llm import client",
        "from .state import AppState",
        "",
        _dict_literal("ROLE_SYSTEM_PROMPTS", role_system_prompts),
        "",
        _dict_literal("ROLE_RESPONSE_GUIDELINES", role_response_guidelines),
        "",
        *analyze_lines,
        "",
        *node_lines,
    ]

    return "\n".join(template_lines) + "\n"


def _normalize_tot_node_id(node_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", node_id.lower()).strip("_")


def _ensure_tot_utils(agent_dir: Path, generated: Dict[str, str]) -> None:
    utils_path = agent_dir / "utils.py"
    provided = _get_generated_content(
        generated,
        "utils.py",
        "src/agent/utils.py",
        "agent/utils.py",
    )
    if provided is not None:
        ensure_dir(utils_path.parent)
        atomic_write(utils_path, provided)
        return

    if not utils_path.exists():
        ensure_dir(utils_path.parent)
        atomic_write(utils_path, TOT_UTILS_TEMPLATE)


def _render_generate_thoughts(node_id: str, sanitized: str) -> str:
    return f"""from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import client
from .state import AppState
from .utils import (
    capture_tot_error,
    extract_goal,
    extract_input_text,
    parse_approaches,
    update_tot_state,
)


def {sanitized}(state: AppState) -> AppState:
    \"\"\"Generate diverse approaches for tree-of-thought reasoning.\"\"\"

    try:
        goal = extract_goal(state)
        user_input = extract_input_text(state)
        llm = client.get_chat_model()
        system_prompt = (
            "You are an expert reasoner generating candidate approaches for tree-of-thought exploration."
        )
        human_prompt = (
            "Using the goal and the latest user input, propose multiple ways to make progress.\\n"
            f"Goal: {{goal}}\\n"
            f"Latest user input: {{user_input}}\\n"
            "Return 3-5 numbered approaches, each with a concise explanation."
        )
        response = llm.invoke([SystemMessage(system_prompt), HumanMessage(human_prompt)])
        content = getattr(response, "content", response)
        text = content if isinstance(content, str) else str(content)
        thoughts: List[str] = parse_approaches(text)
        if not thoughts:
            raise ValueError("Unable to parse any thoughts from the model output.")
        return update_tot_state(
            state,
            {{
                "thoughts": thoughts,
                "raw_generate_response": text,
            }},
        )
    except Exception as exc:
        return capture_tot_error(state, "{sanitized}", exc)
"""


def _render_evaluate_thoughts(node_id: str, sanitized: str) -> str:
    return f"""from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import client
from .state import AppState
from .utils import (
    capture_tot_error,
    extract_goal,
    extract_input_text,
    get_thoughts,
    score_thoughts,
    update_tot_state,
)


def {sanitized}(state: AppState) -> AppState:
    \"\"\"Score generated thoughts to inform downstream selection.\"\"\"

    thoughts = get_thoughts(state)
    if not thoughts:
        return capture_tot_error(state, "{sanitized}", ValueError("No thoughts available for evaluation."))

    try:
        goal = extract_goal(state)
        user_input = extract_input_text(state)
        llm = client.get_chat_model()
        numbered = "\\n".join(f"{{idx}}. {{thought}}" for idx, thought in enumerate(thoughts, start=1))
        system_prompt = "You are an analytical critic who scores solution approaches."
        human_prompt = (
            "Assess each approach for how well it helps reach the goal. Provide a score between 0 and 1 and a short rationale for each item.\\n"
            f"Goal: {{goal}}\\n"
            f"Latest user input: {{user_input}}\\n"
            "Thoughts:\\n"
            f"{{numbered}}"
        )
        response = llm.invoke([SystemMessage(system_prompt), HumanMessage(human_prompt)])
        content = getattr(response, "content", response)
        text = content if isinstance(content, str) else str(content)
        evaluations = score_thoughts(text, thoughts)
        if not evaluations:
            raise ValueError("Unable to parse evaluations for the generated thoughts.")
        return update_tot_state(
            state,
            {{
                "evaluations": evaluations,
                "raw_evaluation_response": text,
            }},
        )
    except Exception as exc:
        return capture_tot_error(state, "{sanitized}", exc)
"""


def generate_nodes_from_architecture(
    architecture_plan: Dict[str, Any], user_goal: str
) -> Dict[str, str]:
    """Generate module templates for structured architecture nodes."""

    if not isinstance(architecture_plan, dict):
        return {}

    nodes = architecture_plan.get("nodes")
    if not isinstance(nodes, list):
        return {}

    generated_modules: Dict[str, str] = {}
    for entry in nodes:
        if not isinstance(entry, dict):
            continue

        raw_name = entry.get("name") or entry.get("id") or entry.get("label")
        if raw_name is None:
            continue

        node_name = str(raw_name).strip()
        if not node_name:
            continue

        purpose_value = entry.get("purpose") or entry.get("description") or ""
        if isinstance(purpose_value, (list, tuple)):
            purpose_text = " ".join(
                str(item).strip() for item in purpose_value if item
            ).strip()
        else:
            purpose_text = str(purpose_value).strip() if purpose_value else ""

        module_name = _sanitize_identifier(node_name)
        filename = f"{module_name}.py"
        generated_modules[filename] = generate_generic_node_template(
            node_name,
            purpose_text,
            user_goal,
        )

    return generated_modules


def _render_select_best_thought(node_id: str, sanitized: str) -> str:
    return f"""from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import client
from .state import AppState
from .utils import (
    capture_tot_error,
    extract_goal,
    extract_input_text,
    get_evaluations,
    get_thoughts,
    select_top_evaluation,
    update_tot_state,
)


def {sanitized}(state: AppState) -> AppState:
    \"\"\"Choose the most promising thought for finalization.\"\"\"

    try:
        evaluations = get_evaluations(state)
        selection_source = "evaluation"
        if evaluations:
            selected = select_top_evaluation(evaluations)
            if not selected:
                raise ValueError("Evaluations were present but no selection could be made.")
        else:
            thoughts = get_thoughts(state)
            if not thoughts:
                raise ValueError("No thoughts available to select from.")

            selection_source = "fallback"
            selected = {{
                "index": 1,
                "thought": thoughts[0],
                "score": 0.0,
                "reasoning": "Defaulted to the first thought due to missing evaluations.",
            }}

        goal = extract_goal(state)
        user_input = extract_input_text(state)
        if isinstance(selected, dict):
            thought_text = str(selected.get("thought") or "")
        else:
            thought_text = str(selected)

        llm = client.get_chat_model()
        system_prompt = (
            "You are a reasoning assistant summarizing why a tree-of-thought option was selected."
        )
        human_prompt = (
            "Provide a concise justification for the chosen thought.\\n"
            f"Goal: {{goal}}\\n"
            f"Latest user input: {{user_input}}\\n"
            f"Selection source: {{source}}\\n"
            f"Chosen thought: {{thought}}\\n"
            "Respond with 1-2 sentences highlighting why this thought is the best next step."
        ).format(
            goal=goal,
            user_input=user_input,
            source=selection_source,
            thought=thought_text,
        )
        response = llm.invoke([SystemMessage(system_prompt), HumanMessage(human_prompt)])
        content = getattr(response, "content", response)
        summary_text = content if isinstance(content, str) else str(content)

        payload = {{
            "selected_thought": selected,
            "selection_summary": summary_text,
            "selection_source": selection_source,
        }}
        return update_tot_state(state, payload)
    except Exception as exc:
        return capture_tot_error(state, "{sanitized}", exc)
"""


def _render_final_answer(node_id: str, sanitized: str) -> str:
    return f"""from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import client
from .state import AppState
from .utils import (
    capture_tot_error,
    extract_goal,
    extract_input_text,
    get_selected_thought,
    update_tot_state,
)


def {sanitized}(state: AppState) -> AppState:
    \"\"\"Produce the final assistant response using the selected thought.\"\"\"

    selected = get_selected_thought(state)
    if not selected:
        return capture_tot_error(state, "{sanitized}", ValueError("No selected thought is available for the final answer."))

    try:
        goal = extract_goal(state)
        user_input = extract_input_text(state)
        llm = client.get_chat_model()
        chosen = str(selected.get("thought") or "")
        rationale = str(selected.get("reasoning") or "")
        system_prompt = (
            "You are a helpful assistant finalizing the response based on the chosen reasoning path."
        )
        human_prompt = (
            "Compose the final assistant response using the selected approach while staying aligned with the goal and latest user input.\\n"
            f"Goal: {{goal}}\\n"
            f"Latest user input: {{user_input}}\\n"
            f"Chosen approach: {{chosen}}\\n"
            f"Rationale: {{rationale}}\\n"
            "Deliver a clear and actionable answer."
        )
        response = llm.invoke([SystemMessage(system_prompt), HumanMessage(human_prompt)])
        content = getattr(response, "content", response)
        answer = content if isinstance(content, str) else str(content)
        base_updates = update_tot_state(
            state,
            {{
                "final_answer": answer,
                "raw_final_response": answer,
                "selected_thought": selected,
            }},
        )
        message = {{"role": "assistant", "content": answer}}
        combined: Dict[str, Any] = dict(base_updates)
        combined.update(
            {
                "messages": [message],
                "final_output": answer,
                "result": answer,
            }
        )
        return combined
    except Exception as exc:
        return capture_tot_error(state, "{sanitized}", exc)
"""


_TOT_RENDERERS = {
    "generate_thoughts": _render_generate_thoughts,
    "evaluate_thoughts": _render_evaluate_thoughts,
    "select_best_thought": _render_select_best_thought,
    "final_answer": _render_final_answer,
}


SELF_CORRECTING_UTILS_TEMPLATE = """from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage


DEFAULT_MAX_ATTEMPTS = 3

MESSAGE_PREFIX_PATTERN = re.compile(r"^(?:human|user|assistant|system)\\s*[:：-]\\s*", re.IGNORECASE)
JSON_BLOCK_PATTERN = re.compile(r"\\{[\\s\\S]*\\}")


def _message_to_dict(message: Any) -> Dict[str, Any]:
    if isinstance(message, dict):
        return {k: message[k] for k in ("role", "content") if k in message}

    data: Dict[str, Any] = {}
    if isinstance(message, BaseMessage):
        data["role"] = getattr(message, "type", None) or getattr(message, "role", None)
        data["content"] = getattr(message, "content", None)
        return {k: v for k, v in data.items() if v is not None}

    if hasattr(message, "role"):
        data["role"] = getattr(message, "role")
    if hasattr(message, "content"):
        data["content"] = getattr(message, "content")
    return {k: v for k, v in data.items() if v is not None}


def _coerce_text(value: Any) -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, str):
        candidate = value
    elif isinstance(value, (list, tuple)):
        parts: List[str] = []
        for item in value:
            coerced = _coerce_text(item)
            if coerced:
                parts.append(coerced)
        candidate = " ".join(parts)
    elif isinstance(value, dict):
        primary = value.get("text") or value.get("content") or value.get("value")
        if primary is not None:
            return _coerce_text(primary)
        return None
    else:
        candidate = str(value)

    cleaned = MESSAGE_PREFIX_PATTERN.sub("", candidate or "", count=1).strip()
    return cleaned or None


def extract_input_text(state: Dict[str, Any]) -> str:
    for key in ("input_text", "last_user_input"):
        direct = _coerce_text(state.get(key))
        if direct:
            return direct

    messages = state.get("messages") or []
    fallback: Optional[str] = None
    for message in reversed(list(messages)):
        payload = _message_to_dict(message)
        text = _coerce_text(payload.get("content"))
        if not text:
            continue
        role = (payload.get("role") or "").lower()
        if role in {"user", "human"}:
            return text
        if fallback is None:
            fallback = text

    return fallback or "No recent user input available."


def extract_goal(state: Dict[str, Any]) -> str:
    plan = state.get("plan")
    if isinstance(plan, dict):
        for key in ("goal", "summary", "description"):
            candidate = _coerce_text(plan.get(key))
            if candidate:
                return candidate

    requirements = state.get("requirements")
    if isinstance(requirements, dict):
        for key in ("goal", "summary", "description", "objective", "problem"):
            candidate = _coerce_text(requirements.get(key))
            if candidate:
                return candidate

    direct = _coerce_text(state.get("goal"))
    if direct:
        return direct

    inferred = extract_input_text(state)
    return inferred or "Complete the requested task."


def _collect_requirement_sections(source: Dict[str, Any], keys: List[str]) -> List[str]:
    sections: List[str] = []
    for key in keys:
        if key not in source:
            continue
        candidate = _coerce_text(source.get(key))
        if not candidate:
            continue
        label = key.replace("_", " ").title()
        sections.append(f"{label}: {candidate}")
    return sections


def extract_requirements(state: Dict[str, Any]) -> str:
    requirements = state.get("requirements")
    sections: List[str] = []
    if isinstance(requirements, dict):
        sections.extend(
            _collect_requirement_sections(
                requirements,
                [
                    "summary",
                    "description",
                    "acceptance_criteria",
                    "requirements",
                    "constraints",
                    "notes",
                ],
            )
        )

    plan = state.get("plan")
    if isinstance(plan, dict):
        sections.extend(
            _collect_requirement_sections(
                plan,
                ["plan", "steps", "milestones", "context"],
            )
        )

    if sections:
        return "\\n".join(dict.fromkeys(section for section in sections if section))

    return "No explicit requirements were provided."


def get_self_correction_payload(state: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(state, dict):
        direct = state.get("self_correction")
        if isinstance(direct, dict):
            return dict(direct)

        scaffold = state.get("scaffold")
        if isinstance(scaffold, dict):
            payload = scaffold.get("self_correction")
            if isinstance(payload, dict):
                return dict(payload)
    return {}


def apply_self_correction_payload(state: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized_payload = dict(payload)
    scaffold = dict((state.get("scaffold") or {}))
    scaffold["self_correction"] = normalized_payload
    return {
        "self_correction": normalized_payload,
        "scaffold": scaffold,
        "error": "",
    }


def ensure_self_correction_payload(state: Dict[str, Any]) -> Dict[str, Any]:
    payload = get_self_correction_payload(state)
    payload.setdefault("history", [])
    payload.setdefault("attempt", int(payload.get("attempt") or 0))
    payload.setdefault("needs_revision", bool(payload.get("needs_revision", False)))
    payload.setdefault("awaiting_validation", bool(payload.get("awaiting_validation", False)))
    payload.setdefault("validation", payload.get("validation") or {})
    payload.setdefault("max_attempts", payload.get("max_attempts") or DEFAULT_MAX_ATTEMPTS)
    return payload


def register_candidate(
    state: Dict[str, Any],
    candidate: str,
    *,
    node: str,
    reasoning: Optional[str] = None,
    attempt: Optional[int] = None,
) -> Dict[str, Any]:
    payload = ensure_self_correction_payload(state)
    attempt_index = attempt if isinstance(attempt, int) and attempt > 0 else int(payload.get("attempt") or 0) + 1
    payload["attempt"] = attempt_index
    payload["latest_candidate"] = candidate
    payload["needs_revision"] = False
    payload["awaiting_validation"] = True
    history = list(payload.get("history") or [])
    entry: Dict[str, Any] = {
        "attempt": attempt_index,
        "node": node,
    }
    snippet = _coerce_text(candidate)
    if snippet:
        entry["summary"] = snippet[:200]
    if reasoning:
        entry["reasoning"] = reasoning
    history.append(entry)
    payload["history"] = history
    payload["validation"] = {}
    return apply_self_correction_payload(state, payload)


def record_validation_result(
    state: Dict[str, Any],
    success: bool,
    feedback: str,
    *,
    node: str,
    raw: Optional[str] = None,
) -> Dict[str, Any]:
    payload = ensure_self_correction_payload(state)
    attempt_index = int(payload.get("attempt") or 0)
    feedback_text = _coerce_text(feedback) or "No feedback provided."
    validation: Dict[str, Any] = {
        "success": bool(success),
        "feedback": feedback_text,
        "attempt": attempt_index,
        "node": node,
    }
    if raw is not None:
        validation["raw"] = raw
    payload["validation"] = validation
    payload["needs_revision"] = not bool(success)
    payload["awaiting_validation"] = False

    history = list(payload.get("history") or [])
    for entry in reversed(history):
        if entry.get("attempt") == attempt_index:
            entry["verdict"] = "pass" if success else "fail"
            if feedback_text:
                entry["feedback"] = feedback_text
            validators = list(entry.get("validators") or [])
            if node not in validators:
                validators.append(node)
            entry["validators"] = validators
            break
    payload["history"] = history
    return apply_self_correction_payload(state, payload)


def summarize_history(payload: Dict[str, Any]) -> str:
    history = payload.get("history")
    if not isinstance(history, list) or not history:
        return "No previous attempts have been recorded."

    lines: List[str] = []
    for entry in history:
        attempt = entry.get("attempt")
        lines.append(f"Attempt {attempt}:")
        verdict = entry.get("verdict")
        if verdict:
            lines.append(f"- Result: {verdict}")
        feedback = entry.get("feedback")
        if feedback:
            lines.append(f"- Feedback: {feedback}")
        reasoning = entry.get("reasoning")
        if reasoning:
            lines.append(f"- Reasoning: {reasoning}")
    return "\\n".join(lines)


def parse_validation_response(text: str) -> Dict[str, Any]:
    if not text:
        return {"success": False, "feedback": "Validator returned an empty response."}

    snippet = text
    match = JSON_BLOCK_PATTERN.search(text)
    if match:
        snippet = match.group(0)

    data: Optional[Dict[str, Any]] = None
    try:
        parsed = json.loads(snippet)
        if isinstance(parsed, dict):
            data = parsed
    except json.JSONDecodeError:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                data = parsed
        except json.JSONDecodeError:
            data = None

    if data is not None:
        success_value = data.get("success")
        if success_value is None:
            success_value = data.get("verdict") or data.get("result")
        if isinstance(success_value, str):
            normalized = success_value.strip().lower()
            success = normalized in {"pass", "passed", "true", "success", "succeeded", "approve", "approved", "ok"}
        else:
            success = bool(success_value)

        feedback_value = data.get("feedback") or data.get("reason") or data.get("notes") or data.get("critique")
        if isinstance(feedback_value, (list, tuple)):
            feedback = " ".join(str(item) for item in feedback_value if item)
        elif feedback_value is None:
            feedback = text.strip()
        else:
            feedback = str(feedback_value)

        return {
            "success": success,
            "feedback": feedback.strip() or text.strip(),
            "data": data,
        }

    lowered = text.strip().lower()
    if lowered.startswith("pass") or lowered.startswith("success"):
        return {"success": True, "feedback": text.strip() or "Validator indicated success."}
    if lowered.startswith("fail") or lowered.startswith("error") or lowered.startswith("issue"):
        return {"success": False, "feedback": text.strip() or "Validator reported issues."}

    return {"success": False, "feedback": text.strip() or "Validator response could not be parsed."}


def should_retry(payload: Dict[str, Any], limit: Optional[int] = None) -> bool:
    if not isinstance(payload, dict):
        return False
    if not payload.get("needs_revision"):
        return False

    if isinstance(limit, int) and limit > 0:
        max_attempts = limit
    else:
        max_attempts = int(payload.get("max_attempts") or DEFAULT_MAX_ATTEMPTS)

    attempt = int(payload.get("attempt") or 0)
    return attempt < max_attempts
"""


_SELF_CORRECTING_ROLE_ALIASES: Dict[str, set[str]] = {
    "generate": {"generate", "generator", "draft", "proposal", "produce", "create", "initial"},
    "validate": {
        "validate",
        "validation",
        "verify",
        "critique",
        "review",
        "check",
        "assess",
        "score",
        "analyze",
    },
    "correct": {"correct", "correction", "fix", "repair", "revise", "refine", "improve", "update"},
}


def _normalize_self_correcting_node_id(node_id: str) -> Optional[str]:
    if not node_id:
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", node_id.lower()).strip("_")
    if not normalized:
        return None

    for role, aliases in _SELF_CORRECTING_ROLE_ALIASES.items():
        for alias in aliases:
            if normalized == alias or normalized.startswith(f"{alias}_") or normalized.endswith(f"_{alias}"):
                return role
    return None


def _self_correcting_purpose(metadata: Dict[str, Any], fallback: str) -> str:
    if not isinstance(metadata, dict):
        return fallback
    for key in (
        "purpose",
        "description",
        "summary",
        "objective",
        "details",
        "task",
    ):
        value = metadata.get(key)
        text = ""
        if isinstance(value, (list, tuple)):
            text = " ".join(str(item).strip() for item in value if item)
        elif value is not None:
            text = str(value).strip()
        if text:
            return text
    return fallback


def _render_self_correcting_generate(
    node_id: str,
    sanitized: str,
    purpose: str,
    user_goal: str,
) -> str:
    purpose_literal = json.dumps(purpose or f"Generate the solution for {node_id}.")
    goal_literal = json.dumps(user_goal or "Complete the requested task.")
    label_literal = json.dumps(node_id)
    template = """from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import client
from .state import AppState
from .self_correction import (
    DEFAULT_MAX_ATTEMPTS,
    apply_self_correction_payload,
    ensure_self_correction_payload,
    extract_goal,
    extract_input_text,
    extract_requirements,
    register_candidate,
    summarize_history,
)


def {sanitized}(state: AppState) -> AppState:
    \"\"\"Generate or regenerate the working solution candidate.\"\"\"

    payload = ensure_self_correction_payload(state)
    payload.setdefault("max_attempts", DEFAULT_MAX_ATTEMPTS)
    if payload.get("attempt", 0) > 0 and not payload.get("needs_revision"):
        return state

    working_state = apply_self_correction_payload(state, payload)

    goal = extract_goal(working_state) or {goal_literal}
    user_input = extract_input_text(working_state)
    requirements = extract_requirements(working_state)
    feedback_summary = summarize_history(payload)
    directive = {purpose_literal}

    llm = client.get_chat_model()
    system_prompt = (
        "You are a focused creator generating high-quality solutions that satisfy detailed requirements."
    )
    human_prompt = (
        "You are operating inside a self-correcting workflow. Produce the best complete solution you can.\n"
        "Overall goal: {{goal}}\n"
        "Latest user input: {{user_input}}\n"
        "Documented requirements:\n"
        "{{requirements}}\n"
        "Prior attempts and feedback:\n"
        "{{feedback_summary}}\n"
        "Specific directive: {{directive}}\n"
        "Respond with the full proposed solution ready for validation."
    ).format(
        goal=goal,
        user_input=user_input,
        requirements=requirements,
        feedback_summary=feedback_summary,
        directive=directive,
    )

    response = llm.invoke(
        [
            SystemMessage(system_prompt),
            HumanMessage(human_prompt),
        ]
    )
    content = getattr(response, "content", response)
    candidate = content if isinstance(content, str) else str(content)

    updated_state = register_candidate(working_state, candidate, node={label_literal})
    return updated_state
"""
    return template.format(
        sanitized=sanitized,
        goal_literal=goal_literal,
        purpose_literal=purpose_literal,
        label_literal=label_literal,
    )

def _render_self_correcting_validate(
    node_id: str,
    sanitized: str,
    purpose: str,
) -> str:
    label_literal = json.dumps(node_id)
    purpose_literal = json.dumps(purpose or f"Review the candidate produced during {node_id}.")
    template = """from __future__ import annotations

import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import client
from .state import AppState
from .self_correction import (
    apply_self_correction_payload,
    ensure_self_correction_payload,
    extract_goal,
    extract_input_text,
    extract_requirements,
    parse_validation_response,
    record_validation_result,
    summarize_history,
)


def {sanitized}(state: AppState) -> AppState:
    \"\"\"Evaluate the latest solution candidate and record validation feedback.\"\"\"

    payload = ensure_self_correction_payload(state)
    candidate = payload.get("latest_candidate")
    if not candidate:
        return state

    goal = extract_goal(state)
    user_input = extract_input_text(state)
    requirements = extract_requirements(state)
    history = summarize_history(payload)
    directive = {purpose_literal}

    llm = client.get_chat_model()
    system_prompt = (
        "You are a meticulous reviewer ensuring the solution fully satisfies the stated requirements."
    )
    human_prompt = (
        "Critically assess the proposed solution. Respond with a JSON object containing the keys "success" and "feedback".\n"
        "Goal: {{goal}}\n"
        "Latest user input: {{user_input}}\n"
        "Requirements to satisfy:\n"
        "{{requirements}}\n"
        "Attempt history:\n"
        "{{history}}\n"
        "Candidate under review:\n"
        "{{candidate}}\n"
        "Validation directive: {{directive}}"
    ).format(
        goal=goal,
        user_input=user_input,
        requirements=requirements,
        history=history,
        candidate=candidate,
        directive=directive,
    )

    response = llm.invoke(
        [
            SystemMessage(system_prompt),
            HumanMessage(human_prompt),
        ]
    )
    content = getattr(response, "content", response)
    text = content if isinstance(content, str) else str(content)
    parsed = parse_validation_response(text)
    success = bool(parsed.get("success"))
    feedback = parsed.get("feedback") or text

    updated = record_validation_result(state, success, feedback, node={label_literal}, raw=text)
    payload = ensure_self_correction_payload(updated)
    payload.setdefault("validation", {{}})
    payload["validation"].setdefault("parsed", {{}})
    if isinstance(parsed.get("data"), dict):
        payload["validation"]["parsed"] = parsed["data"]
    payload["validation"]["raw"] = text
    return apply_self_correction_payload(updated, payload)
"""
    return template.format(
        sanitized=sanitized,
        label_literal=label_literal,
        purpose_literal=purpose_literal,
    )

def _render_self_correcting_correct(
    node_id: str,
    sanitized: str,
    purpose: str,
) -> str:
    label_literal = json.dumps(node_id)
    purpose_literal = json.dumps(purpose or f"Improve the solution when {node_id} reports issues.")
    template = """from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import client
from .state import AppState
from .self_correction import (
    DEFAULT_MAX_ATTEMPTS,
    apply_self_correction_payload,
    ensure_self_correction_payload,
    extract_goal,
    extract_input_text,
    extract_requirements,
    register_candidate,
    summarize_history,
)


def {sanitized}(state: AppState) -> AppState:
    \"\"\"Refine the solution using validator feedback until it passes or retries are exhausted.\"\"\"

    payload = ensure_self_correction_payload(state)
    if not payload.get("needs_revision"):
        return state

    attempt = int(payload.get("attempt") or 0)
    limit = int(payload.get("max_attempts") or DEFAULT_MAX_ATTEMPTS)
    if attempt >= limit:
        return state

    goal = extract_goal(state)
    user_input = extract_input_text(state)
    requirements = extract_requirements(state)
    history = summarize_history(payload)
    feedback = (payload.get("validation") or {{}}).get("feedback", "")
    directive = {purpose_literal}

    llm = client.get_chat_model()
    system_prompt = (
        "You are a senior engineer correcting flaws identified during validation while preserving strengths of the proposal."
    )
    human_prompt = (
        "Use the validator feedback to improve the solution. Provide the fully revised solution in your reply.\n"
        "Goal: {{goal}}\n"
        "Latest user input: {{user_input}}\n"
        "Requirements to respect:\n"
        "{{requirements}}\n"
        "Validator feedback to address:\n"
        "{{feedback}}\n"
        "Attempt history:\n"
        "{{history}}\n"
        "Revision directive: {{directive}}"
    ).format(
        goal=goal,
        user_input=user_input,
        requirements=requirements,
        feedback=feedback or "No feedback provided.",
        history=history,
        directive=directive,
    )

    response = llm.invoke(
        [
            SystemMessage(system_prompt),
            HumanMessage(human_prompt),
        ]
    )
    content = getattr(response, "content", response)
    candidate = content if isinstance(content, str) else str(content)

    updated = register_candidate(state, candidate, node={label_literal})
    payload = ensure_self_correction_payload(updated)
    payload["max_attempts"] = limit
    return apply_self_correction_payload(updated, payload)
"""
    return template.format(
        sanitized=sanitized,
        label_literal=label_literal,
        purpose_literal=purpose_literal,
    )

def generate_self_correcting_nodes(
    node_specs: List[Tuple[str, str, List[str], Dict[str, Any]]],
    user_goal: str,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    role_map: Dict[str, Tuple[str, str, Dict[str, Any]]] = {}
    for node_id, module_name, _, metadata in node_specs:
        role = _normalize_self_correcting_node_id(node_id)
        if role and role not in role_map:
            role_map[role] = (node_id, module_name, metadata)

    required_roles = {"generate", "validate", "correct"}
    if not required_roles.issubset(role_map):
        return {}, {}

    generator_id, generator_module, generator_meta = role_map["generate"]
    validator_id, validator_module, validator_meta = role_map["validate"]
    corrector_id, corrector_module, corrector_meta = role_map["correct"]

    modules: Dict[str, str] = {
        generator_module: _render_self_correcting_generate(
            generator_id,
            generator_module,
            _self_correcting_purpose(generator_meta, f"Generate the solution in {generator_id} step."),
            user_goal,
        ),
        validator_module: _render_self_correcting_validate(
            validator_id,
            validator_module,
            _self_correcting_purpose(validator_meta, f"Validate the output created by {generator_id}."),
        ),
        corrector_module: _render_self_correcting_correct(
            corrector_id,
            corrector_module,
            _self_correcting_purpose(corrector_meta, f"Improve the solution when {validator_id} reports issues."),
        ),
    }

    helpers: Dict[str, str] = {
        "self_correction.py": SELF_CORRECTING_UTILS_TEMPLATE,
    }

    return modules, helpers


def _write_node_modules(
    base: Path,
    node_specs: List[Tuple[str, str, List[str], Dict[str, Any]]],
    generated: Dict[str, str],
    missing_files: List[str],
    user_goal: str,
    errors: List[str],
) -> None:
    agent_dir = base / "src" / "agent"
    ensure_dir(agent_dir)

    self_correcting_modules, self_correcting_helpers = generate_self_correcting_nodes(
        node_specs,
        user_goal,
    )

    for helper_name, helper_source in self_correcting_helpers.items():
        helper_path = agent_dir / helper_name
        provided_helper = _get_generated_content(
            generated,
            helper_name,
            f"src/agent/{helper_name}",
            f"agent/{helper_name}",
        )
        if provided_helper is not None and not provided_helper.strip():
            provided_helper = None

        if provided_helper is not None:
            ensure_dir(helper_path.parent)
            atomic_write(helper_path, provided_helper)
            continue

        if not helper_path.exists():
            ensure_dir(helper_path.parent)
            atomic_write(helper_path, helper_source)
            helper_entry = str(helper_path)
            if helper_entry not in missing_files:
                missing_files.append(helper_entry)

    tot_utils_prepared = False

    for node_id, module_name, _, node_details in node_specs:
        filename = f"{module_name}.py"
        source = _get_generated_content(
            generated,
            filename,
            f"src/agent/{filename}",
            f"agent/{filename}",
        )

        if source is not None and not source.strip():
            source = None

        destination = agent_dir / filename
        generated_locally = False

        if source is None:
            generated_locally = True
            if module_name in self_correcting_modules:
                source = self_correcting_modules[module_name]
            else:
                normalized = _normalize_tot_node_id(node_id)
                renderer = _TOT_RENDERERS.get(normalized)
                if renderer is not None:
                    if not tot_utils_prepared:
                        _ensure_tot_utils(agent_dir, generated)
                        tot_utils_prepared = True
                    source = renderer(node_id, module_name)
                else:
                    purpose = ""
                    node_info = node_details if isinstance(node_details, dict) else {}
                    if node_info:
                        for key in (
                            "purpose",
                            "description",
                            "objective",
                            "summary",
                            "details",
                            "task",
                            "responsibility",
                            "prompt",
                        ):
                            value = node_info.get(key)
                            if not value:
                                continue
                            if isinstance(value, (list, tuple)):
                                candidate = " ".join(
                                    str(item).strip() for item in value if item
                                ).strip()
                            else:
                                candidate = str(value).strip()
                            if candidate:
                                purpose = candidate
                                break
                    source = generate_generic_node_template(
                        node_id,
                        purpose or f"Carry out the {node_id} step.",
                        user_goal,
                    )
            missing_entry = str(destination)
            if missing_entry not in missing_files:
                missing_files.append(missing_entry)

        ensure_dir(destination.parent)
        atomic_write(destination, source)
        _validate_node_module(
            destination,
            node_id,
            module_name,
            user_goal,
            errors,
            allow_regenerate=generated_locally,
        )


def _detect_node_callable(module_path: Path, hints: Iterable[str]) -> str:
    """Best-effort detection of the callable implementing a node."""

    try:
        source = module_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return module_path.stem

    for hint in hints:
        if not hint:
            continue
        pattern = rf"^\s*(?:async\s+)?def\s+{re.escape(hint)}\s*\("
        if re.search(pattern, source, re.MULTILINE):
            return hint

    match = re.search(r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", source, re.MULTILINE)
    if match:
        return match.group(1)

    return module_path.stem


def _render_executor_module(
    node_definitions: List[Dict[str, str]]
) -> str:
    """Render the executor template with populated node implementations."""

    imports: List[str] = []
    implementations: List[str] = []

    for entry in node_definitions:
        node_id = entry.get("id", "")
        module_name = entry.get("module", "")
        target = entry.get("callable") or module_name
        alias = entry.get("alias") or module_name or target

        if not node_id or not module_name or not target:
            continue

        if alias != target:
            imports.append(f"from .{module_name} import {target} as {alias}")
        else:
            imports.append(f"from .{module_name} import {target}")
        implementations.append(f"('{node_id}', {alias})")

    import_section = "\n".join(imports)
    impl_section = ",\n    ".join(implementations)
    impl_block = f"    {impl_section}\n" if impl_section else ""

    template = f"""# generated
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Tuple

{import_section}

NODE_IMPLEMENTATIONS: List[Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]]] = [
{impl_block}]


def iter_node_callables() -> Iterable[Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]]]:
    for spec in NODE_IMPLEMENTATIONS:
        yield spec


def execute(state: Dict[str, Any]) -> Dict[str, Any]:
    current = state
    for _, node_callable in NODE_IMPLEMENTATIONS:
        current = node_callable(current)
    return current
"""

    return template


def _format_edge_endpoint(name: str) -> str:
    upper = name.upper()
    if upper == "START":
        return "START"
    if upper == "END":
        return "END"
    return repr(name)


def _extract_edge_endpoint(value: Any) -> str | None:
    if value is None:
        return None
    candidate = str(value).strip()
    if not candidate:
        return None
    return candidate


def generate_dynamic_workflow_module(architecture_plan: Dict[str, Any]) -> str:
    """Render a graph module that synthesizes workflows at runtime."""

    node_definitions: List[Dict[str, str]] = []
    raw_defs = architecture_plan.get("_node_definitions")
    if isinstance(raw_defs, list):
        for entry in raw_defs:
            if not isinstance(entry, dict):
                continue
            node_id = entry.get("id")
            module = entry.get("module")
            target = entry.get("callable")
            alias = entry.get("alias") or (module if isinstance(module, str) else None)
            if not (isinstance(node_id, str) and isinstance(module, str) and isinstance(target, str)):
                continue
            node_definitions.append(
                {
                    "id": node_id,
                    "module": module,
                    "callable": target,
                    "alias": alias or module,
                }
            )

    if not node_definitions:
        nodes = architecture_plan.get("nodes")
        if isinstance(nodes, list):
            for entry in nodes:
                if not isinstance(entry, dict):
                    continue
                raw_name = entry.get("name") or entry.get("id") or entry.get("label")
                if raw_name is None:
                    continue
                node_id = str(raw_name).strip()
                if not node_id:
                    continue
                module = _sanitize_identifier(node_id)
                node_definitions.append(
                    {
                        "id": node_id,
                        "module": module,
                        "callable": module,
                        "alias": module,
                    }
                )

    architecture_payload: Dict[str, Any] = {}
    if isinstance(architecture_plan, dict):
        architecture_payload = dict(architecture_plan)

    def _normalize_json_value(value: Any) -> Any:
        """Recursively sanitize values so they can be serialized to JSON."""

        if isinstance(value, dict):
            return {str(key): _normalize_json_value(val) for key, val in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [_normalize_json_value(item) for item in value]

        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return value

        if isinstance(value, (int, str, bool)) or value is None:
            return value

        if isinstance(value, (bytes, bytearray)):
            try:
                return value.decode("utf-8")
            except Exception:
                return value.decode("utf-8", errors="ignore")

        return str(value)

    raw_payload = architecture_payload or {}

    try:
        architecture_json = json.dumps(
            raw_payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        normalized_payload = _normalize_json_value(raw_payload)
        try:
            architecture_json = json.dumps(
                normalized_payload,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            logger.warning("Unable to serialize architecture plan: %s", exc)
            architecture_json = "{}"

    try:
        json.loads(architecture_json)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid architecture JSON detected: %s", exc)
        architecture_json = "{}"

    if not architecture_json.strip() or architecture_json.strip() == "{}":
        architecture_json = '{"error": "No architecture plan provided"}'

    lines: List[str] = [
        "# generated",
        "from __future__ import annotations",
        "",
        "import json",
        "import logging",
        "import os",
        "import re",
        "import sqlite3",
        "import sys  # required for runtime argv detection",
        "from typing import Any, Callable, Dict, Iterable, List, Tuple",
        "",
        "from langgraph.checkpoint.sqlite import SqliteSaver",
        "from langgraph.graph import END, START, StateGraph",
        "",
        "from .state import AppState",
    ]

    if node_definitions:
        lines.append("")
        for entry in node_definitions:
            module = entry["module"]
            target = entry["callable"]
            alias = entry.get("alias") or module
            if target == alias:
                lines.append(f"from .{module} import {target}")
            else:
                lines.append(f"from .{module} import {target} as {alias}")

    lines.extend(
        [
            "",
            "ARCHITECTURE_STATE = json.loads(",
            "    \"\"\"",
        ]
    )

    for architecture_line in architecture_json.splitlines():
        lines.append(f"    {architecture_line}")

    lines.extend(
        [
            "    \"\"\"",
            ")",
            "",
            "logger = logging.getLogger(__name__)",
            "",
            "",
        ]
    )

    lines.extend(
        [
            "def _normalize_self_correcting_node_id(node_id: str) -> str | None:",
            "    if not isinstance(node_id, str):",
            "        return None",
            "    normalized = re.sub(r'[^a-z0-9]+', '_', node_id.lower()).strip('_')",
            "    if not normalized:",
            "        return None",
            "    mappings = {",
            "        'generate': {'generate', 'generator', 'draft', 'proposal', 'produce', 'create', 'initial'},",
            "        'validate': {'validate', 'validation', 'verify', 'critique', 'review', 'assess', 'check', 'score'},",
            "        'correct': {'correct', 'correction', 'fix', 'repair', 'revise', 'refine', 'improve', 'update'},",
            "    }",
            "    for role, aliases in mappings.items():",
            "        for alias in aliases:",
            "            if normalized == alias or normalized.startswith(f'{alias}_') or normalized.endswith(f'_{alias}'):",
            "                return role",
            "    return None",
            "",
            "",
            "def _get_self_correction_payload(state: Dict[str, Any]) -> Dict[str, Any]:",
            "    if not isinstance(state, dict):",
            "        return {}",
            "    payload = state.get('self_correction')",
            "    if isinstance(payload, dict):",
            "        return payload",
            "    scaffold = state.get('scaffold')",
            "    if isinstance(scaffold, dict):",
            "        nested = scaffold.get('self_correction')",
            "        if isinstance(nested, dict):",
            "            return nested",
            "    return {}",
            "",
            "",
            "def _should_retry_self_correction(state: Dict[str, Any], limit: int | None = None) -> bool:",
            "    payload = _get_self_correction_payload(state)",
            "    if not payload.get('needs_revision'):",
            "        return False",
            "    attempt = int(payload.get('attempt') or 0)",
            "    try:",
            "        max_attempts = int(limit or payload.get('max_attempts') or 3)",
            "    except Exception:",
            "        max_attempts = 3",
            "    return attempt < max_attempts",
            "",
            "",
            "def generate_self_correcting_nodes(",
            "    pattern: Dict[str, Any],",
            "    registered_nodes: Iterable[Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]]],",
            ") -> List[Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]]]:",
            "    ordered = list(registered_nodes)",
            "    lookup: Dict[str, Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]]] = {}",
            "    for node_id, handler in ordered:",
            "        role = _normalize_self_correcting_node_id(node_id)",
            "        if role and role not in lookup:",
            "            lookup[role] = (node_id, handler)",
            "    if not {'generate', 'validate', 'correct'}.issubset(lookup):",
            "        return ordered",
            "    generate_id, generate_handler = lookup['generate']",
            "    validate_id, validate_handler = lookup['validate']",
            "    correct_id, correct_handler = lookup['correct']",
            "    config = pattern if isinstance(pattern, dict) else {}",
            "    if isinstance(pattern, dict):",
            "        arch = pattern.get('architecture')",
            "        if isinstance(arch, dict):",
            "            config = arch",
            "    max_attempts = None",
            "    if isinstance(config, dict):",
            "        for key in ('max_attempts', 'retry_limit', 'max_retries'):",
            "            value = config.get(key)",
            "            if isinstance(value, int) and value > 0:",
            "                max_attempts = value",
            "                break",
            "    def generator_node(state: Dict[str, Any]) -> Dict[str, Any]:",
            "        payload = _get_self_correction_payload(state)",
            "        if payload.get('attempt') and not payload.get('needs_revision'):",
            "            return state",
            "        return generate_handler(state)",
            "",
            "    def validator_node(state: Dict[str, Any]) -> Dict[str, Any]:",
            "        return validate_handler(state)",
            "",
            "    def corrector_node(state: Dict[str, Any]) -> Dict[str, Any]:",
            "        current = state",
            "        safety = 0",
            "        while _should_retry_self_correction(current, max_attempts):",
            "            current = correct_handler(current)",
            "            current = validate_handler(current)",
            "            safety += 1",
            "            if safety > 10:",
            "                break",
            "        return current",
            "",
            "    return [",
            "        (generate_id, generator_node),",
            "        (validate_id, validator_node),",
            "        (correct_id, corrector_node),",
            "    ]",
            "",
            "",
            "NODE_IMPLEMENTATIONS: List[Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]]] = [",
        ]
    )

    for entry in node_definitions:
        alias = entry.get("alias") or entry["module"]
        lines.append(f"    ({entry['id']!r}, {alias}),")

    lines.extend(
        [
            "]",
            "",
            "",
            "def iter_node_callables() -> Iterable[Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]]]:",
            "    for spec in NODE_IMPLEMENTATIONS:",
            "        yield spec",
            "",
            "",
            "def execute(state: Dict[str, Any]) -> Dict[str, Any]:",
            "    current = state",
            "    for _, node_callable in NODE_IMPLEMENTATIONS:",
            "        current = node_callable(current)",
            "    return current",
            "",
            "",
            "def running_on_langgraph_api() -> bool:",
            "    langgraph_env = os.environ.get('LANGGRAPH_ENV', '').lower()",
            "    if langgraph_env in {'cloud', 'api', 'hosted'}:",
            "        return True",
            "    if os.environ.get('LANGGRAPH_API_URL') or os.environ.get('LANGGRAPH_CLOUD'):",
            "        return True",
            "    argv = [arg.lower() for arg in sys.argv[1:]]",
            "    return '--langgraph-api' in argv or ('langgraph' in argv and 'api' in argv)",
            "",
            "",
            "def analyze_workflow_pattern(state: Dict[str, Any] | None = None) -> Dict[str, Any]:",
            "    architecture: Dict[str, Any] = {}",
            "    if isinstance(state, dict):",
            "        candidate = state.get('architecture')",
            "        if isinstance(candidate, dict):",
            "            architecture = dict(candidate)",
            "        else:",
            "            plan_candidate = state.get('plan')",
            "            if isinstance(plan_candidate, dict):",
            "                nested = plan_candidate.get('architecture')",
            "                if isinstance(nested, dict):",
            "                    architecture = dict(nested)",
            "    if not architecture:",
            "        if isinstance(ARCHITECTURE_STATE, dict):",
            "            architecture = dict(ARCHITECTURE_STATE)",
            "        else:",
            "            architecture = {}",
            "    pattern_name = architecture.get('workflow_pattern') or architecture.get('pattern') or architecture.get('default_pattern')",
            "    if isinstance(pattern_name, str):",
            "        pattern_name = pattern_name.strip() or None",
            "    raw_sequence = architecture.get('workflow_sequence')",
            "    ordered_nodes: List[str] = []",
            "    if isinstance(raw_sequence, list):",
            "        for entry in raw_sequence:",
            "            if isinstance(entry, str):",
            "                candidate = entry.strip()",
            "            elif isinstance(entry, dict):",
            "                candidate = str(entry.get('id') or entry.get('name') or entry.get('label') or '').strip()",
            "            else:",
            "                candidate = str(entry).strip()",
            "            if candidate:",
            "                ordered_nodes.append(candidate)",
            "    if not ordered_nodes:",
            "        raw_nodes = architecture.get('nodes') or architecture.get('graph_structure') or []",
            "        if isinstance(raw_nodes, list):",
            "            for entry in raw_nodes:",
            "                if isinstance(entry, dict):",
            "                    candidate = entry.get('id') or entry.get('name') or entry.get('label')",
            "                    candidate = str(candidate).strip() if candidate is not None else ''",
            "                else:",
            "                    candidate = str(entry).strip()",
            "                if candidate:",
            "                    ordered_nodes.append(candidate)",
            "    raw_edges = architecture.get('edges')",
            "    edges: List[Tuple[str, str]] = []",
            "    if isinstance(raw_edges, list):",
            "        for entry in raw_edges:",
            "            if isinstance(entry, dict):",
            "                source = entry.get('from') or entry.get('source') or entry.get('start') or entry.get('src')",
            "                target = entry.get('to') or entry.get('target') or entry.get('end') or entry.get('dst')",
            "            elif isinstance(entry, (list, tuple)) and len(entry) == 2:",
            "                source, target = entry",
            "            else:",
            "                continue",
            "            source_name = str(source).strip() if source is not None else ''",
            "            target_name = str(target).strip() if target is not None else ''",
            "            if source_name and target_name:",
            "                edges.append((source_name, target_name))",
            "    return {",
            "        'name': pattern_name or 'sequential',",
            "        'ordered_nodes': ordered_nodes,",
            "        'edges': edges,",
            "        'architecture': architecture,",
            "    }",
            "",
            "",
            "def generate_adaptive_nodes(",
            "    pattern: Dict[str, Any],",
            "    registered_nodes: Iterable[Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]]],",
            ") -> List[Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]]]:",
            "    requested = pattern.get('ordered_nodes')",
            "    requested_ids: List[str] = []",
            "    if isinstance(requested, list):",
            "        for entry in requested:",
            "            if isinstance(entry, str):",
            "                candidate = entry.strip()",
            "            elif isinstance(entry, dict):",
            "                candidate = str(entry.get('id') or entry.get('name') or entry.get('label') or '').strip()",
            "            else:",
            "                candidate = str(entry).strip()",
            "            if candidate:",
            "                requested_ids.append(candidate)",
            "    available = list(registered_nodes)",
            "    if not available:",
            "        return []",
            "    pattern_name = ''",
            "    if isinstance(pattern, dict):",
            "        raw_name = pattern.get('name') or pattern.get('pattern')",
            "        if isinstance(raw_name, str):",
            "            pattern_name = raw_name.strip().lower()",
            "    if pattern_name == 'self_correcting_generation':",
            "        adaptive = generate_self_correcting_nodes(pattern, available)",
            "        if adaptive:",
            "            return adaptive",
            "    selected: List[Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]]] = []",
            "    for node_id in requested_ids:",
            "        for candidate_id, handler in available:",
            "            if candidate_id == node_id:",
            "                selected.append((candidate_id, handler))",
            "                break",
            "    if selected:",
            "        return selected",
            "    return available",
            "",
            "",
            "def _fallback_node(state: Dict[str, Any]) -> Dict[str, Any]:",
            "    scaffold = dict((state.get('scaffold') or {}))",
            "    scaffold.setdefault('ok', True)",
            "    return {'scaffold': scaffold}",
            "",
            "",
            "def create_dynamic_graph(",
            "    pattern: Dict[str, Any],",
            "    node_sequence: List[Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]]],",
            "    *,",
            "    path: str | None = None,",
            "):",
            "    graph = StateGraph(AppState)",
            "    ordered_ids = [node_id for node_id, _ in node_sequence]",
            "    for node_id, node_callable in node_sequence:",
            "        graph.add_node(node_id, node_callable)",
            "    normalized_edges = pattern.get('edges') or []",
            "    raw_edges: List[Tuple[str, str]] = []",
            "    if isinstance(normalized_edges, list):",
            "        for entry in normalized_edges:",
            "            if isinstance(entry, (tuple, list)) and len(entry) == 2:",
            "                source, target = entry",
            "            else:",
            "                try:",
            "                    source, target = entry",
            "                except Exception:",
            "                    continue",
            "            source_name = str(source).strip()",
            "            target_name = str(target).strip()",
            "            if source_name and target_name:",
            "                raw_edges.append((source_name, target_name))",
            "    start_hint: str | None = None",
            "    sequential_edges: List[Tuple[str, str]] = []",
            "    for source_name, target_name in raw_edges:",
            "        source_upper = source_name.upper()",
            "        target_upper = target_name.upper()",
            "        if source_upper == 'START':",
            "            if start_hint is None and target_name in ordered_ids:",
            "                start_hint = target_name",
            "            continue",
            "        if target_upper == 'END':",
            "            continue",
            "        if source_name in ordered_ids and target_name in ordered_ids:",
            "            sequential_edges.append((source_name, target_name))",
            "    def _build_chain(order: List[str], connections: List[Tuple[str, str]], start_node: str | None) -> List[str]:",
            "        next_map: Dict[str, str] = {}",
            "        indegree: Dict[str, int] = {node: 0 for node in order}",
            "        for source_name, target_name in connections:",
            "            if source_name not in indegree or target_name not in indegree:",
            "                continue",
            "            if source_name in next_map:",
            "                continue",
            "            next_map[source_name] = target_name",
            "            indegree[target_name] += 1",
            "        chain: List[str] = []",
            "        visited: set[str] = set()",
            "        def _append_path(node: str) -> None:",
            "            while node and node not in visited:",
            "                chain.append(node)",
            "                visited.add(node)",
            "                node = next_map.get(node)",
            "        if start_node and start_node in indegree:",
            "            _append_path(start_node)",
            "        for node in order:",
            "            if indegree.get(node, 0) == 0 and node not in visited:",
            "                _append_path(node)",
            "        for node in order:",
            "            if node not in visited:",
            "                _append_path(node)",
            "        return chain",
            "    chain = _build_chain(ordered_ids, sequential_edges, start_hint)",
            "    if not chain and ordered_ids:",
            "        chain = list(ordered_ids)",
            "    if not chain:",
            "        graph.add_node('fallback', _fallback_node)",
            "        graph.add_edge(START, 'fallback')",
            "        graph.add_edge('fallback', END)",
            "    else:",
            "        first = chain[0]",
            "        graph.add_edge(START, first)",
            "        previous = first",
            "        for node_id in chain[1:]:",
            "            graph.add_edge(previous, node_id)",
            "            previous = node_id",
            "        graph.add_edge(previous, END)",
            "    if running_on_langgraph_api():",
            "        logger.info('LangGraph API runtime detected; compiling without a checkpointer.')",
            "        return graph.compile(checkpointer=None)",
            "    checkpointer = None",
            "    if path:",
            "        directory = os.path.dirname(path)",
            "        if directory:",
            "            os.makedirs(directory, exist_ok=True)",
            "        connection = sqlite3.connect(path, check_same_thread=False)",
            "        checkpointer = SqliteSaver(connection)",
            "    return graph.compile(checkpointer=checkpointer)",
            "",
            "",
            "def generate_dynamic_workflow(",
            "    state: Dict[str, Any] | None = None,",
            "    *,",
            "    path: str | None = None,",
            "):",
            "    pattern = analyze_workflow_pattern(state)",
            "    node_sequence = generate_adaptive_nodes(pattern, NODE_IMPLEMENTATIONS)",
            "    return create_dynamic_graph(pattern, node_sequence, path=path)",
            "",
            "",
            "def _make_graph(path: str | None = None):",
            "    return generate_dynamic_workflow(path=path)",
            "",
            "",
            "graph = _make_graph()",
        ]
    )

    return "\n".join(lines) + "\n"


def generate_dynamic_graph(architecture_plan: Dict[str, Any]) -> str:
    return generate_dynamic_workflow_module(architecture_plan)


def _extract_workflow_metadata(
    architecture_plan: Dict[str, Any] | None,
) -> tuple[str, List[str], List[tuple[str, str]]]:
    """Collect normalized workflow metadata for the generated tests."""

    architecture: Dict[str, Any] = {}
    if isinstance(architecture_plan, dict):
        architecture = dict(architecture_plan)

    pattern_name: Optional[str] = None
    for key in ("workflow_pattern", "pattern", "default_pattern"):
        value = architecture.get(key)
        if isinstance(value, str):
            candidate = value.strip()
            if candidate:
                pattern_name = candidate
                break

    ordered_nodes: List[str] = []
    raw_sequence = architecture.get("workflow_sequence")
    if isinstance(raw_sequence, list):
        for entry in raw_sequence:
            if isinstance(entry, str):
                candidate = entry.strip()
            elif isinstance(entry, dict):
                candidate_raw = (
                    entry.get("id")
                    or entry.get("name")
                    or entry.get("label")
                )
                candidate = str(candidate_raw).strip() if candidate_raw is not None else ""
            else:
                candidate = str(entry).strip()
            if candidate:
                ordered_nodes.append(candidate)

    if not ordered_nodes:
        raw_nodes = architecture.get("nodes") or architecture.get("graph_structure")
        if isinstance(raw_nodes, list):
            for entry in raw_nodes:
                if isinstance(entry, dict):
                    candidate_raw = (
                        entry.get("id")
                        or entry.get("name")
                        or entry.get("label")
                    )
                    candidate = str(candidate_raw).strip() if candidate_raw is not None else ""
                else:
                    candidate = str(entry).strip()
                if candidate:
                    ordered_nodes.append(candidate)

    edges: List[tuple[str, str]] = []
    raw_edges = architecture.get("edges")
    if isinstance(raw_edges, list):
        for entry in raw_edges:
            source: Any
            target: Any
            if isinstance(entry, dict):
                source = (
                    entry.get("from")
                    or entry.get("source")
                    or entry.get("start")
                    or entry.get("src")
                )
                target = (
                    entry.get("to")
                    or entry.get("target")
                    or entry.get("end")
                    or entry.get("dst")
                )
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                source, target = entry
            else:
                try:
                    source, target = entry
                except Exception:
                    continue
            source_name = str(source).strip() if source is not None else ""
            target_name = str(target).strip() if target is not None else ""
            if source_name and target_name:
                edges.append((source_name, target_name))

    normalized_pattern = pattern_name or "sequential"
    return normalized_pattern, ordered_nodes, edges


def _format_literal_list(items: Iterable[Any]) -> str:
    materialized = list(items)
    if not materialized:
        return "[]"
    rendered = [f"    {repr(item)}," for item in materialized]
    return "[\n" + "\n".join(rendered) + "\n]"


def generate_comprehensive_tests(architecture_plan: Dict[str, Any] | None) -> str:
    """Render a comprehensive pytest suite aligned with the workflow pattern."""

    pattern_name, ordered_nodes, edges = _extract_workflow_metadata(architecture_plan)
    include_self_correction = _architecture_requires_self_correction(architecture_plan)

    lines: List[str] = [
        "\"\"\"Comprehensive tests for the generated agent workflow.\"\"\"",
        "",
        "from __future__ import annotations",
        "",
        "from pathlib import Path",
        "",
        "import pytest",
        "",
        "import src.agent.executor as executor_module",
        "import src.agent.graph as graph_module",
        "from src.agent.state import AppState",
        "",
        f"EXPECTED_PATTERN_NAME = {pattern_name!r}",
        f"EXPECTED_NODE_SEQUENCE = {_format_literal_list(ordered_nodes)}",
        f"EXPECTED_EDGE_SET = {_format_literal_list(edges)}",
        f"EXPECTS_SELF_CORRECTION = {include_self_correction}",
        "",
        "",
        "@pytest.fixture(autouse=True)",
        "def stub_node_implementations(monkeypatch):",
        "    original = list(getattr(executor_module, \"NODE_IMPLEMENTATIONS\", []))",
        "    stubs = []",
        "    if original:",
        "        for node_id, _ in original:",
        "            def _make_stub(identifier):",
        "                def _stub(state):",
        "                    updated = dict(state or {})",
        "                    messages = list(updated.get(\"messages\") or [])",
        "                    messages.append({\"role\": \"assistant\", \"content\": f\"Stub executed {identifier}\"})",
        "                    updated[\"messages\"] = messages",
        "                    debug = dict(updated.get(\"debug\") or {})",
        "                    visited = list(debug.get(\"visited_nodes\") or [])",
        "                    visited.append(identifier)",
        "                    debug[\"visited_nodes\"] = visited",
        "                    updated[\"debug\"] = debug",
        "                    scaffold = dict(updated.get(\"scaffold\") or {})",
        "                    scaffold.setdefault(\"ok\", True)",
        "                    updated[\"scaffold\"] = scaffold",
        "                    if EXPECTS_SELF_CORRECTION:",
        "                        payload = dict(updated.get(\"self_correction\") or {})",
        "                        payload.setdefault(\"history\", [])",
        "                        payload.setdefault(\"attempt\", int(payload.get(\"attempt\") or 0))",
        "                        updated[\"self_correction\"] = payload",
        "                    return updated",
        "                return _stub",
        "            stubs.append((node_id, _make_stub(node_id)))",
        "    else:",
        "        def _fallback(state):",
        "            updated = dict(state or {})",
        "            messages = list(updated.get(\"messages\") or [])",
        "            messages.append({\"role\": \"assistant\", \"content\": \"Stub executed fallback\"})",
        "            updated[\"messages\"] = messages",
        "            debug = dict(updated.get(\"debug\") or {})",
        "            visited = list(debug.get(\"visited_nodes\") or [])",
        "            visited.append(\"fallback\")",
        "            debug[\"visited_nodes\"] = visited",
        "            updated[\"debug\"] = debug",
        "            scaffold = dict(updated.get(\"scaffold\") or {})",
        "            scaffold.setdefault(\"ok\", True)",
        "            updated[\"scaffold\"] = scaffold",
        "            if EXPECTS_SELF_CORRECTION:",
        "                payload = dict(updated.get(\"self_correction\") or {})",
        "                payload.setdefault(\"history\", [])",
        "                payload.setdefault(\"attempt\", 0)",
        "                updated[\"self_correction\"] = payload",
        "            return updated",
        "        stubs.append((\"fallback\", _fallback))",
        "    monkeypatch.setattr(executor_module, \"NODE_IMPLEMENTATIONS\", stubs, raising=False)",
        "    monkeypatch.setattr(graph_module, \"NODE_IMPLEMENTATIONS\", stubs, raising=False)",
        "    graph_module.graph = graph_module._make_graph()",
        "    return stubs",
        "",
        "",
        "def test_graph_module_exports_graph():",
        "    assert hasattr(graph_module, \"graph\")",
        "    assert graph_module.graph is not None",
        "",
        "",
        "def test_workflow_pattern_alignment(stub_node_implementations):",
        "    pattern = graph_module.analyze_workflow_pattern()",
        "    assert isinstance(pattern, dict)",
        "    assert pattern.get(\"name\") == EXPECTED_PATTERN_NAME",
        "    actual_nodes = [node_id for node_id, _ in stub_node_implementations]",
        "    if EXPECTED_NODE_SEQUENCE:",
        "        for node_id in EXPECTED_NODE_SEQUENCE:",
        "            assert node_id in actual_nodes",
        "            assert node_id in pattern.get(\"ordered_nodes\", [])",
        "    if EXPECTED_EDGE_SET:",
        "        expected_edges = {tuple(edge) for edge in EXPECTED_EDGE_SET}",
        "        actual_edges = {tuple(edge) for edge in pattern.get(\"edges\", [])}",
        "        assert expected_edges.issubset(actual_edges)",
        "",
        "",
        "def test_end_to_end_execution(tmp_path: Path, stub_node_implementations):",
        "    checkpoint = tmp_path / \"checkpoints\" / \"graph.db\"",
        "    compiled = graph_module._make_graph(str(checkpoint))",
        "    result = compiled.invoke({\"messages\": []}, config={\"configurable\": {\"thread_id\": \"test-e2e\"}})",
        "    assert isinstance(result, dict)",
        "    assert \"messages\" in result",
        "    assert isinstance(result[\"messages\"], list)",
        "    debug = result.get(\"debug\", {})",
        "    visited = debug.get(\"visited_nodes\", [])",
        "    if stub_node_implementations:",
        "        assert visited, \"Expected stub nodes to record execution order\"",
        "    if EXPECTS_SELF_CORRECTION:",
        "        assert \"self_correction\" in result",
        "    else:",
        "        assert \"self_correction\" not in result",
        "",
        "",
        "def test_generate_dynamic_workflow_handles_missing_state(stub_node_implementations):",
        "    compiled = graph_module.generate_dynamic_workflow(state=None, path=None)",
        "    result = compiled.invoke({\"messages\": []}, config={\"configurable\": {\"thread_id\": \"test-dynamic\"}})",
        "    assert isinstance(result, dict)",
        "    assert \"messages\" in result",
        "",
        "",
        "def test_state_schema_self_correction_contract():",
        "    annotations = getattr(AppState, \"__annotations__\", {})",
        "    if EXPECTS_SELF_CORRECTION:",
        "        assert \"self_correction\" in annotations",
        "    else:",
        "        assert \"self_correction\" not in annotations",
        "",
        "",
        "if __name__ == \"__main__\":",
        "    raise SystemExit(pytest.main([__file__]))",
    ]

    return "\n".join(lines) + "\n"


def _render_graph_module() -> str:
    lines: List[str] = [
        "# generated",
        "from __future__ import annotations",
        "",
        "import logging",
        "import os",
        "import sqlite3",
        "import sys  # required for runtime argv detection",
        "from typing import Any, Dict",
        "",
        "from langgraph.checkpoint.sqlite import SqliteSaver",
        "from langgraph.graph import StateGraph, START, END",
        "",
        "from .state import AppState",
        "from .executor import NODE_IMPLEMENTATIONS",
        "",
        "logger = logging.getLogger(__name__)",
        "",
        "",
        "def running_on_langgraph_api() -> bool:",
        "    langgraph_env = os.environ.get('LANGGRAPH_ENV', '').lower()",
        "    if langgraph_env in {'cloud', 'api', 'hosted'}:",
        "        return True",
        "    if os.environ.get('LANGGRAPH_API_URL') or os.environ.get('LANGGRAPH_CLOUD'):",
        "        return True",
        "    argv = [arg.lower() for arg in sys.argv[1:]]",
        "    return '--langgraph-api' in argv or ('langgraph' in argv and 'api' in argv)",
        "",
        "",
        "def _make_graph(path: str | None = None):",
        "    g = StateGraph(AppState)",
        "",
        "    if NODE_IMPLEMENTATIONS:",
        "        previous = START",
        "        for node_id, node_callable in NODE_IMPLEMENTATIONS:",
        "            g.add_node(node_id, node_callable)",
        "            g.add_edge(previous, node_id)",
        "            previous = node_id",
        "        g.add_edge(previous, END)",
        "    else:",
        "        g.add_edge(START, END)",
        "",
        "    if running_on_langgraph_api():",
        "        logger.info('LangGraph API runtime detected; compiling without a checkpointer.')",
        "        return g.compile(checkpointer=None)",
        "",
        "    checkpointer = None",
        "    if path:",
        "        dir_path = os.path.dirname(path)",
        "        if dir_path:",
        "            os.makedirs(dir_path, exist_ok=True)",
        "        connection = sqlite3.connect(path, check_same_thread=False)",
        "        checkpointer = SqliteSaver(connection)",
        "",
        "    return g.compile(checkpointer=checkpointer)",
        "",
        "",
        "graph = _make_graph()",
    ]

    return "\n".join(lines) + "\n"

def scaffold_project(state: Dict[str, Any]) -> Dict[str, Any]:
    plan_goal = (state.get("plan") or {}).get("goal")
    goal = plan_goal or "agent_project"
    name = _slug(str(goal))[:40]
    user_goal = ""
    for candidate in (
        plan_goal,
        state.get("goal"),
        state.get("input_text"),
        state.get("last_user_input"),
        goal,
    ):
        if not candidate:
            continue
        candidate_text = str(candidate).strip()
        if candidate_text:
            user_goal = candidate_text
            break
    if not user_goal:
        user_goal = "Complete the requested task."
    base = ROOT / "projects" / name
    missing_files: List[str] = []
    scaffold_errors: List[str] = []

    state[SCAFFOLD_BASE_PATH_KEY] = base
    state[SCAFFOLD_ROOT_KEY] = ROOT
    try:
        init_project_structure(state)
        for package_dir in ("src", "src/agent", "src/llm", "src/config"):
            init_path = base / package_dir / "__init__.py"
            init_path.parent.mkdir(parents=True, exist_ok=True)
            if not init_path.exists():
                init_path.write_text("", encoding="utf-8")

        write_config_files(state)
        missing_files = copy_base_files(state)
    finally:
        state.pop(SCAFFOLD_BASE_PATH_KEY, None)
        state.pop(SCAFFOLD_ROOT_KEY, None)

    normalized_generated = {
        _normalize_generated_key(key): value
        for key, value in (state.get("generated_files") or {}).items()
    }

    architecture = state.get("architecture") or {}
    architecture_plan = architecture.get("plan")
    if not isinstance(architecture_plan, dict):
        architecture_plan = architecture if isinstance(architecture, dict) else {}

    node_definitions: List[Dict[str, str]] = []
    node_specs: List[Tuple[str, str, List[str], Dict[str, Any]]] = []

    state["_scaffold_generated"] = normalized_generated
    state["_scaffold_architecture_plan"] = architecture_plan
    state["_scaffold_user_goal"] = user_goal
    state["_scaffold_missing_files"] = missing_files
    state["_scaffold_errors"] = scaffold_errors
    state[SCAFFOLD_BASE_PATH_KEY] = base

    try:
        write_state_schema(state)

        schema_issues = validate_state_schema_safety(base)
        if schema_issues:
            scaffold_errors.extend(schema_issues)

        node_definitions, node_specs = write_node_modules(state)
        state["_scaffold_node_definitions"] = node_definitions
        write_graph_module(state)
    finally:
        state.pop("_scaffold_node_definitions", None)
        state.pop("_scaffold_architecture_plan", None)
        state.pop("_scaffold_generated", None)
        state.pop("_scaffold_user_goal", None)
        state.pop("_scaffold_missing_files", None)
        state.pop("_scaffold_errors", None)
        state.pop(SCAFFOLD_BASE_PATH_KEY, None)

    # imports are already correct in copied files

    # prompts (simple copies from parent)
    (base / "prompts" / "plan_system.jinja").write_text(
        "You are a planner. Return only JSON for the 3-node plan/do/finish.", encoding="utf-8")
    (base / "prompts" / "plan_user.jinja").write_text("User goal:\n{{ user_goal }}\n", encoding="utf-8")

    # child planner.py
    (base / "src/agent/planner.py").write_text("""# generated
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from .prompts_util import find_prompts_dir


class PlanNode(BaseModel):
    id: str
    type: str
    prompt: Optional[str] = None
    tool: Optional[str] = None


class PlanEdge(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    if_: Optional[str] = Field(None, alias="if")


class Plan(BaseModel):
    goal: str
    nodes: List[PlanNode]
    edges: List[PlanEdge]
    confidence: Optional[float] = None


PROMPTS_DIR = find_prompts_dir()
SYSTEM_PROMPT = (PROMPTS_DIR / "plan_system.jinja").read_text(encoding="utf-8")
USER_TEMPLATE = (PROMPTS_DIR / "plan_user.jinja").read_text(encoding="utf-8")


def _render_user_prompt(goal: str) -> str:
    return USER_TEMPLATE.replace("{{ user_goal }}", goal)


def _resolve_goal(state: Dict[str, Any]) -> str:
    messages = state.get("messages") or []
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if isinstance(message, dict):
            content = message.get("content") or content
        if content:
            return str(content)
    input_text = state.get("input_text")
    if input_text:
        return str(input_text)
    return "Plan a simple workflow."


def plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        goal = _resolve_goal(state)
        plan = Plan(
            goal=goal,
            nodes=[
                PlanNode(id="plan", type="llm", prompt=SYSTEM_PROMPT.strip() or None),
                PlanNode(id="do", type="llm", prompt=_render_user_prompt(goal)),
                PlanNode(id="finish", type="llm", prompt="Summarize the outcome and next steps."),
            ],
            edges=[
                PlanEdge(from_="plan", to="do"),
                PlanEdge(from_="do", to="do", if_="more_steps"),
                PlanEdge(from_="do", to="finish", if_="steps_done"),
            ],
        ).model_dump(by_alias=True)

        current_step = {"more_steps": True, "steps_done": False}
        return {
            "plan": plan,
            "messages": [AIMessage(content=f"Generated plan for goal: {goal}")],
            "scratch": {"workflow_current_step": current_step},
        }
    except Exception as exc:
        return {"error": str(exc)}
""", encoding="utf-8")

    # tests
    tests_dir = base / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    test_suite_path = tests_dir / "test_workflow.py"
    test_suite_path.write_text(
        generate_comprehensive_tests(architecture_plan),
        encoding="utf-8",
    )
    legacy_smoke = tests_dir / "test_smoke.py"
    if legacy_smoke.exists():
        legacy_smoke.unlink()

    metadata_by_id: Dict[str, Dict[str, Any]] = {}
    for node_id, _, _, metadata in node_specs:
        metadata_by_id[node_id] = dict(metadata)

    if isinstance(architecture_plan, dict):
        raw_nodes = architecture_plan.get("nodes")
        if isinstance(raw_nodes, list):
            for entry in raw_nodes:
                if not isinstance(entry, dict):
                    continue
                node_identifier = (
                    entry.get("id")
                    or entry.get("name")
                    or entry.get("label")
                    or ""
                )
                node_identifier = str(node_identifier).strip()
                if node_identifier and node_identifier not in metadata_by_id:
                    metadata_by_id[node_identifier] = dict(entry)

    architecture_metadata_lines: List[str] = []
    if isinstance(architecture_plan, dict):
        descriptive_keys = {
            "workflow_pattern": "Workflow pattern",
            "pattern": "Pattern",
            "default_pattern": "Default pattern",
            "description": "Description",
        }
        for key, label in descriptive_keys.items():
            value = architecture_plan.get(key)
            if not value:
                continue
            value_text = str(value).strip()
            if value_text:
                architecture_metadata_lines.append(f"- **{label}:** {value_text}")

        edges = architecture_plan.get("edges")
        if isinstance(edges, list) and edges:
            architecture_metadata_lines.append(
                f"- **Edges:** {len(edges)} defined transition{'s' if len(edges) != 1 else ''}"
            )

    node_table_lines: List[str] = []
    table_rows: List[str] = []

    def _render_description(entry: Dict[str, Any]) -> str:
        description_fields = (
            entry.get("description"),
            entry.get("purpose"),
            entry.get("summary"),
        )
        description = ""
        for candidate in description_fields:
            if candidate:
                description = str(candidate)
                break
        details: List[str] = []
        node_type = entry.get("type")
        if node_type:
            details.append(f"type={node_type}")
        tool = entry.get("tool")
        if tool:
            details.append(f"tool={tool}")
        if details:
            detail_text = ", ".join(details)
            if description:
                description = f"{description} ({detail_text})"
            else:
                description = detail_text
        description = description.replace("\n", " ").strip()
        return description or "n/a"

    if node_definitions:
        node_table_lines.append("| Node | Module | Callable | Description |")
        node_table_lines.append("| --- | --- | --- | --- |")
        for definition in node_definitions:
            node_id = definition.get("id", "")
            module_name = definition.get("module", "")
            callable_name = definition.get("callable") or "run"
            metadata = metadata_by_id.get(node_id, {})
            description = _render_description(metadata)
            table_rows.append(
                f"| {node_id} | `src/agent/{module_name}.py` | `{callable_name}` | {description} |"
            )

    if not table_rows and metadata_by_id:
        node_table_lines.append("| Node | Description |")
        node_table_lines.append("| --- | --- |")
        for node_id, metadata in metadata_by_id.items():
            description = _render_description(metadata)
            table_rows.append(f"| {node_id} | {description} |")

    node_table_lines.extend(table_rows)

    architecture_summary_parts: List[str] = []
    if architecture_metadata_lines:
        architecture_summary_parts.append("\n".join(architecture_metadata_lines))
    if node_table_lines:
        architecture_summary_parts.append("\n".join(node_table_lines))

    architecture_summary = (
        "\n\n".join(architecture_summary_parts)
        if architecture_summary_parts
        else "Architecture details were not provided."
    )

    readme_contents = f"""# {name}
Generated by Agentic-System-Builder MVP.

> **Project goal:** {user_goal}

## Installation
1. `pip install -e . \"langgraph-cli[inmem]\"`
2. `cp .env.example .env`
3. Configure environment variables for your LLM provider.

## Usage

### LangGraph Studio (visual development)
```bash
langgraph dev
```
Inspect and iterate on the agent interactively inside LangGraph Studio.

### Command-line execution
```bash
langgraph run src.agent.graph:graph --input payload.json
```
Feed structured payloads into the graph for repeatable workflows.

### Python API integration
```python
from src.agent.graph import graph

result = graph.invoke({{"input": "your request"}})
```
Embed the agent graph inside larger Python applications or tests.

## Testing
```bash
pytest -v
```
Run the automated suite before shipping changes.

## Customization
- Adjust prompts in `prompts/` to refine planning and execution behavior.
- Extend node logic within `src/agent/` and supporting LLM wrappers under `src/llm/`.
- Tune configuration defaults in `src/config/settings.py` and environment variables in `.env`.
- Add scenario coverage in `tests/` to validate new capabilities.

## Architecture summary
{architecture_summary}
"""

    (base / "README.md").write_text(readme_contents, encoding="utf-8")

    scaffold_status: Dict[str, Any] = {"path": str(base), "ok": True}
    if missing_files:
        scaffold_status["missing"] = missing_files
    if scaffold_errors:
        scaffold_status["errors"] = scaffold_errors
        scaffold_status["ok"] = False
    state["scaffold"] = scaffold_status
    return state
