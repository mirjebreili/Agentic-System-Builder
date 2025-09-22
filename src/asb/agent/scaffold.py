from __future__ import annotations
import json, os, re, shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


_ADAPTIVE_STATE_FIELDS: List[tuple[str, str]] = [
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
    ("messages", "Annotated[List[AnyMessage], add_messages]"),
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


def generate_adaptive_state_schema(architecture_plan: Dict[str, Any] | None) -> str:
    """Render the default state.py template with plan-aware result fields."""

    nodes: List[str] = []
    if isinstance(architecture_plan, dict):
        raw_nodes = architecture_plan.get("nodes")
        if isinstance(raw_nodes, list):
            seen: set[str] = set()
            for entry in raw_nodes:
                if not isinstance(entry, dict):
                    continue
                raw_name = (
                    entry.get("name")
                    or entry.get("id")
                    or entry.get("label")
                )
                if raw_name is None:
                    continue
                sanitized = (
                    re.sub(r"\W+", "_", str(raw_name)).strip("_").lower() or "node"
                )
                field_name = f"{sanitized}_result"
                if field_name in seen:
                    continue
                seen.add(field_name)
                nodes.append(field_name)

    lines = [
        "from __future__ import annotations",
        "",
        "from typing import Annotated, Any, Dict, List, Literal, TypedDict",
        "",
        "from langchain_core.messages import AnyMessage",
        "from langgraph.graph.message import add_messages",
        "",
        "",
        "class ChatMessage(TypedDict, total=False):",
        '    role: Literal["human", "user", "assistant", "system", "tool"]',
        "    content: str",
        "",
        "",
        "class AppState(TypedDict, total=False):",
    ]

    for field_name, annotation in _ADAPTIVE_STATE_FIELDS:
        lines.append(f"    {field_name}: {annotation}")

    for node_field in nodes:
        lines.append(f"    {node_field}: Dict[str, Any]")

    lines.extend(
        [
            "",
            "",
            "def update_state_with_circuit_breaker(state: Dict[str, Any]) -> Dict[str, Any]:",
            '    """Add circuit breaker logic to prevent infinite loops"""',
            "",
            '    if "fix_attempts" not in state:',
            '        state["fix_attempts"] = 0',
            "",
            '    if "consecutive_failures" not in state:',
            '        state["consecutive_failures"] = 0',
            "",
            '    if "repair_start_time" not in state:',
            "        import time",
            "",
            '        state["repair_start_time"] = time.time()',
            "",
            "    return state",
        ]
    )

    return "\n".join(lines) + "\n"


STATE_TEMPLATE = generate_adaptive_state_schema({})

# repository root
ROOT = Path(__file__).resolve().parents[3]

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
    new_state = dict(state)
    container = dict(state.get("tot") or {})
    container.update(updates)
    new_state["tot"] = container
    new_state.pop("error", None)
    return new_state


def capture_tot_error(state: Dict[str, Any], node_id: str, error: Exception) -> Dict[str, Any]:
    new_state = dict(state)
    container = dict(state.get("tot") or {})
    errors = list(container.get("errors") or [])
    errors.append({"node": node_id, "message": str(error)})
    container["errors"] = errors
    new_state["tot"] = container
    new_state["error"] = str(error)
    return new_state
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

    return f"""from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import client


def {sanitized}(state: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"Adaptive implementation for the '{node_label}' node.\"\"\"

    llm = client.get_chat_model()
    raw_goal = state.get(\"goal\") or (state.get(\"plan\") or {{}}).get(\"goal\") or {goal_literal}
    goal = str(raw_goal).strip() or {goal_literal}
    context_data: Dict[str, Any] = {{}}
    for candidate in (
        state.get(\"context\"),
        (state.get(\"scaffold\") or {{}}).get(\"context\"),
    ):
        if isinstance(candidate, dict):
            context_data.update(candidate)

    context_lines = []
    if goal:
        context_lines.append(f\"Overall goal: {{goal}}\")
    node_purpose = {purpose_literal}
    if node_purpose:
        context_lines.append(f\"Node purpose: {{node_purpose}}\")
    if context_data:
        context_lines.append(\"Shared context:\")
        for key, value in context_data.items():
            context_lines.append(f\"- {{key}}: {{value}}\")

    messages = list(state.get(\"messages\") or [])
    latest_input = ""
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, dict):
            latest_input = str(last_message.get(\"content\") or "").strip()
        else:
            latest_input = str(getattr(last_message, \"content\", "") or "").strip()
    if latest_input:
        context_lines.append(f\"Latest user input: {{latest_input}}\")

    context = "\\n".join(context_lines) or "No additional context available."

    system_prompt = (
        "You are a LangGraph node executing a focused step in a larger workflow. "
        "Reason carefully about the provided context and respond with the best next action."
    )
    user_prompt = (
        "Use the context to fulfill this node's responsibility.\\n"
        f"{{context}}\\n"
        "Return a concise update describing what you accomplished and any outputs users should see."
    )

    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        content = getattr(response, \"content\", response)
        result_text = content if isinstance(content, str) else str(content)

        updated_state = dict(state)
        updated_state.pop(\"error\", None)
        messages.append({{"role": "assistant", "content": result_text, "node": {label_literal}}})
        updated_state["messages"] = messages
        updated_state[{result_key_literal}] = result_text
        return updated_state
    except Exception as exc:
        failed_state = dict(state)
        failed_state[\"error\"] = str(exc)
        return failed_state
"""


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
        utils_path.write_text(provided, encoding="utf-8")
        return

    if not utils_path.exists():
        utils_path.write_text(TOT_UTILS_TEMPLATE, encoding="utf-8")


def _render_generate_thoughts(node_id: str, sanitized: str) -> str:
    return f"""from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import client
from .utils import (
    capture_tot_error,
    extract_goal,
    extract_input_text,
    parse_approaches,
    update_tot_state,
)


def {sanitized}(state: Dict[str, Any]) -> Dict[str, Any]:
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
from .utils import (
    capture_tot_error,
    extract_goal,
    extract_input_text,
    get_thoughts,
    score_thoughts,
    update_tot_state,
)


def {sanitized}(state: Dict[str, Any]) -> Dict[str, Any]:
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

from .utils import (
    capture_tot_error,
    get_evaluations,
    get_thoughts,
    select_top_evaluation,
    update_tot_state,
)


def {sanitized}(state: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"Choose the most promising thought for finalization.\"\"\"

    try:
        evaluations = get_evaluations(state)
        if evaluations:
            selected = select_top_evaluation(evaluations)
            if not selected:
                raise ValueError("Evaluations were present but no selection could be made.")
            return update_tot_state(state, {{"selected_thought": selected}})

        thoughts = get_thoughts(state)
        if not thoughts:
            raise ValueError("No thoughts available to select from.")

        fallback = {{
            "index": 1,
            "thought": thoughts[0],
            "score": 0.0,
            "reasoning": "Defaulted to the first thought due to missing evaluations.",
        }}
        return update_tot_state(state, {{"selected_thought": fallback}})
    except Exception as exc:
        return capture_tot_error(state, "{sanitized}", exc)
"""


def _render_final_answer(node_id: str, sanitized: str) -> str:
    return f"""from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import client
from .utils import (
    capture_tot_error,
    extract_goal,
    extract_input_text,
    get_selected_thought,
    update_tot_state,
)


def {sanitized}(state: Dict[str, Any]) -> Dict[str, Any]:
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
        updated = update_tot_state(
            state,
            {{
                "final_answer": answer,
                "raw_final_response": answer,
                "selected_thought": selected,
            }},
        )
        messages = list(updated.get("messages") or state.get("messages") or [])
        messages.append({{"role": "assistant", "content": answer}})
        updated["messages"] = messages
        return updated
    except Exception as exc:
        return capture_tot_error(state, "{sanitized}", exc)
"""


_TOT_RENDERERS = {
    "generate_thoughts": _render_generate_thoughts,
    "evaluate_thoughts": _render_evaluate_thoughts,
    "select_best_thought": _render_select_best_thought,
    "final_answer": _render_final_answer,
}


def _write_node_modules(
    base: Path,
    node_specs: List[Tuple[str, str, List[str], Dict[str, Any]]],
    generated: Dict[str, str],
    missing_files: List[str],
    user_goal: str,
) -> None:
    agent_dir = base / "src" / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)

    tot_utils_prepared = False

    for node_id, module_name, _, node_details in node_specs:
        filename = f"{module_name}.py"
        source = _get_generated_content(
            generated,
            filename,
            f"src/agent/{filename}",
            f"agent/{filename}",
        )
        destination = agent_dir / filename

        if source is None:
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

        destination.write_text(source, encoding="utf-8")


def _render_executor_module(
    node_specs: List[Tuple[str, str, List[str], Dict[str, Any]]]
) -> str:
    lines: List[str] = [
        "# generated",
        "from __future__ import annotations",
        "",
        "import importlib",
        "from functools import lru_cache",
        "from typing import Any, Callable, Dict, Iterable, List, Tuple",
        "",
        "",
        "_NODE_SPECS: List[Tuple[str, str, List[str]]] = [",
    ]

    for node_id, module_name, hints, _ in node_specs:
        hint_repr = ", ".join(repr(hint) for hint in hints)
        lines.append(f"    ({node_id!r}, {module_name!r}, [{hint_repr}]),")

    lines.extend(
        [
            "]",
            "",
            "",
            "@lru_cache(maxsize=None)",
            "def _import_node(module: str):",
            "    package = __name__.rsplit('.', 1)[0]",
            "    return importlib.import_module(f\"{package}.{module}\")",
            "",
            "",
            "def _resolve_callable(module, hints: Iterable[str]):",
            "    for attr in hints:",
            "        func = getattr(module, attr, None)",
            "        if callable(func):",
            "            return func",
            "",
            "    for attr in dir(module):",
            "        if attr.startswith('_'):",
            "            continue",
            "        candidate = getattr(module, attr)",
            "        if callable(candidate):",
            "            return candidate",
            "",
            "    def _identity(state: Dict[str, Any]) -> Dict[str, Any]:",
            "        return state",
            "",
            "    module_name = module.__name__.split('.')[-1]",
            "    _identity.__name__ = f\"noop_{module_name}\"",
            "    return _identity",
            "",
            "",
            "NODE_IMPLEMENTATIONS: List[Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]]] = []",
            "for _node_id, _module_name, _hints in _NODE_SPECS:",
            "    _module = _import_node(_module_name)",
            "    _callable = _resolve_callable(_module, _hints)",
            "    NODE_IMPLEMENTATIONS.append((_node_id, _callable))",
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
        ]
    )

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
    base.mkdir(parents=True, exist_ok=True)
    (base / "prompts").mkdir(parents=True, exist_ok=True)
    (base / "src" / "agent").mkdir(parents=True, exist_ok=True)
    (base / "src" / "config").mkdir(parents=True, exist_ok=True)
    (base / "src" / "llm").mkdir(parents=True, exist_ok=True)
    (base / "tests").mkdir(parents=True, exist_ok=True)
    (base / "reports").mkdir(parents=True, exist_ok=True)

    for package_dir in ("src", "src/agent", "src/llm", "src/config"):
        init_path = base / package_dir / "__init__.py"
        init_path.parent.mkdir(parents=True, exist_ok=True)
        if not init_path.exists():
            init_path.write_text("", encoding="utf-8")

    # langgraph.json
    (base / "langgraph.json").write_text(
        json.dumps({"graphs": {"agent": "src.agent.graph:graph"},
                    "dependencies": ["."], "env": "./.env"}, indent=2), encoding="utf-8")

    # pyproject.toml
    (base / "pyproject.toml").write_text("""[project]
name = "generated-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "langgraph>=0.6,<0.7",
  "langchain-core>=0.3,<0.4",
  "langchain-openai>=0.3,<0.4",
  "pydantic>=2.7,<3",
  "langgraph-checkpoint-sqlite>=2.0.0",
  "aiosqlite>=0.17.0",
  "pytest>=7.0.0",
  "langgraph-cli[inmem]>=0.1.0",
  "requests>=2.25.0",
  "black>=22.0.0",
  "isort>=5.0.0",
  "mypy>=1.0.0",
  "bandit[toml]>=1.7.0",
]
[build-system]
requires = ["setuptools","wheel"]
build-backend = "setuptools.build_meta"
[tool.setuptools.packages.find]
where = ["src"]
""", encoding="utf-8")

    # .env.example
    src_env = ROOT / ".env.example"
    if src_env.exists():
        shutil.copy(src_env, base / ".env.example")

    # copy minimal settings, client, state, and prompt utilities
    files = {
        "src/config/settings.py": "src/config/settings.py",
        "src/asb/llm/client.py": "src/llm/client.py",
        "src/asb/agent/prompts_util.py": "src/agent/prompts_util.py",
    }
    missing_files = []
    for src_rel, dest_rel in files.items():
        dst = base / dest_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        src_path = ROOT / src_rel
        if src_path.exists():
            shutil.copy(src_path, dst)
        else:
            missing_files.append(str(src_path))
            print(f"Template file missing, skipping: {src_path}")

    normalized_generated = {
        _normalize_generated_key(key): value
        for key, value in (state.get("generated_files") or {}).items()
    }

    generated_state = _get_generated_content(
        normalized_generated,
        "state.py",
        "src/agent/state.py",
        "agent/state.py",
    )
    state_path = base / "src" / "agent" / "state.py"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if generated_state:
        state_path.write_text(generated_state, encoding="utf-8")
    else:
        fallback_state = generate_adaptive_state_schema(architecture_plan)
        state_path.write_text(fallback_state, encoding="utf-8")

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

        messages = list(state.get("messages") or [])
        messages.append(AIMessage(content=f"Generated plan for goal: {goal}"))
        current_step = {"more_steps": True, "steps_done": False}
        return {"plan": plan, "messages": messages, "current_step": current_step}
    except Exception as exc:
        return {
            "error": str(exc),
            "messages": list(state.get("messages") or []),
            "current_step": {"more_steps": False, "steps_done": False},
        }
""", encoding="utf-8")

    architecture = state.get("architecture") or {}
    architecture_plan = architecture.get("plan")
    if not isinstance(architecture_plan, dict):
        architecture_plan = architecture if isinstance(architecture, dict) else {}

    generated_architecture_nodes = generate_nodes_from_architecture(
        architecture_plan,
        user_goal,
    )

    node_specs: List[Tuple[str, str, List[str], Dict[str, Any]]] = []
    if generated_architecture_nodes:
        nodes_list = architecture_plan.get("nodes") or []
        seen_nodes: set[str] = set()
        for entry in nodes_list:
            if not isinstance(entry, dict):
                continue

            raw_name = entry.get("name") or entry.get("id") or entry.get("label")
            if raw_name is None:
                continue

            node_id = str(raw_name).strip()
            if not node_id or node_id in seen_nodes:
                continue

            module_name = _sanitize_identifier(node_id)
            hints = _candidate_call_hints(node_id, module_name)
            metadata = dict(entry)
            metadata.setdefault("id", node_id)
            node_specs.append((node_id, module_name, hints, metadata))
            seen_nodes.add(node_id)

        agent_dir = base / "src" / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        for filename, source in generated_architecture_nodes.items():
            (agent_dir / filename).write_text(source, encoding="utf-8")
    else:
        node_specs = _collect_architecture_nodes(architecture)

        if node_specs:
            _write_node_modules(
                base,
                node_specs,
                normalized_generated,
                missing_files,
                user_goal,
            )

    if node_specs:
        executor_source = _get_generated_content(
            normalized_generated,
            "executor.py",
            "src/agent/executor.py",
            "agent/executor.py",
        )
        if executor_source is None:
            executor_source = _render_executor_module(node_specs)

        (base / "src/agent/executor.py").write_text(executor_source, encoding="utf-8")

        graph_source = _get_generated_content(
            normalized_generated,
            "graph.py",
            "src/agent/graph.py",
            "agent/graph.py",
        )
        if graph_source is None:
            graph_source = _render_graph_module()

        (base / "src/agent/graph.py").write_text(graph_source, encoding="utf-8")
    else:
        (base / "src/agent/executor.py").write_text("""# generated
from langchain_core.messages import HumanMessage
from llm.client import get_chat_model
from typing import Dict, Any

def execute(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        llm = get_chat_model()
        plan = state.get("plan", {})
        nodes = {n["id"]: n for n in plan.get("nodes", [])}

        # Assuming the input text is in the last message
        input_text = (state.get("messages") or [{}])[-1].get("content", "")

        summarize_prompt = (nodes.get("summarize") or {}).get("prompt", "Summarize the following text: {{input_text}}")
        prompt = summarize_prompt.replace("{{input_text}}", input_text)

        summary = llm.invoke([HumanMessage(prompt)]).content

        messages = list(state.get("messages") or []) + [{"role": "assistant", "content": summary}]
        return {"messages": messages}
    except Exception as e:
        # Handle potential errors, e.g., LLM call fails
        return {"error": str(e), "messages": list(state.get("messages") or [])}
""", encoding="utf-8")

        (base / "src/agent/graph.py").write_text("""# generated
from __future__ import annotations

import logging
import os
import sqlite3
import sys  # required for runtime argv detection
from typing import Any, Dict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END

from .state import AppState
from .planner import plan_node
from .executor import execute

logger = logging.getLogger(__name__)


def running_on_langgraph_api() -> bool:
    langgraph_env = os.environ.get("LANGGRAPH_ENV", "").lower()
    if langgraph_env in {"cloud", "api", "hosted"}:
        return True
    if os.environ.get("LANGGRAPH_API_URL") or os.environ.get("LANGGRAPH_CLOUD"):
        return True
    argv = [arg.lower() for arg in sys.argv[1:]]
    return "--langgraph-api" in argv or ("langgraph" in argv and "api" in argv)


def _make_graph(path: str | None = None):
    g = StateGraph(AppState)
    g.add_node("plan", plan_node)
    g.add_node("execute", execute)
    g.add_edge(START, "plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", END)

    if running_on_langgraph_api():
        logger.info("LangGraph API runtime detected; compiling without a checkpointer.")
        return g.compile(checkpointer=None)

    checkpointer = None
    if path:
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        connection = sqlite3.connect(path, check_same_thread=False)
        checkpointer = SqliteSaver(connection)

    return g.compile(checkpointer=checkpointer)


graph = _make_graph()
""", encoding="utf-8")

    # tests
    (base / "tests" / "test_smoke.py").write_text(
        '''"""Smoke tests for the generated agent project."""

import importlib
from pathlib import Path

import pytest


def test_import_graph():
    module = importlib.import_module("src.agent.graph")
    assert hasattr(module, "graph")
    assert module.graph is not None


def test_state_structure():
    state_module = importlib.import_module("src.agent.state")
    assert hasattr(state_module, "AppState")
    state_keys = set(getattr(state_module, "AppState").__annotations__)
    expected_keys = {
        "architecture",
        "artifacts",
        "debug",
        "flags",
        "generated_files",
        "messages",
        "metrics",
        "passed",
        "plan",
        "replan",
        "report",
        "requirements",
        "review",
        "sandbox",
        "scaffold",
        "syntax_validation",
        "tests",
    }
    assert expected_keys.issubset(state_keys)


def test_graph_execution(tmp_path: Path):
    from src.agent.graph import _make_graph

    checkpoint_path = tmp_path / "checkpoints" / "graph.db"
    graph = _make_graph(str(checkpoint_path))
    result = graph.invoke(
        {"messages": []}, config={"configurable": {"thread_id": "test-smoke"}}
    )
    assert isinstance(result, dict)
    assert "messages" in result
    assert isinstance(result["messages"], list)
    assert any(
        key in result for key in ("plan", "current_step", "scaffold", "report")
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
''',
        encoding="utf-8",
    )

    # README
    (base / "README.md").write_text(
        f"""# {name}
Generated by Agentic-System-Builder MVP.

## Features
- Production-ready LangGraph scaffold with planner, executor, and state management modules.
- Prompt templates and configuration helpers for rapid iteration.
- Testing and reporting directories to validate agent behavior out of the box.

## Installation
1. `pip install -e . "langgraph-cli[inmem]"`
2. `cp .env.example .env`
3. Configure environment variables for your LLM provider.

## Usage

### Chat-style interaction
```bash
langgraph dev
```
Launch LangGraph Studio and iterate conversationally with the agent.

### Direct data invocation
```bash
langgraph run src.agent.graph:graph --input data.json
```
Supply structured payloads directly to the graph for batch workflows.

## Development
- Customize prompts in `prompts/` to refine planning behavior.
- Extend business logic under `src/agent/` and `src/llm/`.
- Update configuration defaults in `src/config/settings.py`.

## Testing
```bash
pytest -v
```
Run the automated suite before shipping changes.
""",
        encoding="utf-8",
    )

    state["scaffold"] = {"path": str(base), "ok": True}
    if missing_files:
        state["scaffold"]["missing"] = missing_files
    return state
