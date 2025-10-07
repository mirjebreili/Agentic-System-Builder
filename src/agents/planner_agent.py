"""Planner graph responsible for producing a plugin execution plan."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from ..llm import get_llm
from ..prompts import PLANNER_PROMPT, SYSTEM_PROMPT
from ..utils import PlannerState


def _format_available_plugins(plugins: Iterable[Dict[str, Any]]) -> str:
    """Render the available plugins into a human readable bullet list."""

    items: List[str] = []
    for plugin in plugins or []:
        name = plugin.get("name") or plugin.get("plugin") or "unknown"
        description = plugin.get("description") or plugin.get("summary") or ""
        entry = f"- {name}: {description}".rstrip()
        signature = plugin.get("schema") or plugin.get("parameters") or plugin.get("args")
        if signature:
            if isinstance(signature, (dict, list)):
                formatted = json.dumps(signature, indent=2, sort_keys=True)
            else:
                formatted = str(signature)
            entry = f"{entry}\n  Args: {formatted}"
        items.append(entry)
    return "\n".join(items) if items else "(no plugins registered)"


def _normalise_plan(plan_text: str) -> List[Dict[str, Any]]:
    """Normalise the LLM response into the canonical plugin sequence format."""

    try:
        parsed = json.loads(plan_text)
    except json.JSONDecodeError:
        parsed = None

    sequence: List[Dict[str, Any]] = []
    if isinstance(parsed, list):
        for step in parsed:
            if isinstance(step, dict):
                sequence.append(
                    {
                        "plugin": step.get("plugin") or step.get("name") or "",
                        "description": step.get("description")
                        or step.get("reasoning")
                        or "",
                        "args": step.get("args")
                        or step.get("parameters")
                        or step.get("kwargs")
                        or {},
                    }
                )
            else:
                sequence.append(
                    {
                        "plugin": str(step),
                        "description": "",
                        "args": {},
                    }
                )
    else:
        sequence.append(
            {
                "plugin": "",
                "description": plan_text.strip(),
                "args": {},
            }
        )

    return sequence


def plan_plugins(state: PlannerState) -> PlannerState:
    """Plan the sequence of plugins to execute for the given input state."""

    llm = get_llm()
    available_plugins = _format_available_plugins(state.get("available_plugins", []))
    prompt = PLANNER_PROMPT.format(
        user_question=state.get("input_text", ""),
        plugin_descriptions=available_plugins,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    content: Any = getattr(response, "content", response)
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    plan_text = str(content)

    new_state: PlannerState = dict(state)
    new_state["plugin_sequence"] = _normalise_plan(plan_text)
    return new_state


def create_planner_graph() -> Any:
    """Create and compile the planner graph."""

    workflow = StateGraph(PlannerState)
    workflow.add_node("plan_plugins", plan_plugins)
    workflow.set_entry_point("plan_plugins")
    workflow.add_edge("plan_plugins", END)
    return workflow.compile()


graph = create_planner_graph()

