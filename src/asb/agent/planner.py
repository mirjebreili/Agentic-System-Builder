from __future__ import annotations
import json
import logging
import re
from typing import Any, Dict, List, Literal
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, ValidationError
from asb.agent.prompts_util import find_prompts_dir
from asb.llm.client import get_chat_model
from asb.utils.message_utils import extract_last_message_content

class Capability(BaseModel):
    name: str
    description: str
    ecosystem_priority: List[Literal["npm", "pypi"]]
    search_terms: List[str]
    complexity: Literal["low", "medium", "high"]
    required: bool = True


class Plan(BaseModel):
    goal: str
    capabilities: List[Capability]
    architecture_approach: Literal["microservices", "monolithic", "hybrid"]
    primary_language: Literal["javascript", "python", "mixed"]
    integration_strategy: str
    confidence: float | None = None

logger = logging.getLogger(__name__)

PROMPTS_DIR = find_prompts_dir()
SYSTEM_PROMPT = (PROMPTS_DIR / "plan_system.jinja").read_text(encoding="utf-8")
USER_TMPL = (PROMPTS_DIR / "plan_user.jinja").read_text(encoding="utf-8")

def _render_user_prompt(goal: str, constraints: str | None = None) -> str:
    txt = USER_TMPL.replace("{{ user_goal }}", goal)
    txt = txt.replace('{{ constraints | default("Keep it simple and low-cost.") }}',
                      constraints or "Keep it simple and low-cost.")
    return txt

_JSON_BLOCK = re.compile(r"```json\s*(.*?)```", re.S | re.I)
def _extract_json(text: str) -> str:
    m = _JSON_BLOCK.search(text or "")
    return m.group(1) if m else (text or "")

def plan_tot(state: Dict[str, Any]) -> Dict[str, Any]:
    """ToT: generate K=3 plans, judge, pick best; attach confidence."""
    llm = get_chat_model()
    messages = state.get("messages") or []
    user_goal = extract_last_message_content(messages, "Plan a tiny workflow.")
    K = 3

    sys = SystemMessage(
        SYSTEM_PROMPT
        + (
            f"\nReturn EXACTLY {K} alternative plan JSON objects as a JSON array."
            " Each plan must follow the schema above, include unique capability"
            " sets, and must not include any explanatory text."
        )
    )
    user = HumanMessage(_render_user_prompt(user_goal))
    resp = llm.invoke([sys, user]).content

    # Parse / repair
    try:
        cand = json.loads(_extract_json(resp))
    except Exception:
        fix = llm.invoke([SystemMessage("Output ONLY a JSON array of plan objects."), HumanMessage(resp)]).content
        logger.debug("Planner fix raw output: %s", fix)
        try:
            cand = json.loads(_extract_json(fix))
        except json.JSONDecodeError:
            logger.warning("Planner fix output is not valid JSON. Falling back to default plan.", exc_info=True)
            cand = []
    if isinstance(cand, dict) and "alternatives" in cand:
        cand = cand["alternatives"]

    valid: list[dict] = []
    for c in cand if isinstance(cand, list) else []:
        try:
            valid.append(Plan.model_validate(c).model_dump(by_alias=True))
        except ValidationError:
            continue
    if not valid:
        valid = [
            {
                "goal": user_goal,
                "capabilities": [
                    {
                        "name": "requirements_analysis",
                        "description": "Collect requirements and determine package-friendly deliverables.",
                        "ecosystem_priority": ["npm", "pypi"],
                        "search_terms": [
                            "project requirements toolkit",
                            "npm checklist",
                            "pypi analyst tools",
                        ],
                        "complexity": "low",
                        "required": True,
                    },
                    {
                        "name": "solution_implementation",
                        "description": "Assemble key packages to build the requested outcome with minimal glue code.",
                        "ecosystem_priority": ["pypi", "npm"],
                        "search_terms": [
                            "fastapi",
                            "express",
                            "automation toolkit",
                        ],
                        "complexity": "medium",
                        "required": True,
                    },
                ],
                "architecture_approach": "monolithic",
                "primary_language": "python",
                "integration_strategy": "Coordinate selected packages via a lightweight Python orchestrator.",
                "confidence": 0.5,
            }
        ]

    # Judge
    scored: list[tuple[float, str, dict]] = []
    judge = get_chat_model()
    for p in valid:
        crit = judge.invoke(
            [
                SystemMessage(
                    "Score 0..1 on coverage, ecosystem fit, and clarity."
                    " Output JSON {\"score\":x,\"reason\":\"...\"}."
                ),
                HumanMessage(json.dumps({"goal": user_goal, "plan": p}, ensure_ascii=False)),
            ]
        ).content
        try:
            j = json.loads(_extract_json(crit))
            scored.append((float(j.get("score", 0.0)), j.get("reason",""), p))
        except Exception:
            scored.append((0.5, "fallback", p))

    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][2]
    best["confidence"] = float(min(1.0, max(0.0, best.get("confidence") or scored[0][0])))

    msgs = list(state.get("messages") or [])
    msgs.append({"role":"assistant","content":f"Selected ToT plan (score={scored[0][0]:.2f})."})

    try:
        from asb.agent.executor import update_node_implementations

        update_node_implementations(best)
    except Exception:
        logger.debug("Unable to update node implementations for plan.", exc_info=True)

    logger.debug("Planner debug - selected plan: %s", best)

    return {"plan": best, "messages": msgs, "flags":{"more_steps": True, "steps_done": False}}
