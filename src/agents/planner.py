from __future__ import annotations
import json
import re
import copy
from typing import Any, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, ValidationError

# New infrastructure imports
from src.utils.prompt_manager import get_prompt_manager
from src.utils.retry import invoke_llm_with_retry
from src.utils.logger import get_logger, log_node_execution, PerformanceLogger
from src.utils.metrics import get_metrics_collector
from src.config.app_settings import settings
from src.llm.client import get_chat_model
from src.utils.message_utils import extract_last_message_content

# Get structured logger, metrics, and prompt manager
_logger = get_logger(__name__)
metrics = get_metrics_collector()
pm = get_prompt_manager()

class PlanNode(BaseModel):
    id: str
    prompt: str | None = None
    tool: str | None = None
    reasoning: str | None = None

class PlanEdge(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    if_: str | None = Field(None, alias="if")

class Plan(BaseModel):
    goal: str
    nodes: list[PlanNode]
    edges: list[PlanEdge]
    confidence: float | None = None
    reasoning: str | None = None


def _render_system_prompt(has_system_elements: bool, system_elements: list[str], K: int = 3) -> str:
    """Render the system prompt with context about available system elements."""
    return pm.render(
        "plan_system",
        has_system_elements=has_system_elements,
        system_elements=system_elements,
        K=K
    )

def _render_user_prompt(goal: str, constraints: str | None = None, split_tasks: list[dict] | None = None, system_elements: list[str] | None = None) -> str:
    """Render user prompt with goal, tasks, and system elements."""
    return pm.render(
        "plan_user",
        user_goal=goal,
        constraints=constraints or "Keep it simple and actionable.",
        split_tasks=split_tasks or [],
        system_elements=system_elements or []
    )

_JSON_BLOCK = re.compile(r"```json\s*(.*?)```", re.S | re.I)
def _extract_json(text: str) -> str:
    m = _JSON_BLOCK.search(text or "")
    return m.group(1) if m else (text or "")


@log_node_execution("plan_tot")
def plan_tot(state: Dict[str, Any]) -> Dict[str, Any]:
    """ToT: generate K=3 plans, judge, pick best; attach confidence.
    
    Plan B Refactoring: Now works directly with goal and system_elements from extract_context,
    no longer depends on split_task subtasks.
    """
    llm = get_chat_model()
    
    # Get goal and system elements directly from state (set by extract_context)
    user_goal = state.get("goal", "")
    system_elements = state.get("system_elements", [])
    has_system_elements = state.get("has_system_elements", False)
    
    if not user_goal:
        _logger.warning("No goal found in state, using fallback")
        user_goal = "Plan a tiny workflow."
    
    _logger.info("Planning with ToT", extra={
        "user_goal_preview": user_goal[:50] + "..." if len(user_goal) > 50 else user_goal,
        "system_elements_count": len(system_elements),
        "has_system_elements": has_system_elements
    })
    
    K = 3

    # Render system prompt with context
    system_prompt = _render_system_prompt(has_system_elements, system_elements, K)
    sys = SystemMessage(system_prompt + f"\nReturn {K} ALTERNATIVE JSON plans as a JSON array.")
    # Note: split_tasks removed - planner now has full creative freedom to decompose
    user = HumanMessage(_render_user_prompt(user_goal, split_tasks=None, system_elements=system_elements))
    
    with PerformanceLogger(_logger, "plan_tot_llm_call"):
        ai_message = invoke_llm_with_retry(llm, [sys, user])
        resp = ai_message.content
    

    # Parse / repair
    try:
        cand = json.loads(_extract_json(resp))
    except Exception:
        _logger.warning("Initial plan parsing failed, attempting repair")
        fix_sys = SystemMessage("Output ONLY a JSON array of plan objects.")
        fix_user = HumanMessage(resp)
        
        with PerformanceLogger(_logger, "plan_tot_repair_llm_call"):
            fix_message = invoke_llm_with_retry(llm, [fix_sys, fix_user])
            fix = fix_message.content
        
        _logger.debug("Planner repair output received", extra={"output_length": len(fix)})
        try:
            cand = json.loads(_extract_json(fix))
        except json.JSONDecodeError:
            _logger.warning("Planner repair output is not valid JSON, falling back to default plan", exc_info=True)
            cand = []
    if isinstance(cand, dict) and "alternatives" in cand:
        cand = cand["alternatives"]

    valid: list[dict] = []
    for c in cand if isinstance(cand, list) else []:
        try:
            validated_plan = Plan.model_validate(c).model_dump(by_alias=True)
            valid.append(validated_plan)
            _logger.info(
                "Validated plan",
                extra={
                    "confidence": validated_plan.get("confidence", "N/A"),
                    "nodes_count": len(validated_plan.get("nodes", [])),
                    "first_tool": validated_plan.get("nodes", [{}])[0].get("tool", "N/A") if validated_plan.get("nodes") else "N/A"
                }
            )
        except ValidationError as exc:
            _logger.warning("Plan validation failed", extra={"error": str(exc)})

    if not valid:
        _logger.warning("Planner produced no valid plans, falling back to empty plan")
        empty_plan: dict[str, Any] = {"goal": user_goal, "nodes": [], "edges": [], "confidence": 0.0}
        return {"plan": empty_plan, "messages": state.get("messages", []), "flags": {"more_steps": True, "steps_done": False}}

    # Select the plan with the highest confidence directly
    best = copy.deepcopy(max(valid, key=lambda p: float(p.get("confidence") or 0.0)))
    best_confidence = float(best.get("confidence") or 0.0)
    _logger.info(
        "SELECTED BEST PLAN",
        extra={
            "confidence": best_confidence,
            "nodes_count": len(best.get("nodes", [])),
            "first_tool": best.get("nodes", [{}])[0].get("tool", "N/A") if best.get("nodes") else "N/A"
        }
    )

    # Store all candidate plans for downstream debugging/inspection
    debug = dict(state.get("debug") or {})
    debug_candidates = []
    for p in valid:
        debug_candidates.append(
            {
                "confidence": float(p.get("confidence") or 0.0),
                "first_tool": p.get("nodes", [{}])[0].get("tool", "N/A") if p.get("nodes") else "N/A",
                "node_count": len(p.get("nodes", [])),
                "plan": copy.deepcopy(p),
            }
        )
    debug["plan_candidates"] = debug_candidates
    state_debug_messages = debug

    msgs = list(state.get("messages") or [])
    msgs.append({"role": "assistant", "content": f"Selected ToT plan (score={best_confidence:.2f})."})

    _logger.debug("Planner debug - selected plan: %s", best)

    return {
        "plan": copy.deepcopy(best),
        "messages": msgs,
        "debug": state_debug_messages,
    }
