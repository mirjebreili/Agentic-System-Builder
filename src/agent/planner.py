import json
import logging
from typing import Optional, List

from jinja2 import Environment, FileSystemLoader
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, ValidationError

from prompt2graph.config.settings import get_settings
from prompt2graph.llm.client import get_chat_model
from .confidence import compute_plan_confidence
from .state import AppState, ChatMessage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Schemas (already defined, but kept here for context) ---

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

# --- Prompt Rendering ---

def _render_prompt(template_file: str, **kwargs) -> str:
    """Renders a Jinja2 prompt template."""
    env = Environment(loader=FileSystemLoader("prompt2graph/prompts/"))
    template = env.get_template(template_file)
    return template.render(**kwargs)

# --- JSON Parsing with Repair ---

def _parse_and_validate_plan(llm_output: str, llm) -> Optional[Plan]:
    """Parses LLM output into a Plan object, with one repair attempt."""
    try:
        # Basic cleanup
        cleaned_output = llm_output.strip().removeprefix("```json").removesuffix("```")
        plan_dict = json.loads(cleaned_output)
        return Plan.model_validate(plan_dict)
    except (json.JSONDecodeError, ValidationError) as e:
        logging.warning(f"Initial plan validation failed: {e}. Attempting repair.")
        repair_prompt = f"The following JSON is invalid. Please fix it and return only the valid JSON object.\n\nInvalid JSON:\n{llm_output}\n\nError:\n{e}"
        repaired_output = llm.invoke([HumanMessage(content=repair_prompt)]).content
        try:
            cleaned_repaired_output = repaired_output.strip().removeprefix("```json").removesuffix("```")
            repaired_dict = json.loads(cleaned_repaired_output)
            return Plan.model_validate(repaired_dict)
        except (json.JSONDecodeError, ValidationError) as e2:
            logging.error(f"Plan repair failed: {e2}")
            return None

# --- Fallback Plan ---

def _create_fallback_plan(goal: str) -> Plan:
    """Creates a hardcoded, minimal fallback plan."""
    logging.warning("All candidates failed validation. Creating a fallback plan.")
    return Plan(
        goal=f"Fallback plan for: {goal}",
        nodes=[
            PlanNode(id="plan", type="llm", prompt="Break down the user's goal into simple steps."),
            PlanNode(id="do", type="llm", prompt="Execute the steps. Output 'DONE' when complete."),
            PlanNode(id="finish", type="llm", prompt="Summarize the results.")
        ],
        edges=[
            PlanEdge(from_="plan", to="do"),
            PlanEdge(from_="do", to="do", if_="more_steps"),
            PlanEdge(from_="do", to="finish", if_="steps_done")
        ],
        confidence=0.1
    )

# --- ToT Planner Node ---

def plan_tot(state: AppState) -> AppState:
    """
    The main planner node using a simplified Tree-of-Thoughts (ToT) approach.
    Generates K candidate plans, judges them, and selects the best one.
    """
    llm = get_chat_model()
    user_goal = state["messages"][-1]["content"]

    # Load configuration
    settings = get_settings()
    tot_gate = settings.tot_gate
    branches = settings.tot_branches if tot_gate else 1

    # 1. Generate K candidate plans
    plan_system_prompt = _render_prompt("plan_system.jinja")
    plan_user_prompt = _render_prompt("plan_user.jinja", user_goal=user_goal)

    if tot_gate:
        logging.info(
            f"Generating {branches} plan candidates for goal: '{user_goal}'"
        )
    else:
        logging.info(f"Generating single plan candidate for goal: '{user_goal}'")

    candidate_futures = [
        llm.ainvoke([SystemMessage(content=plan_system_prompt), HumanMessage(content=plan_user_prompt)])
        for _ in range(branches)
    ]

    # For simplicity in this sync function, we'll await them one by one.
    # In a real async node, we'd use asyncio.gather.
    raw_candidates = [future.result().content for future in candidate_futures]

    # 2. Validate and Judge Candidates
    judge_system_prompt = "You are a plan evaluator. Given a user goal and a JSON plan, rate the plan's quality on a scale from 0.0 to 1.0. Output ONLY a JSON object with 'score' and 'reason' keys."
    valid_plans = []
    for raw_plan in raw_candidates:
        plan = _parse_and_validate_plan(raw_plan, llm)
        if plan:
            judge_user_prompt = f"User Goal:\n{user_goal}\n\nPlan:\n{plan.model_dump_json(indent=2)}"
            judge_response = llm.invoke([SystemMessage(content=judge_system_prompt), HumanMessage(content=judge_user_prompt)]).content
            try:
                judge_result = json.loads(judge_response)
                valid_plans.append({"plan": plan, "score": judge_result.get("score", 0.0), "reason": judge_result.get("reason", "")})
            except json.JSONDecodeError:
                logging.warning("Failed to parse judge's response.")
                valid_plans.append({"plan": plan, "score": 0.3, "reason": "Judge response was malformed."})

    # 3. Select Best Plan or Create Fallback
    if not valid_plans:
        best_plan = _create_fallback_plan(user_goal)
        best_score = 0.1
    else:
        best_candidate = max(valid_plans, key=lambda x: x["score"])
        best_plan = best_candidate["plan"]
        best_score = best_candidate["score"]
        logging.info(f"Selected best plan with judge score: {best_score}. Reason: {best_candidate['reason']}")

    # 4. Compute Final Confidence and Update State
    final_confidence, confidence_terms = compute_plan_confidence(best_plan, state)
    best_plan.confidence = final_confidence

    new_messages = state.get("messages", []) + [ChatMessage(role="assistant", content=f"Selected ToT plan (judge_score={best_score:.2f}, final_confidence={final_confidence:.2f}).")]

    # Initialize metrics if they don't exist
    metrics = state.get("metrics", {"prior_attempts": 0, "prior_successes": 0, "fail_streak": 0})

    return {
        **state,
        "plan": best_plan.model_dump(),
        "messages": new_messages,
        "flags": {"more_steps": True, "steps_done": False},
        "metrics": metrics,
        "debug": {"confidence_terms": confidence_terms},
    }
