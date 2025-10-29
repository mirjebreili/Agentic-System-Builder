from __future__ import annotations
import json
import logging
import re
from typing import Any, Dict, List
from jinja2 import Template
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, ValidationError
from agents.prompts_util import find_prompts_dir
from llm.client import get_chat_model
from utils.message_utils import extract_last_message_content

logger = logging.getLogger(__name__)

PROMPTS_DIR = find_prompts_dir()
SYSTEM_PROMPT_TMPL = Template((PROMPTS_DIR / "split_system.jinja").read_text(encoding="utf-8"))
USER_TMPL = (PROMPTS_DIR / "split_user.jinja").read_text(encoding="utf-8")

_JSON_BLOCK = re.compile(r"```json\s*(.*?)```", re.S | re.I)

def _extract_json(text: str) -> str:
    """Extract JSON from markdown code blocks."""
    m = _JSON_BLOCK.search(text or "")
    return m.group(1) if m else (text or "")

class SubTask(BaseModel):
    """A single atomic subtask."""
    id: int
    description: str
    
class SplitTaskResult(BaseModel):
    """Result of task splitting."""
    original_goal: str
    subtasks: List[SubTask]
    system_elements: List[str] = []

def _render_system_prompt(has_system_elements: bool, system_elements: List[str]) -> str:
    """Render the system prompt with context about available system elements."""
    return SYSTEM_PROMPT_TMPL.render(
        has_system_elements=has_system_elements,
        system_elements=system_elements
    )

def _render_user_prompt(goal: str) -> str:
    """Render the user prompt with the goal."""
    txt = USER_TMPL.replace("{{ user_goal }}", goal)
    return txt

def split_task(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Split the user's goal into atomic subtasks that cannot be further decomposed.
    
    This node analyzes the user's request and breaks it down into the simplest possible
    tasks. It also extracts any system elements (existing functions, databases, APIs, etc.)
    mentioned in the user's input.
    
    Args:
        state: The current AppState containing messages
        
    Returns:
        Updated state with split_tasks list and system_elements
    """
    llm = get_chat_model()
    messages = state.get("messages") or []
    user_goal = extract_last_message_content(messages, "No goal specified.")
    
    # Get system elements info from state (populated by extract_system_elements node)
    has_system_elements = state.get("has_system_elements", False)
    system_elements = state.get("system_elements", [])
    
    logger.info("Starting task splitting for goal: %s (has_system_elements=%s)", 
                user_goal, has_system_elements)
    
    # Create the prompt messages with context
    system_prompt = _render_system_prompt(has_system_elements, system_elements)
    sys = SystemMessage(system_prompt)
    user = HumanMessage(_render_user_prompt(user_goal))
    
    # Get the LLM response
    resp = llm.invoke([sys, user]).content
    logger.debug("Split task raw response: %s", resp)
    
    # Parse the JSON response
    try:
        json_str = _extract_json(resp)
        result_data = json.loads(json_str)
        
        # Validate with Pydantic
        split_result = SplitTaskResult.model_validate(result_data)
        
        # CRITICAL: Check if we have at least 1 subtask
        if not split_result.subtasks or len(split_result.subtasks) == 0:
            logger.error("Split task returned 0 subtasks! Attempting recovery...")
            raise ValueError("No subtasks generated")
        
        logger.info(
            "Successfully split task into %d subtasks with %d system elements",
            len(split_result.subtasks),
            len(split_result.system_elements)
        )
        
        # Convert to dict for state
        subtasks_list = [
            {"id": st.id, "description": st.description} 
            for st in split_result.subtasks
        ]
        
        # Update messages
        msgs = list(state.get("messages") or [])
        msgs.append({
            "role": "assistant", 
            "content": f"Split goal into {len(subtasks_list)} atomic subtasks."
        })
        
        return {
            "split_tasks": subtasks_list,
            "system_elements": split_result.system_elements,
            "messages": msgs,
        }
        
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        logger.warning("Failed to parse split task response, attempting repair: %s", exc)
        
        # Try to repair the JSON
        try:
            fix_sys = SystemMessage(
                "You MUST output a valid JSON object with at least 1 subtask. "
                "CRITICAL: The subtasks array cannot be empty! "
                "Even for simple goals, create at least 2-3 basic subtasks."
            )
            fix_user = HumanMessage(
                f"Original goal: {user_goal}\n\n"
                f"Previous response was invalid:\n{resp}\n\n"
                f"Generate a valid JSON with at least 3 subtasks for this goal."
            )
            fix_resp = llm.invoke([fix_sys, fix_user]).content
            
            json_str = _extract_json(fix_resp)
            result_data = json.loads(json_str)
            split_result = SplitTaskResult.model_validate(result_data)
            
            # Check again
            if not split_result.subtasks or len(split_result.subtasks) == 0:
                logger.error("Repair also returned 0 subtasks! Using fallback.")
                raise ValueError("Repair failed to generate subtasks")
            
            logger.info("Successfully repaired split task: %d subtasks", len(split_result.subtasks))
            
            subtasks_list = [
                {"id": st.id, "description": st.description} 
                for st in split_result.subtasks
            ]
            
            msgs = list(state.get("messages") or [])
            msgs.append({
                "role": "assistant", 
                "content": f"Split goal into {len(subtasks_list)} atomic subtasks (after repair)."
            })
            
            return {
                "split_tasks": subtasks_list,
                "system_elements": split_result.system_elements,
                "messages": msgs,
            }
            
        except Exception as repair_exc:
            logger.error("Repair failed: %s. Using fallback minimal split.", repair_exc)
            
            # FALLBACK: Create a minimal but valid split
            fallback_subtasks = [
                {"id": 1, "description": "Analyze requirements and constraints"},
                {"id": 2, "description": "Design solution architecture"},
                {"id": 3, "description": "Implement core functionality"},
                {"id": 4, "description": "Test and validate solution"},
                {"id": 5, "description": "Deploy and document"}
            ]
            
            logger.warning("Using fallback subtasks for goal: %s", user_goal)
            
            msgs = list(state.get("messages") or [])
            msgs.append({
                "role": "assistant", 
                "content": f"Split goal into {len(fallback_subtasks)} generic subtasks (fallback mode)."
            })
            
            return {
                "split_tasks": fallback_subtasks,
                "system_elements": [],
                "messages": msgs,
            }
