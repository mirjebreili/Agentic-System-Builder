from __future__ import annotations
import json
import logging
import re
from typing import Any, Dict, List
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, ValidationError
from agents.prompts_util import find_prompts_dir
from llm.client import get_chat_model
from utils.message_utils import extract_last_message_content

logger = logging.getLogger(__name__)

PROMPTS_DIR = find_prompts_dir()
SYSTEM_PROMPT = (PROMPTS_DIR / "split_system.jinja").read_text(encoding="utf-8")
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
    
    logger.info("Starting task splitting for goal: %s", user_goal)
    
    # Create the prompt messages
    sys = SystemMessage(SYSTEM_PROMPT)
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
        
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Failed to parse split task response, attempting repair: %s", exc)
        
        # Try to repair the JSON
        try:
            fix_sys = SystemMessage("Output ONLY a valid JSON object matching the schema.")
            fix_user = HumanMessage(f"Original response:\n{resp}\n\nFix this to valid JSON.")
            fix_resp = llm.invoke([fix_sys, fix_user]).content
            
            json_str = _extract_json(fix_resp)
            result_data = json.loads(json_str)
            split_result = SplitTaskResult.model_validate(result_data)
            
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
            logger.error("Failed to repair split task response: %s", repair_exc)
            
            # Fall back to a single task
            msgs = list(state.get("messages") or [])
            msgs.append({
                "role": "assistant", 
                "content": "Failed to split task; using original goal as single task."
            })
            
            return {
                "split_tasks": [{"id": 1, "description": user_goal}],
                "system_elements": [],
                "messages": msgs,
            }
