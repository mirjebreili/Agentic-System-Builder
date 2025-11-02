from __future__ import annotations
import json
import re
from typing import Any, Dict, List
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, ValidationError

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

def _build_system_prompt(
    has_system_elements: bool,
    system_elements: List[Dict[str, Any]]
) -> SystemMessage:
    """Render split_system.jinja with system elements context."""
    system_text = pm.render(
        "split_system",
        has_system_elements=has_system_elements,
        system_elements=system_elements
    )
    return SystemMessage(content=system_text)


def _build_user_msg(goal: str) -> HumanMessage:
    """Render split_user.jinja with the user's goal."""
    user_text = pm.render("split_user", user_goal=goal)
    return HumanMessage(content=user_text)


@log_node_execution("split_task")
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
    
    _logger.info("Starting task splitting", 
                extra={"user_goal": user_goal, "has_system_elements": has_system_elements})
    
    # Create the prompt messages with context
    sys = _build_system_prompt(has_system_elements, system_elements)
    user = _build_user_msg(user_goal)
    
    # Get the LLM response with retry logic
    with PerformanceLogger(_logger, "split_task_llm_call"):
        ai_message = invoke_llm_with_retry(llm, [sys, user])
        resp = ai_message.content
    
    _logger.debug("Split task raw response received", extra={"response_length": len(resp)})
    
    # Parse the JSON response
    try:
        json_str = _extract_json(resp)
        result_data = json.loads(json_str)
        
        # Validate with Pydantic
        split_result = SplitTaskResult.model_validate(result_data)
        
        # CRITICAL: Check if we have at least 1 subtask
        if not split_result.subtasks or len(split_result.subtasks) == 0:
            _logger.error("Split task returned 0 subtasks, attempting recovery")
            raise ValueError("No subtasks generated")
        
        _logger.info(
            "Successfully split task",
            extra={
                "subtasks_count": len(split_result.subtasks),
                "system_elements_count": len(split_result.system_elements)
            }
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
        _logger.warning("Failed to parse split task response, attempting repair", 
                      extra={"error": str(exc), "error_type": type(exc).__name__})
        
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
            
            with PerformanceLogger(_logger, "split_task_repair_llm_call"):
                fix_message = invoke_llm_with_retry(llm, [fix_sys, fix_user])
                fix_resp = fix_message.content
            
            json_str = _extract_json(fix_resp)
            result_data = json.loads(json_str)
            split_result = SplitTaskResult.model_validate(result_data)
            
            # Check again
            if not split_result.subtasks or len(split_result.subtasks) == 0:
                _logger.error("Repair also returned 0 subtasks, using fallback")
                raise ValueError("Repair failed to generate subtasks")
            
            _logger.info("Successfully repaired split task", 
                       extra={"subtasks_count": len(split_result.subtasks)})
            
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
            _logger.error("Repair failed, using fallback minimal split", 
                        extra={"error": str(repair_exc), "error_type": type(repair_exc).__name__})
            
            # FALLBACK: Create a minimal but valid split
            fallback_subtasks = [
                {"id": 1, "description": "Analyze requirements and constraints"},
                {"id": 2, "description": "Design solution architecture"},
                {"id": 3, "description": "Implement core functionality"},
                {"id": 4, "description": "Test and validate solution"},
                {"id": 5, "description": "Deploy and document"}
            ]
            
            _logger.warning("Using fallback subtasks", extra={"user_goal": user_goal})
            
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
