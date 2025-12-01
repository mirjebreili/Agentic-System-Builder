from __future__ import annotations
from typing import Dict, Any, Callable
from langgraph.graph import StateGraph, END
from src.agents.state import AppState

# Plan B Refactored Nodes (6 total)
from src.agents.context_extractor import extract_context
from src.agents.planner import plan_tot
from src.agents.plan_validator import validate_plan
from src.agents.hitl import review_plan
from src.agents.format_output import format_output

# State validation
from src.agents.state_validator import StateValidator

from langfuse.langchain import CallbackHandler
from src.utils.logger import get_logger

lf_handler = CallbackHandler()

# Initialize state validator and logger
state_validator = StateValidator()
_logger = get_logger(__name__)


def validate_state_wrapper(node_func: Callable, node_name: str) -> Callable:
    """Wrap a node function with state validation.
    
    Args:
        node_func: The original node function to wrap
        node_name: Name of the node for logging
        
    Returns:
        Wrapped function that validates state before and after execution
    """
    def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        # Validate input state
        errors = state_validator.validate_state(state, strict=False)
        if errors:
            _logger.warning(
                "State validation failed before node execution",
                extra={
                    "node": node_name,
                    "errors": errors,
                }
            )
        
        # Execute the node
        result = node_func(state)
        
        # Validate output state (merged with input)
        output_state = {**state, **(result if result else {})}
        output_errors = state_validator.validate_state(output_state, strict=False)
        if output_errors:
            _logger.warning(
                "State validation failed after node execution",
                extra={
                    "node": node_name,
                    "errors": output_errors,
                }
            )
        
        return result
    
    return wrapper



def route_after_review(state: Dict[str, Any]) -> str:
    """Route after review: either format_output if approved or back to plan_tot if replan needed."""
    return "plan_tot" if state.get("replan") else "format_output"


def route_after_validation(state: Dict[str, Any]) -> str:
    """Route after plan validation: to review_plan if valid, back to planner if needs replan."""
    validation = state.get("plan_validation", {})
    if validation.get("needs_replan", False):
        return "plan_tot"
    return "review_plan"


def _make_graph():
    """
    Create the LangGraph state graph - Plan B Refactored (6 nodes).
    
    Workflow:
        extract_context → plan_tot → validate_plan → review_plan → format_output → END
                                           ↓ (if validation fails)
                                       plan_tot
                                           
        review_plan → plan_tot (if user requests revision)
        
    Removed nodes:
        - extract_goal (merged into extract_context)
        - recognize_plugin_pattern (merged into extract_context)
        - extract_system_elements (merged into extract_context)
        - split_task (no longer needed, planner has full creative freedom)
        - resolve_dependencies (not needed for planning phase)
        - confidence (merged into format_output)
        - format_plan_order (renamed to format_output with confidence)
    """
    g = StateGraph(AppState)
    
    # Add Plan B nodes with validation wrappers
    g.add_node("extract_context", validate_state_wrapper(extract_context, "extract_context"))
    g.add_node("plan_tot", validate_state_wrapper(plan_tot, "plan_tot"))
    g.add_node("validate_plan", validate_state_wrapper(validate_plan, "validate_plan"))
    g.add_node("review_plan", validate_state_wrapper(review_plan, "review_plan"))
    g.add_node("format_output", validate_state_wrapper(format_output, "format_output"))

    # Define workflow edges
    g.set_entry_point("extract_context")
    
    # Context extraction → Planning
    g.add_edge("extract_context", "plan_tot")
    
    # Planning → Validation
    g.add_edge("plan_tot", "validate_plan")
    
    # Validation → Review (if valid) or back to Planning (if needs replan)
    g.add_conditional_edges(
        "validate_plan",
        route_after_validation,
        {"plan_tot": "plan_tot", "review_plan": "review_plan"},
    )
    
    # Review → Format (if approved) or back to Planning (if revision requested)
    g.add_conditional_edges(
        "review_plan",
        route_after_review,
        {"plan_tot": "plan_tot", "format_output": "format_output"},
    )
    
    # Format → End
    g.add_edge("format_output", END)
    
    # Compile with interrupt for HITL review
    return g.compile()


graph = _make_graph().with_config({"callbacks": [lf_handler]})
