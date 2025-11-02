from __future__ import annotations
from typing import Dict, Any, Callable
from langgraph.graph import StateGraph, END
from agents.state import AppState
from agents.plugin_analyzer import extract_system_elements
from agents.splitter import split_task
from agents.planner import plan_tot
from agents.confidence import compute_plan_confidence
from agents.hitl import review_plan
from agents.formatter import format_plan_order

# New nodes
from agents.goal_extractor import extract_goal
from agents.dependency_resolver import resolve_dependencies
from agents.plan_validator import validate_plan

# State validation
from agents.state_validator import StateValidator

from langfuse.langchain import CallbackHandler
from utils.logger import get_logger

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
    """Route after review: either format_plan_order if approved or back to plan_tot if replan needed."""
    return "plan_tot" if state.get("replan") else "format_plan_order"


def route_after_validation(state: Dict[str, Any]) -> str:
    """Route after plan validation: to confidence if valid, back to planner if needs replan."""
    validation = state.get("plan_validation", {})
    if validation.get("needs_replan", False):
        return "plan_tot"
    return "confidence"


def _make_graph():
    """Create the LangGraph state graph with integrated validation and new capabilities."""
    g = StateGraph(AppState)
    
    # Add existing nodes with validation
    g.add_node("extract_system_elements", validate_state_wrapper(extract_system_elements, "extract_system_elements"))
    g.add_node("split_task", validate_state_wrapper(split_task, "split_task"))
    g.add_node("plan_tot", validate_state_wrapper(plan_tot, "plan_tot"))
    g.add_node("confidence", validate_state_wrapper(compute_plan_confidence, "confidence"))
    g.add_node("review_plan", validate_state_wrapper(review_plan, "review_plan"))
    g.add_node("format_plan_order", validate_state_wrapper(format_plan_order, "format_plan_order"))
    
    # Add new nodes with validation for enhanced workflow
    g.add_node("extract_goal", validate_state_wrapper(extract_goal, "extract_goal"))
    g.add_node("resolve_dependencies", validate_state_wrapper(resolve_dependencies, "resolve_dependencies"))
    g.add_node("validate_plan", validate_state_wrapper(validate_plan, "validate_plan"))

    # Define enhanced workflow edges
    g.set_entry_point("extract_goal")
    
    # Goal extraction → System elements extraction
    g.add_edge("extract_goal", "extract_system_elements")
    
    # System elements → Dependency resolution (if has system elements)
    g.add_edge("extract_system_elements", "resolve_dependencies")
    
    # Dependencies → Task splitting
    g.add_edge("resolve_dependencies", "split_task")
    
    # Task splitting → Main planning
    g.add_edge("split_task", "plan_tot")
    
    # Planning → Validation
    g.add_edge("plan_tot", "validate_plan")
    
    # Validation → Confidence (if valid) or back to planning (if needs replan)
    g.add_conditional_edges(
        "validate_plan",
        route_after_validation,
        {"plan_tot": "plan_tot", "confidence": "confidence"},
    )
    
    # Confidence → Review
    g.add_edge("confidence", "review_plan")
    
    # Review → Format (if approved) or back to planning (if replan)
    g.add_conditional_edges(
        "review_plan",
        route_after_review,
        {"plan_tot": "plan_tot", "format_plan_order": "format_plan_order"},
    )
    
    # Format → End
    g.add_edge("format_plan_order", END)
    
    # Compile with interrupt for HITL review
    return g.compile(interrupt_before=["review_plan"])


graph = _make_graph().with_config({"callbacks": [lf_handler]})
