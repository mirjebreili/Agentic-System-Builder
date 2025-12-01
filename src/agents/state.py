from __future__ import annotations

from typing import Annotated, Any, Dict, List
import operator
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class AppState(TypedDict, total=False):
    """Application state for the agentic system builder - Plan B Refactored.
    
    Active fields (Plan B):
    - messages: Conversation history
    - goal: User's objective (from extract_context)
    - system_elements: List of component names (from extract_context)
    - has_system_elements: Boolean flag (from extract_context)
    - plugins: Available plugins with metadata (from extract_context)
    - plan: Selected execution plan (from plan_tot)
    - plan_validation: Validation results (from validate_plan)
    - review: HITL review state (from review_plan)
    - replan: Replan trigger flag
    - debug: Debug information
    - trace_id: Request tracing ID
    
    Deprecated fields (kept for compatibility):
    - split_tasks: No longer populated (split_task node removed)
    - plugin_definition_pattern: No longer populated (pattern_recognizer removed)
    - pattern_confidence: No longer populated (pattern_recognizer removed)
    - plugin_dependencies: No longer populated (dependency_resolver removed)
    - dependency_order: No longer populated (dependency_resolver removed)
    - dependency_validation: No longer populated (dependency_resolver removed)
    - alternative_plans: Stored in debug.plan_candidates instead
    """
    
    # Conversation history
    messages: Annotated[List[AnyMessage], add_messages]

    # Core inputs (ACTIVE - from extract_context)
    goal: str
    system_elements: List[str]  # Component names extracted from user input
    has_system_elements: bool  # Whether concrete system components were provided
    plugins: Annotated[List[Dict[str, Any]], operator.add]  # Available plugins with metadata
    
    # DEPRECATED - Pattern recognition (recognize_plugin_pattern node removed)
    plugin_definition_pattern: Dict[str, Any]  # No longer populated
    pattern_confidence: float  # No longer populated
    
    # DEPRECATED - Task splitting (split_task node removed)
    split_tasks: List[Dict[str, Any]]  # No longer populated
    
    # DEPRECATED - Dependency resolution (resolve_dependencies node removed)
    plugin_dependencies: Dict[str, List[str]]  # No longer populated
    dependency_order: List[str]  # No longer populated
    dependency_validation: Dict[str, Any]  # No longer populated
    dependency_errors: List[str]  # No longer populated
    
    # Planning (ACTIVE - from plan_tot)
    plan: Annotated[Dict[str, Any], operator.or_]
    
    # DEPRECATED - Alternative plans now in debug.plan_candidates
    alternative_plans: List[Dict[str, Any]]
    has_alternatives: bool

    # Validation (ACTIVE - from validate_plan)
    plan_validation: Dict[str, Any]  # Plan validation results
    input_validated: bool  # Whether input passed security checks
    
    # Cost estimation (optional feature)
    cost_estimate: Dict[str, Any]
    
    # Visualization (optional feature)
    plan_visualizations: Dict[str, Any]

    # HITL review state (ACTIVE - from review_plan)
    review: Annotated[Dict[str, Any], operator.or_]
    replan: bool

    # Debug information (ACTIVE - all plan candidates, confidence breakdown)
    debug: Annotated[Dict[str, Any], operator.or_]
    
    # Tracking (ACTIVE)
    trace_id: str
    _request_count: int


