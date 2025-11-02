from __future__ import annotations

from typing import Annotated, Any, Dict, List
import operator
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class AppState(TypedDict, total=False):
    """Application state for the agentic system builder."""
    
    # Conversation history
    messages: Annotated[List[AnyMessage], add_messages]

    # Core inputs
    goal: str
    
    # Task splitting
    split_tasks: List[Dict[str, Any]]  # Atomic subtasks from splitter
    system_elements: List[str]  # Existing components/functions mentioned by user
    has_system_elements: bool  # Whether concrete system components were provided
    
    # Plugin information
    plugins: Annotated[List[Dict[str, Any]], operator.add]  # Available plugins
    plugin_dependencies: Dict[str, List[str]]  # Plugin dependency graph
    dependency_order: List[str]  # Topologically sorted plugin order
    dependency_validation: Dict[str, Any]  # Dependency validation results
    
    # Planning
    plan: Annotated[Dict[str, Any], operator.or_]
    alternative_plans: List[Dict[str, Any]]  # Alternative plan strategies
    has_alternatives: bool  # Whether alternatives were generated

    # Validation
    plan_validation: Dict[str, Any]  # Plan validation results
    input_validated: bool  # Whether input passed security checks
    dependency_errors: List[str]  # Dependency resolution errors
    
    # Cost estimation
    cost_estimate: Dict[str, Any]  # Execution cost estimates
    
    # Visualization
    plan_visualizations: Dict[str, Any]  # Visual representations (Mermaid, ASCII, etc.)

    # HITL review state
    review: Annotated[Dict[str, Any], operator.or_]
    replan: bool

    # Debug information (all plan candidates)
    debug: Annotated[Dict[str, Any], operator.or_]
    
    # Tracking
    trace_id: str  # Unique identifier for request tracing
    _request_count: int  # For rate limiting


