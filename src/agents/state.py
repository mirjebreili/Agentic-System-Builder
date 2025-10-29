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
    
    # Planning
    plan: Annotated[Dict[str, Any], operator.or_]

    # HITL review state
    review: Annotated[Dict[str, Any], operator.or_]
    replan: bool

    # Debug information (all plan candidates)
    debug: Annotated[Dict[str, Any], operator.or_]


