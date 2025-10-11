from __future__ import annotations

from typing import Annotated, Any, Dict, List
import operator
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class AppState(TypedDict, total=False):
    # Conversation/history
    messages: Annotated[List[AnyMessage], add_messages]

    # Core inputs
    goal: str
    input_text: str
    question: str  # Persian question about data operations
    
    # Plugin information
    plugins: Annotated[List[Dict[str, Any]], operator.add]  # Available plugins
    plugin_sequence: str  # Determined sequence like "Plugin1 --> Plugin2"
    
    # Planning
    plan: Annotated[Dict[str, Any], operator.or_]

    # HITL review state
    review: Annotated[Dict[str, Any], operator.or_]
    replan: bool

    # Diagnostics
    error: str
    errors: Annotated[List[str], operator.add]

    # Flexible scratchpad for intermediate values
    scratch: Annotated[Dict[str, Any], operator.or_]


