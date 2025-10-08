"""
Top-level LangGraph entrypoint wrapper.

The LangGraph CLI looks for a callable named `get_graph` importable from the project
root. This module provides a small, lazy wrapper that imports the real implementation
from `src.graph.graph` only when `get_graph` is called. That avoids import-time
errors when langgraph isn't installed and makes it simple for `langgraph dev` to
discover the graph by importing `graph.get_graph`.

Usage (from project root):
    langgraph dev

The CLI will import this module and call `get_graph()` to obtain the compiled app.
"""

from typing import Any


def get_graph() -> Any:
    """Return the compiled LangGraph app.

    This function imports the project's real graph implementation on demand so
    importing this module doesn't require langgraph or other heavy deps to be
    installed.
    """
    # Import lazily to avoid raising ImportError at module import time
    from src.graph.graph import get_graph as _get_graph

    return _get_graph()


__all__ = ["get_graph"]
