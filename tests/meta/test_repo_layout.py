"""
Meta tests: verify project structure and imports.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


def test_required_files_exist():
    """Check that required files exist."""
    required = [
        "langgraph.json",
        "pyproject.toml",
        "src/agents/graph.py",
        "src/agents/state.py",
    ]
    
    for rel_path in required:
        path = Path(rel_path)
        assert path.exists(), f"Required file missing: {rel_path}"


def test_can_import_graph():
    """Verify we can import the application graph."""
    try:
        from agents.graph import graph
        assert graph is not None
        assert callable(getattr(graph, "invoke", None)) or callable(getattr(graph, "stream", None))
    except ImportError as e:
        pytest.fail(f"Failed to import graph: {e}")


def test_can_import_state():
    """Verify we can import the state definition."""
    try:
        from agents.state import AppState
        assert AppState is not None
    except ImportError as e:
        pytest.fail(f"Failed to import AppState: {e}")


def test_python_version():
    """Ensure we're running on Python 3.11+."""
    assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version_info}"
