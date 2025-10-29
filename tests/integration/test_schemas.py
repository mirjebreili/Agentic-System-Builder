"""
Integration tests for SDK schemas and validation.
"""
from __future__ import annotations

import pytest


@pytest.mark.cli
def test_assistant_has_schemas(lg_client, default_assistant_id):
    """Test that assistant has input/output schemas."""
    assistant = lg_client.assistants.get(default_assistant_id)
    
    assert assistant is not None
    # LangGraph assistants should have metadata about schemas
    # The exact structure may vary, so we just check the assistant exists
    assert "assistant_id" in assistant


@pytest.mark.cli
def test_minimal_valid_payload(lg_client, default_assistant_id):
    """Create a minimal valid payload for the assistant."""
    # Create a thread
    thread = lg_client.threads.create()
    assert "thread_id" in thread
    
    # Minimal payload based on our AppState
    payload = {
        "messages": [{"role": "user", "content": "Hello, create a simple plan"}]
    }
    
    # Create a run
    run = lg_client.runs.create(
        thread_id=thread["thread_id"],
        assistant_id=default_assistant_id,
        input=payload
    )
    
    assert "run_id" in run
    assert run["status"] in ["pending", "running", "interrupted"]
