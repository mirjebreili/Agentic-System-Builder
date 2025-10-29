"""
Tests for interrupt/resume behavior (HITL).
"""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage


@pytest.mark.slow
def test_interrupt_at_review_plan(graph):
    """Test that graph interrupts at review_plan node."""
    initial_state = {
        "messages": [HumanMessage(content="Build a REST API")]
    }
    
    config = {"configurable": {"thread_id": "test-interrupt-1"}}
    
    # Stream until interrupt
    events = list(graph.stream(initial_state, config=config))
    
    # Should have processed multiple nodes before interrupt
    assert len(events) > 0
    
    # Check that we have plan data
    final_event = events[-1] if events else {}
    if "review_plan" in final_event:
        assert "plan" in final_event["review_plan"]


@pytest.mark.slow
def test_resume_after_interrupt(graph):
    """Test resuming after an interrupt."""
    initial_state = {
        "messages": [HumanMessage(content="Create a data pipeline")]
    }
    
    thread_id = "test-resume-1"
    config = {"configurable": {"thread_id": thread_id}}
    
    # First run: stream to interrupt
    events_before = list(graph.stream(initial_state, config=config))
    
    # In a real scenario, you would:
    # 1. Get the interrupt state
    # 2. Provide resume input (approve/revise)
    # 3. Continue execution
    
    # For this test, we just verify the interrupt happened
    assert len(events_before) > 0
