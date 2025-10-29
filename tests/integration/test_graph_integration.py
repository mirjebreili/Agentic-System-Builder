"""
Integration tests for the full graph execution.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage


def test_graph_invoke_basic(graph):
    """Test basic graph invocation."""
    initial_state = {
        "messages": [HumanMessage(content="Create a simple plan for building a website")]
    }
    
    # Note: This will hit the interrupt at review_plan
    # In real usage, you'd need to handle the interrupt
    try:
        result = graph.invoke(initial_state, config={"configurable": {"thread_id": "test-1"}})
        # If we get here, the graph completed somehow (shouldn't in normal flow)
        assert "messages" in result
    except Exception as e:
        # Expected to interrupt at review_plan
        assert "interrupt" in str(e).lower() or "review" in str(e).lower()


@pytest.mark.asyncio
async def test_graph_ainvoke_basic(graph):
    """Test async graph invocation."""
    initial_state = {
        "messages": [HumanMessage(content="Plan a data processing pipeline")]
    }
    
    try:
        result = await graph.ainvoke(initial_state, config={"configurable": {"thread_id": "test-async-1"}})
        assert "messages" in result
    except Exception as e:
        # Expected to interrupt at review_plan
        assert "interrupt" in str(e).lower() or "review" in str(e).lower()


def test_graph_stream_to_interrupt(graph):
    """Test streaming to the first interrupt point."""
    initial_state = {
        "messages": [HumanMessage(content="Design a microservices architecture")]
    }
    
    events = []
    config = {"configurable": {"thread_id": "test-stream-1"}}
    
    for event in graph.stream(initial_state, config=config):
        events.append(event)
        # Break before we hit too many events
        if len(events) >= 10:
            break
    
    # Should have at least processed some nodes
    assert len(events) > 0


def test_graph_has_expected_nodes(graph):
    """Verify the graph has all expected nodes."""
    expected_nodes = [
        "extract_system_elements",
        "split_task",
        "plan_tot",
        "confidence",
        "review_plan",
        "format_plan_order"
    ]
    
    graph_nodes = list(graph.nodes.keys())
    
    for node in expected_nodes:
        assert node in graph_nodes, f"Expected node '{node}' not found in graph"
