"""
Unit tests for individual graph nodes.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest


def test_extract_system_elements_no_prompt():
    """Test extract_system_elements with no human message."""
    from agents.plugin_analyzer import extract_system_elements
    
    state = {"messages": []}
    result = extract_system_elements(state)
    
    assert "system_elements" in result
    assert "plugins" in result
    assert "has_system_elements" in result
    assert result["has_system_elements"] is False


def test_extract_system_elements_with_json():
    """Test extraction with JSON plugin definition."""
    from agents.plugin_analyzer import extract_system_elements
    from langchain_core.messages import HumanMessage
    
    prompt = """
    Here are my plugins:
    {"plugins": [{"name": "TestPlugin", "goal": "Do something"}]}
    """
    
    state = {"messages": [HumanMessage(content=prompt)]}
    result = extract_system_elements(state)
    
    assert result["has_system_elements"] is True
    assert len(result["plugins"]) > 0
    assert result["plugins"][0]["name"] == "TestPlugin"


def test_split_task_simple_goal():
    """Test task splitting with a simple goal."""
    from agents.splitter import split_task
    from langchain_core.messages import HumanMessage
    
    state = {
        "messages": [HumanMessage(content="Build a simple web app")],
        "has_system_elements": False,
        "system_elements": []
    }
    
    result = split_task(state)
    
    assert "split_tasks" in result
    assert len(result["split_tasks"]) > 0
    assert all("id" in task and "description" in task for task in result["split_tasks"])


def test_compute_plan_confidence():
    """Test confidence calculation."""
    from agents.confidence import compute_plan_confidence
    
    plan = {
        "goal": "test",
        "nodes": [{"id": "node1", "tool": "test_tool"}],
        "edges": [],
        "confidence": 0.8
    }
    
    state = {"plan": plan, "debug": {}}
    result = compute_plan_confidence(state)
    
    assert "plan" in result
    assert "confidence" in result["plan"]
    assert 0.0 <= result["plan"]["confidence"] <= 1.0


def test_review_plan_structure():
    """Test that review_plan has correct structure (without interrupt)."""
    from agents.hitl import review_plan
    
    plan = {
        "goal": "test",
        "nodes": [],
        "edges": [],
        "confidence": 0.9
    }
    
    state = {"plan": plan}
    
    # Note: This will trigger an interrupt in real execution
    # In unit tests, we just verify the function exists and is callable
    assert callable(review_plan)


def test_format_plan_order():
    """Test plan formatting."""
    from agents.formatter import format_plan_order
    from langchain_core.messages import HumanMessage
    
    plan = {
        "goal": "test",
        "nodes": [
            {"id": "step1", "tool": "tool1", "reasoning": "First step"},
            {"id": "step2", "tool": "tool2", "reasoning": "Second step"}
        ],
        "edges": [{"from": "step1", "to": "step2"}],
        "confidence": 0.85
    }
    
    state = {
        "plan": plan,
        "messages": [HumanMessage(content="Test")],
        "debug": {"plan_candidates": [{"confidence": 0.85, "plan": plan}]}
    }
    
    result = format_plan_order(state)
    
    assert "messages" in result
    assert len(result["messages"]) > 0
