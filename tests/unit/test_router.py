"""
Unit tests for router/conditional edge logic.
"""
from __future__ import annotations

import pytest


def test_route_after_review_approve():
    """Test routing after plan approval."""
    from agents.graph import route_after_review
    
    state = {"replan": False}
    next_node = route_after_review(state)
    
    assert next_node == "format_plan_order"


def test_route_after_review_revise():
    """Test routing when replan is needed."""
    from agents.graph import route_after_review
    
    state = {"replan": True}
    next_node = route_after_review(state)
    
    assert next_node == "plan_tot"


def test_route_after_review_default():
    """Test routing with missing replan field."""
    from agents.graph import route_after_review
    
    state = {}
    next_node = route_after_review(state)
    
    # Should default to approve (replan=False)
    assert next_node == "format_plan_order"
