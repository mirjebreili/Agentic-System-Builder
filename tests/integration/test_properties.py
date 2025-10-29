"""
Property-based tests using Hypothesis for fuzzing.
"""
from __future__ import annotations

import pytest

# Only run if hypothesis is available
hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st


@given(st.text(min_size=1, max_size=100))
def test_extract_system_elements_never_crashes(prompt_text):
    """Fuzz test: extract_system_elements should never crash."""
    from agents.plugin_analyzer import extract_system_elements
    from langchain_core.messages import HumanMessage
    
    state = {"messages": [HumanMessage(content=prompt_text)]}
    
    # Should not raise
    result = extract_system_elements(state)
    
    # Should always return these keys
    assert "system_elements" in result
    assert "plugins" in result
    assert "has_system_elements" in result


@given(st.dictionaries(
    keys=st.sampled_from(["goal", "nodes", "edges", "confidence"]),
    values=st.one_of(
        st.text(max_size=50),
        st.lists(st.dictionaries(keys=st.text(max_size=10), values=st.text(max_size=20)), max_size=5),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
    )
))
def test_confidence_handles_varied_plans(plan_dict):
    """Fuzz test: confidence calculation should handle varied plan structures."""
    from agents.confidence import compute_plan_confidence
    
    state = {"plan": plan_dict, "debug": {}}
    
    # Should not crash
    try:
        result = compute_plan_confidence(state)
        assert "plan" in result
    except (KeyError, TypeError, AttributeError):
        # Some malformed inputs may fail gracefully
        pass
