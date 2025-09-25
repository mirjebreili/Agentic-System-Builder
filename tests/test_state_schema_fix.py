"""Regression tests for state schema aggregator safety."""

from __future__ import annotations

import pytest

from asb.agent.scaffold import generate_enhanced_state_schema


def test_state_schema_has_proper_aggregators() -> None:
    """Test that generated state schema prevents InvalidUpdateError."""

    schema_code = generate_enhanced_state_schema({})

    # Check for proper aggregators
    assert "Annotated[List[AnyMessage], add_messages]" in schema_code
    assert "Annotated[Dict[str, Any], operator.or_]" in schema_code
    assert "import operator" in schema_code

    # Ensure it's not using plain Dict[str, Any] for mergeable fields
    lines = schema_code.split("\n")
    dict_lines = [line for line in lines if "Dict[str, Any]" in line and "Annotated" not in line]

    # Only goal, input_text, result, final_output, error should use plain types
    allowed_plain = {"goal: str", "input_text: str", "result: str", "final_output: str", "error: str"}

    for line in dict_lines:
        clean_line = line.strip()
        if clean_line and clean_line not in allowed_plain:
            pytest.fail(f"Found Dict[str, Any] without Annotated aggregator: {clean_line}")
