import pytest

from asb.agent.scaffold import generate_enhanced_state_schema


def test_state_schema_has_proper_aggregators():
    """Generated schema should include safe aggregators for mergeable fields."""

    schema_code = generate_enhanced_state_schema({})

    assert "Annotated[List[AnyMessage], add_messages]" in schema_code
    assert "Annotated[Dict[str, Any], operator.or_]" in schema_code
    assert "import operator" in schema_code

    lines = schema_code.split("\n")
    dict_lines = [line for line in lines if "Dict[str, Any]" in line and "Annotated" not in line]

    allowed_plain = {"goal: str", "input_text: str", "result: str", "final_output: str", "error: str"}

    for line in dict_lines:
        clean_line = line.strip()
        if clean_line and clean_line not in allowed_plain:
            pytest.fail(f"Found Dict[str, Any] without Annotated aggregator: {clean_line}")
