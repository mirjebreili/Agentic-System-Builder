import pytest
from src.graph.graph import get_graph

# This is the example message provided in the task.
# It contains a question in Persian and two plugin descriptions.
TEST_MESSAGE = """
«مجموع مقادیر انتهایی کلیدهایی که با price_ شروع میشن رو بده.»
---
{
    "name": "HttpBasedAtlasReadByKey",
    "description": "A producer that reads from a remote service and yields a stream of objects.",
    "role": "producer",
    "inputs": {"key": {"type": "string"}},
    "outputs": {"type": "stream/json", "path": "data.result.*", "keys_field": "keys"}
}
{
    "name": "membasedAtlasKeyStreamAggregator",
    "description": "A consumer/transformer that aggregates a stream of keys.",
    "role": "consumer",
    "inputs": {"name": {"type": "string"}, "index": {"type": "integer"}},
    "outputs": {"type": "number"}
}
"""

def test_planner_minimal():
    """
    Tests the full planning pipeline with a minimal example.
    It checks if the top-ranked plan is the correct one for the given task.
    """
    # Get the compiled graph
    app = get_graph()

    # Define the input for the graph
    inputs = {"first_message": TEST_MESSAGE}

    # Run the graph
    final_state = app.invoke(inputs)

    # Extract the planner output from the final state
    planner_output = final_state.get("planner_output", {})
    candidates = planner_output.get("candidates", [])

    # Ensure that at least one candidate was generated
    assert len(candidates) > 0, "No candidates were generated."

    # The top candidate should be the first one in the sorted list
    top_plan = candidates[0]

    # Assert that the top plan is the one we expect
    expected_plan = ["HttpBasedAtlasReadByKey", "membasedAtlasKeyStreamAggregator"]
    assert top_plan["plan"] == expected_plan, f"Expected plan {expected_plan}, but got {top_plan['plan']}"

    # Check that the scores and confidence are present
    assert "scores" in top_plan
    assert "confidence" in top_plan
    assert top_plan["confidence"] > 0