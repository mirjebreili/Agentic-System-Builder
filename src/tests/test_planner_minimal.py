import pytest
from src.graph.graph import get_graph

def test_planner_minimal():
    """
    Tests the full graph with a synthetic first message to ensure the
    planner produces the correct top candidate.
    """
    # 1. Build the synthetic "first message"
    question = "مجموع مقادیر انتهایی کلیدهایی که با price_ شروع میشن رو بده."
    plugin_doc_1 = "HttpBasedAtlasReadByKey: A producer tool that reads data from an HTTP-based Atlas service by key."
    plugin_doc_2 = "membasedAtlasKeyStreamAggregator: A consumer/transformer that aggregates a stream of keys."

    first_message = f"{question}\n---\n{plugin_doc_1}\n{plugin_doc_2}"

    # 2. Get the compiled graph
    app = get_graph()

    # 3. Invoke the graph with the input state
    # The input to a compiled graph is a dictionary.
    final_state = app.invoke({"first_message": first_message})

    # 4. Assert the output structure and top candidate
    assert "planner_output" in final_state
    planner_output = final_state["planner_output"]

    assert "candidates" in planner_output
    assert "chosen" in planner_output

    candidates = planner_output["candidates"]
    chosen_index = planner_output["chosen"]

    # The stubbed LLM returns 2 candidates, so we expect 2 here.
    assert len(candidates) > 0
    # The chosen index should be 0, as it has the higher score.
    assert chosen_index == 0

    top_candidate = candidates[chosen_index]

    # Assert the plan
    expected_plan = ["HttpBasedAtlasReadByKey", "membasedAtlasKeyStreamAggregator"]
    assert top_candidate["plan"] == expected_plan

    # Assert the rationale (from the stubbed LLM)
    expected_rationale = "Fetch Atlas data; aggregate numeric suffixes by prefix."
    assert top_candidate["rationale"] == expected_rationale

    # Assert scores based on our scoring logic
    scores = top_candidate["scores"]
    assert scores["coverage"] >= 0.9
    assert scores["io"] == 1.0
    assert scores["simplicity"] >= 0.5 # 1 / len(plan) = 0.5
    assert scores["constraints"] >= 0.9

    # Assert raw score and confidence
    assert top_candidate["raw_score"] > 0.85
    assert top_candidate["confidence"] > 0.6