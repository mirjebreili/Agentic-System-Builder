import pytest
from langchain_core.messages import HumanMessage, AIMessage
from src.graph.graph import get_graph

@pytest.fixture
def app():
    """Fixture to provide a compiled graph for each test."""
    return get_graph()

@pytest.fixture
def initial_message_content():
    """Fixture to provide the standard initial user message for tests."""
    question = "مجموع مقادیر انتهایی کلیدهایی که با price_ شروع میشن رو بده."
    plugin_doc_1 = "HttpBasedAtlasReadByKey: A producer tool."
    plugin_doc_2 = "membasedAtlasKeyStreamAggregator: A consumer/transformer."
    return f"{question}\n---\n{plugin_doc_1}\n{plugin_doc_2}"

def test_hitl_approve_flow(app, initial_message_content):
    """
    Tests the full interactive flow where the user approves a plan.
    """
    # 1. Setup the conversation
    initial_state = {"messages": [HumanMessage(content=initial_message_content)]}
    config = {"configurable": {"thread_id": "test-approve-thread"}}

    # 2. Run to the first interruption
    app.invoke(initial_state, config=config)

    # 3. Simulate user approval and continue execution
    user_approval_message = HumanMessage(content="APPROVE 0")
    app.invoke({"messages": [user_approval_message]}, config=config)

    # 4. Retrieve the final state from the checkpoint
    final_state = app.get_state(config)

    # 5. Assert the final state
    assert "approved_plan" in final_state.values
    assert len(final_state.values["approved_plan"]) > 0
    expected_plan = ["HttpBasedAtlasReadByKey", "membasedAtlasKeyStreamAggregator"]
    assert final_state.values["approved_plan"] == expected_plan
    # The full message history should be preserved in the checkpoint
    assert len(final_state.values["messages"]) >= 3

def test_hitl_revise_flow(app, initial_message_content):
    """
    Tests the flow where the user requests a revision.
    """
    # 1. Setup the conversation
    initial_state = {"messages": [HumanMessage(content=initial_message_content)]}
    config = {"configurable": {"thread_id": "test-revise-thread"}}

    # 2. Run to the first interruption
    app.invoke(initial_state, config=config)

    # 3. Simulate user revision request and continue
    user_revision_message = HumanMessage(content="REVISE Please try a shorter plan.")
    app.invoke({"messages": [user_revision_message]}, config=config)

    # 4. Retrieve the state after the second interruption
    final_interrupted_state = app.get_state(config)

    # 5. Assert the state
    assert "candidates" in final_interrupted_state.values
    # The message history should contain the full conversation
    assert len(final_interrupted_state.values["messages"]) >= 4
    last_message = final_interrupted_state.values["messages"][-1]
    assert isinstance(last_message, AIMessage)
    assert "I have generated the following plan candidates." in last_message.content