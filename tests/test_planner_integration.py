"""Integration test to ensure planner works with real LangChain messages."""

from langchain_core.messages import HumanMessage

from asb.agent.planner import plan_tot


def test_planner_with_langchain_messages():
    """Test planner works with actual LangChain message objects."""

    state = {
        "messages": [
            HumanMessage(content="Create a summarizer app that makes bullet points"),
        ],
        "goal": "Create a summarizer app",
    }

    result = plan_tot(state)

    assert isinstance(result, dict)
    assert "plan" in result
    assert not result.get("error")


def test_planner_with_empty_messages():
    """Test planner handles empty messages gracefully."""

    state = {
        "messages": [],
        "goal": "Test goal",
    }

    result = plan_tot(state)

    assert isinstance(result, dict)
    assert "plan" in result
    assert not result.get("error")
