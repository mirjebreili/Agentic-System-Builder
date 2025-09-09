import conftest
import prompt2graph.llm.client as llm_client


def test_get_chat_model_returns_fake_instance():
    llm = llm_client.get_chat_model()
    assert isinstance(llm, conftest.FakeChatModel)
    # invoking without special system message should return plan JSON
    content = llm.invoke([]).content
    assert "Test goal" in content
