"""
Unit tests for LLM client functionality.
"""
from __future__ import annotations

import pytest
from unittest.mock import Mock, patch


def test_get_chat_model_default():
    """Test get_chat_model returns ChatOpenAI instance."""
    from llm.client import get_chat_model
    
    model = get_chat_model()
    
    assert model is not None
    assert hasattr(model, 'invoke')
    assert hasattr(model, 'ainvoke')


def test_get_chat_model_with_overrides():
    """Test get_chat_model accepts override parameters."""
    from llm.client import get_chat_model
    
    model = get_chat_model(temperature=0.5, max_retries=5)
    
    assert model is not None
    # Check that overrides are applied
    assert model.temperature == 0.5
    assert model.max_retries == 5


def test_get_chat_model_uses_settings():
    """Test that get_chat_model uses config settings."""
    from llm.client import get_chat_model
    from config.settings import settings
    
    model = get_chat_model()
    
    # Should use settings from config
    assert model.model_name == settings.LLM_MODEL
    assert model.temperature == settings.TEMPERATURE


@patch('llm.client.ChatOpenAI')
def test_run_llm_basic(mock_chat_openai):
    """Test run_llm function with mocked LLM."""
    from llm.client import run_llm
    
    # Setup mock
    mock_instance = Mock()
    mock_instance.invoke.return_value = Mock(content="Test response")
    mock_chat_openai.return_value = mock_instance
    
    state = {}
    result = run_llm("Test prompt", state)
    
    assert "output_text" in result
    assert result["output_text"] == "Test response"
    mock_instance.invoke.assert_called_once()


def test_llm_with_fake_fixture(fake_llm):
    """Test using the fake_llm fixture."""
    from langchain_core.messages import HumanMessage
    
    messages = [HumanMessage(content="Test message")]
    response = fake_llm.invoke(messages)
    
    assert response is not None
    assert len(fake_llm.prompts) == 1
    assert fake_llm.prompts[0] == messages


@pytest.mark.asyncio
async def test_llm_async_with_fake_fixture(fake_llm):
    """Test async LLM invocation with fake fixture."""
    from langchain_core.messages import HumanMessage
    
    messages = [HumanMessage(content="Async test message")]
    response = await fake_llm.ainvoke(messages)
    
    assert response is not None
    assert len(fake_llm.prompts) == 1
    assert fake_llm.prompts[0] == messages


def test_llm_records_multiple_prompts(fake_llm):
    """Test that fake_llm records all prompts."""
    from langchain_core.messages import HumanMessage
    
    messages1 = [HumanMessage(content="First")]
    messages2 = [HumanMessage(content="Second")]
    messages3 = [HumanMessage(content="Third")]
    
    fake_llm.invoke(messages1)
    fake_llm.invoke(messages2)
    fake_llm.invoke(messages3)
    
    assert len(fake_llm.prompts) == 3
    assert fake_llm.prompts[0] == messages1
    assert fake_llm.prompts[1] == messages2
    assert fake_llm.prompts[2] == messages3


def test_llm_connection_settings():
    """Test that LLM is configured with correct connection settings."""
    from llm.client import get_chat_model
    from config.settings import settings
    
    model = get_chat_model()
    
    # Verify connection settings
    assert model.openai_api_base == settings.LLM_BASE_URL
    assert model.openai_api_key is not None
    # Note: timeout and max_retries are set during initialization
    # They may not be directly accessible as attributes depending on LangChain version


def test_llm_handles_empty_response():
    """Test LLM handling of empty or None responses."""
    from llm.client import run_llm
    
    with patch('llm.client.get_chat_model') as mock_get_model:
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="")
        mock_get_model.return_value = mock_llm
        
        result = run_llm("Test prompt", {})
        
        # Should handle empty content gracefully
        assert "output_text" in result
        assert result["output_text"] == ""


def test_llm_temperature_setting():
    """Test that temperature is correctly set."""
    from llm.client import get_chat_model
    from config.settings import settings
    
    model = get_chat_model()
    
    # Should use configured temperature
    assert model.temperature == settings.TEMPERATURE
    
    # Test override
    model_with_override = get_chat_model(temperature=0.9)
    assert model_with_override.temperature == 0.9
