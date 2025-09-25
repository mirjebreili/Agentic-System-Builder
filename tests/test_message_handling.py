"""Test message handling utilities work with both LangChain objects and dicts."""

from langchain_core.messages import AIMessage, HumanMessage

from asb.utils.message_utils import (
    extract_last_message_content,
    extract_user_messages_content,
    safe_message_access,
)


def test_extract_last_message_content_langchain():
    """Test with LangChain message objects."""
    messages = [
        HumanMessage(content="Hello world"),
        AIMessage(content="Hi there"),
    ]

    result = extract_last_message_content(messages)
    assert result == "Hi there"


def test_extract_last_message_content_dict():
    """Test with dictionary format messages."""
    messages = [
        {"role": "user", "content": "Hello world"},
        {"role": "assistant", "content": "Hi there"},
    ]

    result = extract_last_message_content(messages)
    assert result == "Hi there"


def test_extract_last_message_content_empty():
    """Test with empty messages list."""
    result = extract_last_message_content([], "default")
    assert result == "default"


def test_extract_user_messages_content():
    """Test extracting only user message content."""
    messages = [
        HumanMessage(content="Question 1"),
        AIMessage(content="Answer 1"),
        HumanMessage(content="Question 2"),
    ]

    result = extract_user_messages_content(messages)
    assert result == ["Question 1", "Question 2"]


def test_safe_message_access_langchain():
    """Test safe access to LangChain message fields."""
    message = HumanMessage(content="test content")

    assert safe_message_access(message, "content") == "test content"
    assert safe_message_access(message, "type") == "human"
    assert safe_message_access(message, "nonexistent", "default") == "default"


def test_safe_message_access_dict():
    """Test safe access to dictionary message fields."""
    message = {"role": "user", "content": "test content"}

    assert safe_message_access(message, "content") == "test content"
    assert safe_message_access(message, "role") == "user"
    assert safe_message_access(message, "nonexistent", "default") == "default"


def test_mixed_message_formats():
    """Test handling mixed LangChain and dict formats."""
    messages = [
        {"role": "user", "content": "Dict message"},
        HumanMessage(content="LangChain message"),
    ]

    result = extract_last_message_content(messages)
    assert result == "LangChain message"
