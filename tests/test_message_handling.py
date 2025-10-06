"""Test message handling utilities work with both LangChain objects and dicts."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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


def test_extract_user_messages_content_dict_variants():
    messages = [
        {"role": "USER", "content": "Upper"},
        {"role": "assistant", "content": "ignored"},
        {"role": "Human", "content": "Case"},
    ]

    assert extract_user_messages_content(messages) == ["Upper", "Case"]


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


def test_extract_last_message_content_string_fallback():
    messages = ["raw string", SystemMessage(content="system"), HumanMessage(content="human")]
    assert extract_last_message_content(messages) == "human"


def test_safe_message_access_handles_missing_roles():
    message = {"content": "payload"}
    assert safe_message_access(message, "role", "default") == "default"


def test_extract_last_message_content_handles_block_list():
    messages = [
        AIMessage(content="assistant"),
        HumanMessage(
            content=[
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ]
        ),
    ]

    assert extract_last_message_content(messages) == "Hello\nWorld"


def test_extract_message_content_with_text_attachments(tmp_path):
    attachment_path = tmp_path / "note.txt"
    attachment_path.write_text("File body", encoding="utf-8")

    inline_attachment = {
        "type": "file",
        "data": b"Inline bytes",
        "mime_type": "text/markdown",
    }
    path_attachment = {"type": "file", "file_path": str(attachment_path), "mime_type": "text/plain"}
    non_text = {"type": "file", "data": b"\x89PNG", "mime_type": "image/png"}

    messages = [
        HumanMessage(content="Earlier"),
        HumanMessage(
            content=[
                "Lead",  # raw text
                {"type": "text", "text": "Block"},
                inline_attachment,
                path_attachment,
                non_text,
            ]
        ),
    ]

    expected = "Lead\nBlock\nInline bytes\nFile body"
    assert extract_last_message_content(messages) == expected

    user_contents = extract_user_messages_content(messages)
    assert user_contents == ["Earlier", expected]


def test_non_text_attachments_are_skipped():
    messages = [
        HumanMessage(
            content=[
                {"type": "file", "data": b"\x89PNG", "mime_type": "image/png"},
                {"type": "text", "text": "Readable"},
            ]
        )
    ]

    assert extract_last_message_content(messages) == "Readable"
    assert extract_user_messages_content(messages) == ["Readable"]
