"""Tests for preparing the initial state with file attachments."""

from __future__ import annotations

from pathlib import Path

from asb.utils.state_preparer import prepare_initial_state


def _make_attachment(tmp_path: Path, name: str, content: str) -> Path:
    file_path = tmp_path / name
    file_path.write_text(content, encoding="utf-8")
    return file_path


def test_prepare_initial_state_appends_attachments_to_existing_message(tmp_path):
    file_path = _make_attachment(tmp_path, "notes.txt", "Important context")

    state = {
        "input_text": "Please review the attached notes.",
        "messages": [{"role": "user", "content": "Please review the attached notes."}],
        "attachments": [
            {"type": "file", "data": b"Inline bytes", "mime_type": "text/plain"},
            {"type": "file", "file_path": str(file_path)},
        ],
    }

    prepared = prepare_initial_state(state)

    assert len(prepared["messages"]) == 1
    content = prepared["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0] == "Please review the attached notes."
    # Ensure both attachment entries are included
    assert any(isinstance(entry, dict) and entry.get("data") == b"Inline bytes" for entry in content)
    assert any(
        isinstance(entry, dict) and Path(entry.get("file_path", "")) == file_path for entry in content
    )

    assert "Inline bytes" in prepared["input_text"]
    assert "Important context" in prepared["input_text"]


def test_prepare_initial_state_creates_message_when_missing(tmp_path):
    file_path = _make_attachment(tmp_path, "summary.txt", "Summary details")

    state = {
        "input_text": "Summarize the material.",
        "attachments": [str(file_path)],
    }

    prepared = prepare_initial_state(state)

    assert len(prepared["messages"]) == 1
    message = prepared["messages"][0]
    assert message["role"] == "user"
    assert isinstance(message["content"], list)
    assert message["content"][0] == "Summarize the material."
    assert any(Path(entry.get("file_path", "")) == file_path for entry in message["content"] if isinstance(entry, dict))
    assert "Summary details" in prepared["input_text"]


def test_prepare_initial_state_handles_non_text_attachments():
    state = {
        "input_text": "Look at the binary file.",
        "attachments": [
            {"type": "file", "data": b"\x89PNG", "mime_type": "image/png"},
        ],
    }

    prepared = prepare_initial_state(state)

    assert len(prepared["messages"]) == 1
    message = prepared["messages"][0]
    assert isinstance(message["content"], list)
    assert message["content"][0] == "Look at the binary file."
    assert any(entry.get("mime_type") == "image/png" for entry in message["content"] if isinstance(entry, dict))
    # Non-text attachments should not modify the textual input
    assert prepared["input_text"] == "Look at the binary file."

