"""Tests for the attachment preparer applied to compiled graphs."""

from __future__ import annotations

import sys
import types


def _install_langfuse_stubs() -> None:
    """Provide lightweight langfuse stubs so the graph module can import."""

    if "langfuse" in sys.modules:
        return

    langfuse_module = types.ModuleType("langfuse")

    class _Client:
        @staticmethod
        def auth_check() -> bool:  # pragma: no cover - trivial stub
            return False

    def get_client():  # pragma: no cover - trivial stub
        return _Client()

    langfuse_module.get_client = get_client

    langchain_module = types.ModuleType("langfuse.langchain")

    class CallbackHandler:  # pragma: no cover - trivial stub
        pass

    langchain_module.CallbackHandler = CallbackHandler

    langfuse_module.langchain = langchain_module

    sys.modules["langfuse"] = langfuse_module
    sys.modules["langfuse.langchain"] = langchain_module


_install_langfuse_stubs()

from asb.agent.graph import _apply_attachment_preparer


class DummyGraph:
    def __init__(self):
        self.invocations = []

    def invoke(self, state, *_, **__):
        self.invocations.append(state)
        return state

    async def ainvoke(self, state, *_, **__):
        self.invocations.append(state)
        return state

    def batch(self, states, *_, **__):
        self.invocations.append(states)
        return states

    async def abatch(self, states, *_, **__):
        self.invocations.append(states)
        return states


def _make_state(text: str, attachment_text: str):
    return {
        "input_text": text,
        "messages": [{"role": "user", "content": text}],
        "attachments": [
            {
                "type": "file",
                "data": attachment_text.encode("utf-8"),
                "mime_type": "text/plain",
            }
        ],
    }


def test_apply_attachment_preparer_wraps_invoke():
    dummy = DummyGraph()
    wrapped = _apply_attachment_preparer(dummy)

    result = wrapped.invoke(_make_state("Question?", "Attachment contents"))

    assert wrapped is dummy
    assert getattr(dummy, "_attachment_preparer_applied") is True
    assert result["input_text"].endswith("Attachment contents")
    assert dummy.invocations[0]["messages"][0]["content"][1]["mime_type"] == "text/plain"


def test_apply_attachment_preparer_wraps_batch():
    dummy = DummyGraph()
    wrapped = _apply_attachment_preparer(dummy)

    batch_input = [_make_state("Prompt", "Details"), _make_state("Prompt 2", "More details")]
    wrapped.batch(batch_input)

    assert len(dummy.invocations) == 1
    first, second = dummy.invocations[0]
    assert first["input_text"].endswith("Details")
    assert second["input_text"].endswith("More details")
