"""LLM client factory package."""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from .client import get_chat_model, run_llm


def get_llm(**overrides: Any) -> ChatOpenAI:
    """Return the configured chat model instance.

    This is a public alias used by async nodes that expect a ``get_llm``
    factory function when importing :mod:`asb.llm`.
    """

    return get_chat_model(**overrides)


__all__ = ["get_chat_model", "get_llm", "run_llm"]
