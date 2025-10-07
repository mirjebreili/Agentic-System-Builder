"""Helper utilities for language model access."""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI


def get_llm() -> ChatOpenAI:
    """Return the configured OpenAI chat model."""

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return ChatOpenAI(model="gpt-4", temperature=0, api_key=api_key)
    return ChatOpenAI(model="gpt-4", temperature=0)
