"""Language model utilities."""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI


DEFAULT_PLANNER_MODEL = os.getenv("ASB_PLANNER_MODEL", "gpt-4o-mini")


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Instantiate the chat model used by the planner."""

    model = os.getenv("ASB_PLANNER_MODEL", DEFAULT_PLANNER_MODEL)
    temperature = float(os.getenv("ASB_PLANNER_TEMPERATURE", "0"))
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if api_key:
        return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
    return ChatOpenAI(model=model, temperature=temperature)

