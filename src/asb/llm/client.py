from __future__ import annotations
from typing import Any
from langchain_openai import ChatOpenAI
from config.settings import get_settings

def get_chat_model(**overrides: Any) -> ChatOpenAI:
    cfg = get_settings()
    params = {
        "model": cfg.model,
        "api_key": cfg.openai_api_key or "dummy",
        "temperature": cfg.temperature,
        "timeout": 60,
        "max_retries": 2,
    }

    if cfg.openai_base_url:
        params["base_url"] = cfg.openai_base_url

    params.update(overrides)
    return ChatOpenAI(**params)
