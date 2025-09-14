from __future__ import annotations
from typing import Any
from langchain_openai import ChatOpenAI
from asb_cfg.settings_v2 import get_settings

def get_chat_model(**overrides: Any) -> ChatOpenAI:
    cfg = get_settings()
    params = {
        "model": cfg.model,
        "base_url": cfg.openai_base_url,
        "api_key": cfg.openai_api_key or None,
        "temperature": cfg.temperature,
        "timeout": 60,
        "max_retries": 2,
    }
    params.update(overrides)
    return ChatOpenAI(**params)
