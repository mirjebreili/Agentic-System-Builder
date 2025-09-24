from __future__ import annotations
from typing import Any, Dict

from langchain_core.messages import HumanMessage
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


def run_llm(prompt: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the configured chat model with a single human prompt."""

    llm = get_chat_model()
    response = llm.invoke([HumanMessage(prompt)])
    content = getattr(response, "content", response)
    if isinstance(content, str):
        output_text = content
    else:
        output_text = str(content)
    return {"output_text": output_text}
