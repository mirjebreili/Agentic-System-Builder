from langchain_openai import ChatOpenAI
from prompt2graph.config.settings import settings

def get_chat_model():
    """
    Returns a ChatOpenAI instance configured from the global settings.
    """
    return ChatOpenAI(
        model=settings.model,
        temperature=settings.temperature,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
    )
