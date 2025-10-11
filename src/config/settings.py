from dotenv import load_dotenv
import os
from pydantic_settings import BaseSettings



class Settings(BaseSettings):
    load_dotenv(override=False)
    LLM_BASE_URL: str = os.getenv('LLM_BASE_URL')
    LLM_MODEL: str = os.getenv('LLM_MODEL')
    TEMPERATURE: float = float(os.getenv('TEMPERATURE', '0.0'))
    LLM_API_KEY: str = os.getenv('LLM_API_KEY')
    LANGFUSE_HOST: str = os.getenv('LANGFUSE_HOST')
    LANGFUSE_PUBLIC_KEY: str = os.getenv('LANGFUSE_PUBLIC_KEY')
    LANGFUSE_SECRET_KEY: str = os.getenv('LANGFUSE_SECRET_KEY')
settings = Settings()