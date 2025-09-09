from __future__ import annotations
from functools import lru_cache
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    # LLM
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, env="OPENAI_BASE_URL")
    model: str = Field(default="gpt-4o-mini", env="MODEL")
    temperature: float = Field(default=0.0, env="TEMPERATURE")

    # Executor
    exec_max_iters: int = Field(default=6, env="EXEC_MAX_ITERS")
    reflexion: bool = Field(default=True, env="REFLEXION")
    reflexion_max_retries: int = Field(default=1, env="REFLEXION_MAX_RETRIES")

    # ToT / planning
    tot_gate: bool = Field(default=True, env="TOT_GATE")
    tot_branches: int = Field(default=3, env="TOT_BRANCHES")
    planner_confidence_threshold: float = Field(default=0.6, env="PLANNER_CONFIDENCE_THRESHOLD")

    # Confidence weights
    conf_w_self: float = Field(default=0.4, env="CONF_W_SELF")
    conf_w_struct: float = Field(default=0.25, env="CONF_W_STRUCT")
    conf_w_cov: float = Field(default=0.2, env="CONF_W_COV")
    conf_w_prior: float = Field(default=0.15, env="CONF_W_PRIOR")

    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache
def get_settings() -> Settings:
    return Settings()
