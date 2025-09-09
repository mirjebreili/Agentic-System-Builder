from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

class ConfidenceWeights(BaseModel):
    """Weights for blending confidence scores."""
    conf_w_self: float = 0.4
    conf_w_struct: float = 0.25
    conf_w_cov: float = 0.2
    conf_w_prior: float = 0.15

class Settings(BaseSettings):
    """
    Application settings loaded from .env file and environment variables.
    """
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

    # LLM Settings from .env
    openai_base_url: str = Field(default="http://localhost:11434/v1", alias="OPENAI_BASE_URL")
    openai_api_key: str = Field(default="NA", alias="OPENAI_API_KEY")
    model: str = Field(default="qwen2.5:7b-instruct", alias="MODEL")
    temperature: float = Field(default=0.0, alias="TEMPERATURE")

    # Planning / ToT Settings from .env
    tot_gate: bool = Field(default=True, alias="TOT_GATE")
    tot_branches: int = Field(default=3, alias="TOT_BRANCHES")
    planner_confidence_threshold: float = Field(default=0.6, alias="PLANNER_CONFIDENCE_THRESHOLD")

    # Executor Settings from .env
    exec_max_iters: int = Field(default=6, alias="EXEC_MAX_ITERS")
    reflexion: bool = Field(default=True, alias="REFLEXION")
    reflexion_max_retries: int = Field(default=1, alias="REFLEXION_MAX_RETRIES")

    # HITL Settings from .env
    require_plan_approval: bool = Field(default=True, alias="REQUIRE_PLAN_APPROVAL")

    # Confidence Weights (not from .env, using defaults)
    confidence_weights: ConfidenceWeights = Field(default_factory=ConfidenceWeights)


# Single, shared instance of the settings
settings = Settings()
