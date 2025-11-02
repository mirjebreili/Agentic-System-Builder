"""
Centralized application settings and configuration.

This module provides a single source of truth for all application settings
including LLM configuration, retry settings, timeouts, costs, and monitoring.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class AppSettings(BaseSettings):
    """Application settings with environment variable support."""
    
    # LLM Settings (backward compatible with existing env vars)
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model: str = "gpt-4"  # Also accepts llm_model from env
    default_model: str = "gpt-4"
    max_tokens: int = 4000
    temperature: float = 0.7
    
    # ToT (Tree of Thought) Settings
    tot_gate: int = 1
    tot_branches: int = 3
    
    # Planning Settings
    planner_confidence_threshold: float = 0.7
    confidence_threshold: float = 0.5
    enable_alternative_plans: bool = True
    enable_cost_estimation: bool = True
    enable_plan_validation: bool = True
    
    # Langfuse (tracing/observability) Settings
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: Optional[str] = None
    
    # Retry Settings
    max_retries: int = 3
    retry_min_wait: int = 2
    retry_max_wait: int = 10
    retry_multiplier: int = 2
    
    # Timeout Settings
    node_timeout_seconds: int = 60
    llm_timeout_seconds: int = 30
    default_timeout_seconds: int = 30
    
    # Cost Settings (USD per 1K tokens)
    cost_per_1k_input_tokens: float = 0.003
    cost_per_1k_output_tokens: float = 0.015
    avg_tokens_per_node: int = 500
    avg_output_tokens: int = 200
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_structured_logging: bool = True
    log_level: str = "INFO"
    json_logs: bool = False
    
    # Storage
    feedback_storage_path: str = "data/feedback.jsonl"
    cache_dir: str = "data/cache"
    cache_ttl_hours: int = 24
    
    # Security
    max_prompt_length: int = 10000
    enable_sanitization: bool = True
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 100
    
    # Visualization
    enable_visualization: bool = True
    
    class Config:
        env_file = ".env"
        env_prefix = "ASB_"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields from .env to be ignored


# Global settings instance
settings = AppSettings()
