import os
from dataclasses import dataclass
from typing import Optional

try:
    # load .env if present
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # dotenv is optional; environment variables may be provided externally
    pass


@dataclass
class Settings:
    OPENAI_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    MODEL: Optional[str] = os.getenv("MODEL", "gpt-4o")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))

    # Planning / ToT
    TOT_GATE: bool = os.getenv("TOT_GATE", "0") in ("1", "true", "True")
    TOT_BRANCHES: int = int(os.getenv("TOT_BRANCHES", "3"))
    PLANNER_CONFIDENCE_THRESHOLD: float = float(
        os.getenv("PLANNER_CONFIDENCE_THRESHOLD", "0.6")
    )


settings = Settings()
