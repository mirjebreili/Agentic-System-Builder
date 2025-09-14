from __future__ import annotations

from pathlib import Path


def find_prompts_dir() -> Path:
    """Search for the prompts directory."""
    if (Path.cwd() / "prompts").exists():
        return Path.cwd() / "prompts"
    if (Path(__file__).resolve().parents[3] / "prompts").exists():
        return Path(__file__).resolve().parents[3] / "prompts"
    raise FileNotFoundError("Could not find the 'prompts' directory.")
