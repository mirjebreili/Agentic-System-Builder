from __future__ import annotations

from pathlib import Path


def find_prompts_dir() -> Path:
    """Search for the prompts directory."""
    # Look for prompts directory relative to src
    src_dir = Path(__file__).resolve().parent.parent
    prompts_dir = src_dir / "prompts"
    if prompts_dir.exists():
        return prompts_dir
    
    # Fallback to current working directory
    if (Path.cwd() / "prompts").exists():
        return Path.cwd() / "prompts"
    
    raise FileNotFoundError("Could not find the 'prompts' directory.")
