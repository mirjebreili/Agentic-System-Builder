from __future__ import annotations
import os
from pathlib import Path
from typing import List


def ensure_dir(path: Path) -> None:
    """Create directory and parents if they don't exist."""
    path.mkdir(parents=True, exist_ok=True)


def atomic_write(path: Path, content: str) -> None:
    """Write file atomically using temporary file and rename."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def read_text(path: Path) -> str | None:
    """Read file safely, return None if file doesn't exist."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def nonempty(path: Path, min_bytes: int = 10) -> bool:
    """Check if file exists and has minimum content."""
    try:
        return path.stat().st_size >= min_bytes
    except FileNotFoundError:
        return False


def list_py_files(root: Path) -> List[Path]:
    """List all .py files recursively under root."""
    return list(root.rglob("*.py"))
