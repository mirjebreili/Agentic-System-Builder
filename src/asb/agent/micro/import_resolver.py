"""Guarantee generated modules import local shims for shared utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

from asb.utils.fileops import atomic_write, ensure_dir

_MESSAGE_UTILS_SOURCE = """from __future__ import annotations

from typing import Any, List


def extract_last_message_content(messages: List[Any], default: str = "") -> str:
    if not messages:
        return default

    last_message = messages[-1]
    content = None
    if hasattr(last_message, "content"):
        content = getattr(last_message, "content", None)
    elif isinstance(last_message, dict):
        content = last_message.get("content")
    elif isinstance(last_message, str):
        content = last_message

    if content in (None, ""):
        return default
    return str(content)


def extract_user_messages_content(messages: List[Any]) -> List[str]:
    collected: List[str] = []
    for message in messages:
        role = None
        content = None
        if hasattr(message, "type"):
            role = getattr(message, "type", None)
        if hasattr(message, "role"):
            role = getattr(message, "role", role)
        if hasattr(message, "content"):
            content = getattr(message, "content", None)
        elif isinstance(message, dict):
            role = message.get("role", role)
            content = message.get("content")
        if isinstance(role, str) and role.lower() in {"human", "user"}:
            if content not in (None, ""):
                collected.append(str(content))
    return collected


def safe_message_access(message: Any, field: str, default: Any = "") -> Any:
    if hasattr(message, field):
        return getattr(message, field, default)
    if isinstance(message, dict):
        return message.get(field, default)
    return default


__all__ = [
    "extract_last_message_content",
    "extract_user_messages_content",
    "safe_message_access",
]
"""


def _determine_project_root(state: Dict[str, Any]) -> Path | None:
    scaffold = state.get("scaffold")
    if isinstance(scaffold, dict):
        path = scaffold.get("path")
        if path:
            return Path(path)
    candidate = state.get("project_root")
    if candidate:
        return Path(str(candidate))
    return None


def _iter_node_modules(agent_dir: Path) -> Iterable[Path]:
    for path in agent_dir.glob("*.py"):
        if path.name in {"__init__.py", "state.py"}:
            continue
        yield path


def _ensure_message_utils(agent_dir: Path) -> None:
    utils_dir = agent_dir / "utils"
    ensure_dir(utils_dir)
    init_path = utils_dir / "__init__.py"
    if not init_path.exists():
        atomic_write(init_path, "__all__ = []\n")
    target = utils_dir / "message_utils.py"
    if not target.exists() or target.read_text(encoding="utf-8") != _MESSAGE_UTILS_SOURCE:
        atomic_write(target, _MESSAGE_UTILS_SOURCE)


def _normalize_imports(module_path: Path) -> None:
    contents = module_path.read_text(encoding="utf-8")
    updated = contents.replace(
        "from asb.utils.message_utils import extract_last_message_content",
        "from .utils.message_utils import extract_last_message_content",
    )
    if "extract_last_message_content" in updated and "from .utils.message_utils" not in updated:
        lines = updated.splitlines()
        insert_at = 0
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("from __future__"):
                insert_at = index + 1
                continue
            if stripped.startswith("import ") or stripped.startswith("from "):
                insert_at = index + 1
            else:
                break
        lines.insert(insert_at, "from .utils.message_utils import extract_last_message_content")
        updated = "\n".join(lines) + ("\n" if updated.endswith("\n") else "")
    if updated != contents:
        atomic_write(module_path, updated)


def import_resolver_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure generated nodes depend only on local utility shims."""

    working_state: Dict[str, Any] = dict(state or {})
    project_root = _determine_project_root(working_state)
    if project_root is None:
        return working_state

    agent_dir = project_root / "src" / "agent"
    if not agent_dir.exists():
        return working_state

    _ensure_message_utils(agent_dir)
    for module in _iter_node_modules(agent_dir):
        _normalize_imports(module)

    scratch = dict(working_state.get("scratch") or {})
    scratch.setdefault("artifacts", {})["imports"] = str(agent_dir / "utils" / "message_utils.py")
    working_state["scratch"] = scratch
    return working_state


__all__ = ["import_resolver_node"]
