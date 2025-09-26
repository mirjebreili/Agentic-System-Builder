"""Emit minimal function skeletons for plan nodes."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from asb.utils.fileops import atomic_write, ensure_dir

_SENTINEL = "raise NotImplementedError(\"node logic not yet implemented\")"


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


def _sanitize_identifier(value: str) -> str:
    sanitized = re.sub(r"\W+", "_", value).strip("_")
    return sanitized or "node"


def _iter_plan_nodes(state: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    plan = state.get("plan")
    nodes: Iterable[Any] = []
    if isinstance(plan, dict):
        raw_nodes = plan.get("nodes")
        if isinstance(raw_nodes, list):
            nodes = raw_nodes
    for node in nodes:
        if isinstance(node, dict):
            node_id = node.get("id") or node.get("name") or node.get("label")
            if not isinstance(node_id, str):
                continue
            identifier = node_id.strip()
            if not identifier:
                continue
            yield identifier, _sanitize_identifier(identifier)


def _render_skeleton(node_id: str, module_name: str) -> str:
    return (
        "from __future__ import annotations\n\n"
        "from typing import Any\n\n"
        "from langchain_core.messages import AIMessage, HumanMessage, SystemMessage\n"
        "\n"
        "from .state import AppState\n"
        "from .utils.message_utils import extract_last_message_content\n\n"
        f"__all__ = ['node_{module_name}']\n\n"
        f"def node_{module_name}(state: AppState) -> AppState:\n"
        "    \"\"\"Generated skeleton for a single-purpose node.\"\"\"\n"
        "    messages = list(state.get('messages') or [])\n"
        "    _ = extract_last_message_content(messages, '')\n"
        f"    {_SENTINEL}\n"
        "\n"
        "    return state\n"
    )


def skeleton_writer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Create deterministic node skeletons for each plan entry."""

    working_state: Dict[str, Any] = dict(state or {})
    project_root = _determine_project_root(working_state)
    if project_root is None:
        return working_state

    agent_dir = project_root / "src" / "agent"
    ensure_dir(agent_dir)

    created: List[str] = []
    for node_id, module_name in _iter_plan_nodes(working_state):
        path = agent_dir / f"{module_name}.py"
        skeleton = _render_skeleton(node_id, module_name)
        needs_write = True
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            needs_write = _SENTINEL in existing or not existing.strip()
        if needs_write:
            ensure_dir(path.parent)
            atomic_write(path, skeleton)
            created.append(str(path))

    scratch = dict(working_state.get("scratch") or {})
    if created:
        scratch.setdefault("artifacts", {})["skeletons"] = created
    working_state["scratch"] = scratch
    return working_state


__all__ = ["skeleton_writer_node"]
