from __future__ import annotations
import json, os, re, shutil
from pathlib import Path
from typing import Any, Dict

# repository root
ROOT = Path(__file__).resolve().parents[3]

def _slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+","-", s.strip())
    return s.strip("-").lower() or "project"

def scaffold_project(state: Dict[str, Any]) -> Dict[str, Any]:
    goal = (state.get("plan") or {}).get("goal", "agent_project")
    name = _slug(goal)[:40]
    base = ROOT / "projects" / name
    base.mkdir(parents=True, exist_ok=True)
    (base / "prompts").mkdir(parents=True, exist_ok=True)
    (base / "src" / "agent").mkdir(parents=True, exist_ok=True)
    (base / "src" / "config").mkdir(parents=True, exist_ok=True)
    (base / "src" / "llm").mkdir(parents=True, exist_ok=True)
    (base / "tests").mkdir(parents=True, exist_ok=True)
    (base / "reports").mkdir(parents=True, exist_ok=True)

    for package_dir in ("src", "src/agent", "src/llm", "src/config"):
        init_path = base / package_dir / "__init__.py"
        init_path.parent.mkdir(parents=True, exist_ok=True)
        if not init_path.exists():
            init_path.write_text("", encoding="utf-8")

    # langgraph.json
    (base / "langgraph.json").write_text(
        json.dumps({"graphs": {"agent": "agent.graph:graph"},
                    "dependencies": ["."], "env": "./.env"}, indent=2), encoding="utf-8")

    # pyproject.toml
    (base / "pyproject.toml").write_text("""[project]
name = "generated-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "langgraph>=0.6,<0.7",
  "langchain-core>=0.3,<0.4",
  "langchain-openai>=0.3,<0.4",
  "pydantic>=2.7,<3",
  "langgraph-checkpoint-sqlite>=2.0.0",
  "aiosqlite>=0.17.0",
  "pytest>=7.0.0",
  "langgraph-cli[inmem]>=0.1.0",
  "requests>=2.25.0",
]
[build-system]
requires = ["setuptools","wheel"]
build-backend = "setuptools.build_meta"
[tool.setuptools.packages.find]
where = ["src"]
""", encoding="utf-8")

    # .env.example
    src_env = ROOT / ".env.example"
    if src_env.exists():
        shutil.copy(src_env, base / ".env.example")

    # copy minimal settings, client, state, and prompt utilities
    files = {
        "src/config/settings.py": "src/config/settings.py",
        "src/asb/llm/client.py": "src/llm/client.py",
        "src/asb/agent/state.py": "src/agent/state.py",
        "src/asb/agent/prompts_util.py": "src/agent/prompts_util.py",
    }
    missing_files = []
    for src_rel, dest_rel in files.items():
        dst = base / dest_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        src_path = ROOT / src_rel
        if src_path.exists():
            shutil.copy(src_path, dst)
        else:
            missing_files.append(str(src_path))
            print(f"Template file missing, skipping: {src_path}")

    # imports are already correct in copied files

    # prompts (simple copies from parent)
    (base / "prompts" / "plan_system.jinja").write_text(
        "You are a planner. Return only JSON for the 3-node plan/do/finish.", encoding="utf-8")
    (base / "prompts" / "plan_user.jinja").write_text("User goal:\n{{ user_goal }}\n", encoding="utf-8")

    # child planner.py
    (base / "src/agent/planner.py").write_text("""# generated
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from langchain_core.messages import HumanMessage
from llm.client import get_chat_model
from .prompts_util import find_prompts_dir

class PlanNode(BaseModel):
    id: str; type: str; prompt: Optional[str] = None; tool: Optional[str] = None
class PlanEdge(BaseModel):
    from_: str = Field(..., alias="from"); to: str; if_: Optional[str] = Field(None, alias="if")
class Plan(BaseModel):
    goal: str; nodes: List[PlanNode]; edges: List[PlanEdge]; confidence: Optional[float] = None

PROMPTS_DIR = find_prompts_dir()
SYSTEM_PROMPT = (PROMPTS_DIR / "plan_system.jinja").read_text(encoding="utf-8")
USER_TMPL = (PROMPTS_DIR / "plan_user.jinja").read_text(encoding="utf-8")

def plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        llm = get_chat_model()
        goal = (state.get("messages") or [{}])[-1].get("content","Goal")
        user_prompt = USER_TMPL.replace("{{ user_goal }}", goal)
        nodes = [
            {"id":"summarize","type":"llm","prompt": "Summarize the following text: {{input_text}}"},
        ]
        edges = []
        plan = Plan(goal=goal, nodes=[PlanNode(**n) for n in nodes], edges=[PlanEdge(**e) for e in edges], confidence=0.9).model_dump(by_alias=True)
        messages = list(state.get("messages") or []) + [{"role":"assistant","content":"Planned summarization workflow."}]
        return {"plan": plan, "messages": messages}
    except Exception as e:
        # Handle potential errors, e.g., LLM call fails
        return {"error": str(e), "messages": list(state.get("messages") or [])}
""", encoding="utf-8")

    # child executor.py
    (base / "src/agent/executor.py").write_text("""# generated
from langchain_core.messages import HumanMessage
from llm.client import get_chat_model
from typing import Dict, Any

def execute(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        llm = get_chat_model()
        plan = state.get("plan", {})
        nodes = {n["id"]: n for n in plan.get("nodes", [])}

        # Assuming the input text is in the last message
        input_text = (state.get("messages") or [{}])[-1].get("content", "")

        summarize_prompt = (nodes.get("summarize") or {}).get("prompt", "Summarize the following text: {{input_text}}")
        prompt = summarize_prompt.replace("{{input_text}}", input_text)

        summary = llm.invoke([HumanMessage(prompt)]).content

        messages = list(state.get("messages") or []) + [{"role": "assistant", "content": summary}]
        return {"messages": messages}
    except Exception as e:
        # Handle potential errors, e.g., LLM call fails
        return {"error": str(e), "messages": list(state.get("messages") or [])}
""", encoding="utf-8")

    # child db_setup.py
    (base / "src/agent/db_setup.py").write_text("""# generated
from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_DB_FILENAME = "state.db"
DEFAULT_DB_DIR = Path(os.environ.get("AGENT_SQLITE_DIR", ".agent"))


def _resolve_path(path: Optional[str]) -> Path:
    if path:
        return Path(path)
    env_path = os.environ.get("AGENT_SQLITE_DB_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_DB_DIR / DEFAULT_DB_FILENAME


def ensure_sqlite_db(path: Optional[str] = None) -> Tuple[str, sqlite3.Connection]:
    db_path = _resolve_path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("Ensuring SQLite database at %s", db_path)
    connection = sqlite3.connect(str(db_path), check_same_thread=False)
    return str(db_path), connection
""", encoding="utf-8")

    # child graph.py
    (base / "src/agent/graph.py").write_text("""# generated
from __future__ import annotations

import logging
import os
import sys
from typing import Dict, Any

from langgraph.graph import StateGraph, START, END

logger = logging.getLogger(__name__)

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except Exception as exc:  # pragma: no cover - handled via logging below
    SqliteSaver = None  # type: ignore[assignment]
    _SQLITE_IMPORT_ERROR = exc
else:
    _SQLITE_IMPORT_ERROR = None

from .planner import plan_node
from .executor import execute


def running_on_langgraph_api() -> bool:
    langgraph_env = os.environ.get("LANGGRAPH_ENV", "").lower()
    if langgraph_env in {"cloud", "api", "hosted"}:
        return True
    if os.environ.get("LANGGRAPH_API_URL") or os.environ.get("LANGGRAPH_CLOUD"):
        return True
    argv = [arg.lower() for arg in sys.argv[1:]]
    return "--langgraph-api" in argv or ("langgraph" in argv and "api" in argv)


def _make_graph(path: str | None = None):
    g = StateGraph(dict)
    g.add_node("plan", plan_node)
    g.add_node("execute", execute)
    g.add_edge(START, "plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", END)

    if running_on_langgraph_api():
        logger.info("LangGraph API runtime detected; compiling without a checkpointer.")
        return g.compile(checkpointer=None)

    if SqliteSaver is None:
        if _SQLITE_IMPORT_ERROR:
            logger.warning(
                "langgraph.checkpoint.sqlite unavailable (%s); using in-memory checkpoints.",
                _SQLITE_IMPORT_ERROR,
            )
        else:
            logger.warning("SqliteSaver is unavailable; using in-memory checkpoints.")
        return g.compile()

    try:
        from . import db_setup
    except Exception as exc:
        logger.warning("Failed to import agent.db_setup (%s); using in-memory checkpoints.", exc)
        return g.compile()

    try:
        resolved_path, connection = db_setup.ensure_sqlite_db(path)
    except Exception as exc:
        logger.warning("SQLite setup failed (%s); using in-memory checkpoints.", exc)
        return g.compile()

    logger.debug("Using SQLite checkpointer at %s", resolved_path)
    memory = SqliteSaver(connection)
    return g.compile(checkpointer=memory)


graph = _make_graph()
""", encoding="utf-8")

    # tests
    (base / "tests" / "test_smoke.py").write_text("""def test_import_graph():
    from agent.graph import graph
    assert graph is not None
""", encoding="utf-8")

    # README
    (base / "README.md").write_text(f"""# {name}
Generated by Agentic-System-Builder MVP.

## Run
pip install -e . "langgraph-cli[inmem]"
cp .env.example .env
langgraph dev
""", encoding="utf-8")

    state["scaffold"] = {"path": str(base), "ok": True}
    if missing_files:
        state["scaffold"]["missing"] = missing_files
    return state
