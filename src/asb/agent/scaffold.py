from __future__ import annotations
import json, os, re, shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


STATE_TEMPLATE = """from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ChatMessage(TypedDict, total=False):
    role: Literal[\"human\", \"user\", \"assistant\", \"system\", \"tool\"]
    content: str


class AppState(TypedDict, total=False):
    architecture: Dict[str, Any]
    artifacts: Dict[str, Any]
    debug: Dict[str, Any]
    flags: Dict[str, bool]
    generated_files: Dict[str, str]
    messages: Annotated[List[AnyMessage], add_messages]
    metrics: Dict[str, Any]
    passed: bool
    plan: Dict[str, Any]
    replan: bool
    report: Dict[str, Any]
    requirements: Dict[str, Any]
    review: Dict[str, Any]
    sandbox: Dict[str, Any]
    scaffold: Dict[str, Any]
    syntax_validation: Dict[str, Any]
    tests: Dict[str, Any]


def update_state_with_circuit_breaker(state: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"Add circuit breaker logic to prevent infinite loops\"\"\"

    if \"fix_attempts\" not in state:
        state[\"fix_attempts\"] = 0

    if \"consecutive_failures\" not in state:
        state[\"consecutive_failures\"] = 0

    if \"repair_start_time\" not in state:
        import time

        state[\"repair_start_time\"] = time.time()

    return state
"""

# repository root
ROOT = Path(__file__).resolve().parents[3]

def _slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+","-", s.strip())
    return s.strip("-").lower() or "project"


def _normalize_generated_key(key: str) -> str:
    """Normalize generated file keys for reliable lookups."""

    normalized = key.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _get_generated_content(
    generated: Dict[str, str],
    *candidates: str,
) -> str | None:
    for candidate in candidates:
        normalized = _normalize_generated_key(candidate)
        if normalized in generated:
            return generated[normalized]
    return None


def _sanitize_identifier(value: str) -> str:
    sanitized = re.sub(r"\W+", "_", value).strip("_")
    return sanitized or "node"


def _candidate_call_hints(node_id: str, module_name: str) -> List[str]:
    """Build an ordered list of attribute names likely to contain the node callable."""

    hints: List[str] = ["run", "execute"]
    variants = {module_name, module_name.lower()}
    normalized_id = re.sub(r"\W+", "_", node_id).strip("_")
    if normalized_id:
        variants.add(normalized_id)
        variants.add(normalized_id.lower())

    for variant in variants:
        if not variant:
            continue
        hints.append(variant)
        hints.append(f"run_{variant}")
        hints.append(f"{variant}_run")

    ordered: List[str] = []
    for hint in hints:
        if hint and hint not in ordered:
            ordered.append(hint)
    return ordered


def _extract_node_id(node: Dict[str, Any]) -> str | None:
    for key in ("id", "node", "name", "label"):
        value = node.get(key)
        if value is None:
            continue
        candidate = str(value).strip()
        if candidate:
            return candidate
    if len(node) == 1:
        only_key = next(iter(node))
        candidate = str(node[only_key]).strip()
        return candidate or None
    return None


def _collect_architecture_nodes(architecture: Dict[str, Any]) -> List[Tuple[str, str, List[str]]]:
    nodes = architecture.get("graph_structure")
    if not nodes:
        return []

    ordered_nodes: List[Tuple[str, str, List[str]]] = []
    seen: set[str] = set()

    iterable: Iterable[Any]
    if isinstance(nodes, dict):
        iterable = nodes.items()
    else:
        iterable = nodes

    for item in iterable:
        node: Dict[str, Any]
        if isinstance(item, tuple) and len(item) == 2 and isinstance(nodes, dict):
            node_id_raw, details = item
            if isinstance(details, dict):
                node = dict(details)
                node.setdefault("id", node_id_raw)
            else:
                node = {"id": node_id_raw, "description": details}
        elif isinstance(item, dict):
            node = item
        else:
            continue

        node_id = _extract_node_id(node)
        if not node_id:
            continue
        if node_id in seen:
            continue

        sanitized = _sanitize_identifier(node_id)
        hints = _candidate_call_hints(node_id, sanitized)
        ordered_nodes.append((node_id, sanitized, hints))
        seen.add(node_id)

    return ordered_nodes


def _render_node_stub(node_id: str, sanitized: str) -> str:
    return (
        "from typing import Any, Dict\n\n"
        f"def {sanitized}(state: Dict[str, Any]) -> Dict[str, Any]:\n"
        f"    \"\"\"Placeholder implementation for node '{node_id}'.\"\"\"\n"
        "    return state\n"
    )


def _write_node_modules(
    base: Path,
    node_specs: List[Tuple[str, str, List[str]]],
    generated: Dict[str, str],
    missing_files: List[str],
) -> None:
    agent_dir = base / "src" / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)

    for node_id, module_name, _ in node_specs:
        filename = f"{module_name}.py"
        source = _get_generated_content(
            generated,
            filename,
            f"src/agent/{filename}",
            f"agent/{filename}",
        )
        destination = agent_dir / filename

        if source is None:
            source = _render_node_stub(node_id, module_name)
            missing_entry = str(destination)
            if missing_entry not in missing_files:
                missing_files.append(missing_entry)

        destination.write_text(source, encoding="utf-8")


def _render_executor_module(node_specs: List[Tuple[str, str, List[str]]]) -> str:
    lines: List[str] = [
        "# generated",
        "from __future__ import annotations",
        "",
        "import importlib",
        "from functools import lru_cache",
        "from typing import Any, Callable, Dict, Iterable, List, Tuple",
        "",
        "",
        "_NODE_SPECS: List[Tuple[str, str, List[str]]] = [",
    ]

    for node_id, module_name, hints in node_specs:
        hint_repr = ", ".join(repr(hint) for hint in hints)
        lines.append(f"    ({node_id!r}, {module_name!r}, [{hint_repr}]),")

    lines.extend(
        [
            "]",
            "",
            "",
            "@lru_cache(maxsize=None)",
            "def _import_node(module: str):",
            "    package = __name__.rsplit('.', 1)[0]",
            "    return importlib.import_module(f\"{package}.{module}\")",
            "",
            "",
            "def _resolve_callable(module, hints: Iterable[str]):",
            "    for attr in hints:",
            "        func = getattr(module, attr, None)",
            "        if callable(func):",
            "            return func",
            "",
            "    for attr in dir(module):",
            "        if attr.startswith('_'):",
            "            continue",
            "        candidate = getattr(module, attr)",
            "        if callable(candidate):",
            "            return candidate",
            "",
            "    def _identity(state: Dict[str, Any]) -> Dict[str, Any]:",
            "        return state",
            "",
            "    module_name = module.__name__.split('.')[-1]",
            "    _identity.__name__ = f\"noop_{module_name}\"",
            "    return _identity",
            "",
            "",
            "NODE_IMPLEMENTATIONS: List[Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]]] = []",
            "for _node_id, _module_name, _hints in _NODE_SPECS:",
            "    _module = _import_node(_module_name)",
            "    _callable = _resolve_callable(_module, _hints)",
            "    NODE_IMPLEMENTATIONS.append((_node_id, _callable))",
            "",
            "",
            "def iter_node_callables() -> Iterable[Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]]]:",
            "    for spec in NODE_IMPLEMENTATIONS:",
            "        yield spec",
            "",
            "",
            "def execute(state: Dict[str, Any]) -> Dict[str, Any]:",
            "    current = state",
            "    for _, node_callable in NODE_IMPLEMENTATIONS:",
            "        current = node_callable(current)",
            "    return current",
            "",
        ]
    )

    return "\n".join(lines) + "\n"


def _render_graph_module() -> str:
    lines: List[str] = [
        "# generated",
        "from __future__ import annotations",
        "",
        "import logging",
        "import os",
        "import sqlite3",
        "import sys  # required for runtime argv detection",
        "from typing import Any, Dict",
        "",
        "from langgraph.checkpoint.sqlite import SqliteSaver",
        "from langgraph.graph import StateGraph, START, END",
        "",
        "from .state import AppState",
        "from .executor import NODE_IMPLEMENTATIONS",
        "",
        "logger = logging.getLogger(__name__)",
        "",
        "",
        "def running_on_langgraph_api() -> bool:",
        "    langgraph_env = os.environ.get('LANGGRAPH_ENV', '').lower()",
        "    if langgraph_env in {'cloud', 'api', 'hosted'}:",
        "        return True",
        "    if os.environ.get('LANGGRAPH_API_URL') or os.environ.get('LANGGRAPH_CLOUD'):",
        "        return True",
        "    argv = [arg.lower() for arg in sys.argv[1:]]",
        "    return '--langgraph-api' in argv or ('langgraph' in argv and 'api' in argv)",
        "",
        "",
        "def _make_graph(path: str | None = None):",
        "    g = StateGraph(AppState)",
        "",
        "    if NODE_IMPLEMENTATIONS:",
        "        previous = START",
        "        for node_id, node_callable in NODE_IMPLEMENTATIONS:",
        "            g.add_node(node_id, node_callable)",
        "            g.add_edge(previous, node_id)",
        "            previous = node_id",
        "        g.add_edge(previous, END)",
        "    else:",
        "        g.add_edge(START, END)",
        "",
        "    if running_on_langgraph_api():",
        "        logger.info('LangGraph API runtime detected; compiling without a checkpointer.')",
        "        return g.compile(checkpointer=None)",
        "",
        "    checkpointer = None",
        "    if path:",
        "        dir_path = os.path.dirname(path)",
        "        if dir_path:",
        "            os.makedirs(dir_path, exist_ok=True)",
        "        connection = sqlite3.connect(path, check_same_thread=False)",
        "        checkpointer = SqliteSaver(connection)",
        "",
        "    return g.compile(checkpointer=checkpointer)",
        "",
        "",
        "graph = _make_graph()",
    ]

    return "\n".join(lines) + "\n"

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
        json.dumps({"graphs": {"agent": "src.agent.graph:graph"},
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
  "black>=22.0.0",
  "isort>=5.0.0",
  "mypy>=1.0.0",
  "bandit[toml]>=1.7.0",
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

    normalized_generated = {
        _normalize_generated_key(key): value
        for key, value in (state.get("generated_files") or {}).items()
    }

    generated_state = _get_generated_content(
        normalized_generated,
        "state.py",
        "src/agent/state.py",
        "agent/state.py",
    )
    state_path = base / "src" / "agent" / "state.py"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if generated_state:
        state_path.write_text(generated_state, encoding="utf-8")
    else:
        state_path.write_text(STATE_TEMPLATE, encoding="utf-8")

    # imports are already correct in copied files

    # prompts (simple copies from parent)
    (base / "prompts" / "plan_system.jinja").write_text(
        "You are a planner. Return only JSON for the 3-node plan/do/finish.", encoding="utf-8")
    (base / "prompts" / "plan_user.jinja").write_text("User goal:\n{{ user_goal }}\n", encoding="utf-8")

    # child planner.py
    (base / "src/agent/planner.py").write_text("""# generated
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from .prompts_util import find_prompts_dir


class PlanNode(BaseModel):
    id: str
    type: str
    prompt: Optional[str] = None
    tool: Optional[str] = None


class PlanEdge(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    if_: Optional[str] = Field(None, alias="if")


class Plan(BaseModel):
    goal: str
    nodes: List[PlanNode]
    edges: List[PlanEdge]
    confidence: Optional[float] = None


PROMPTS_DIR = find_prompts_dir()
SYSTEM_PROMPT = (PROMPTS_DIR / "plan_system.jinja").read_text(encoding="utf-8")
USER_TEMPLATE = (PROMPTS_DIR / "plan_user.jinja").read_text(encoding="utf-8")


def _render_user_prompt(goal: str) -> str:
    return USER_TEMPLATE.replace("{{ user_goal }}", goal)


def _resolve_goal(state: Dict[str, Any]) -> str:
    messages = state.get("messages") or []
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if isinstance(message, dict):
            content = message.get("content") or content
        if content:
            return str(content)
    input_text = state.get("input_text")
    if input_text:
        return str(input_text)
    return "Plan a simple workflow."


def plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        goal = _resolve_goal(state)
        plan = Plan(
            goal=goal,
            nodes=[
                PlanNode(id="plan", type="llm", prompt=SYSTEM_PROMPT.strip() or None),
                PlanNode(id="do", type="llm", prompt=_render_user_prompt(goal)),
                PlanNode(id="finish", type="llm", prompt="Summarize the outcome and next steps."),
            ],
            edges=[
                PlanEdge(from_="plan", to="do"),
                PlanEdge(from_="do", to="do", if_="more_steps"),
                PlanEdge(from_="do", to="finish", if_="steps_done"),
            ],
        ).model_dump(by_alias=True)

        messages = list(state.get("messages") or [])
        messages.append(AIMessage(content=f"Generated plan for goal: {goal}"))
        current_step = {"more_steps": True, "steps_done": False}
        return {"plan": plan, "messages": messages, "current_step": current_step}
    except Exception as exc:
        return {
            "error": str(exc),
            "messages": list(state.get("messages") or []),
            "current_step": {"more_steps": False, "steps_done": False},
        }
""", encoding="utf-8")

    architecture = state.get("architecture") or {}
    node_specs = _collect_architecture_nodes(architecture)

    if node_specs:
        _write_node_modules(base, node_specs, normalized_generated, missing_files)

        executor_source = _get_generated_content(
            normalized_generated,
            "executor.py",
            "src/agent/executor.py",
            "agent/executor.py",
        )
        if executor_source is None:
            executor_source = _render_executor_module(node_specs)

        (base / "src/agent/executor.py").write_text(executor_source, encoding="utf-8")

        graph_source = _get_generated_content(
            normalized_generated,
            "graph.py",
            "src/agent/graph.py",
            "agent/graph.py",
        )
        if graph_source is None:
            graph_source = _render_graph_module()

        (base / "src/agent/graph.py").write_text(graph_source, encoding="utf-8")
    else:
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

        (base / "src/agent/graph.py").write_text("""# generated
from __future__ import annotations

import logging
import os
import sqlite3
import sys  # required for runtime argv detection
from typing import Any, Dict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END

from .state import AppState
from .planner import plan_node
from .executor import execute

logger = logging.getLogger(__name__)


def running_on_langgraph_api() -> bool:
    langgraph_env = os.environ.get("LANGGRAPH_ENV", "").lower()
    if langgraph_env in {"cloud", "api", "hosted"}:
        return True
    if os.environ.get("LANGGRAPH_API_URL") or os.environ.get("LANGGRAPH_CLOUD"):
        return True
    argv = [arg.lower() for arg in sys.argv[1:]]
    return "--langgraph-api" in argv or ("langgraph" in argv and "api" in argv)


def _make_graph(path: str | None = None):
    g = StateGraph(AppState)
    g.add_node("plan", plan_node)
    g.add_node("execute", execute)
    g.add_edge(START, "plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", END)

    if running_on_langgraph_api():
        logger.info("LangGraph API runtime detected; compiling without a checkpointer.")
        return g.compile(checkpointer=None)

    checkpointer = None
    if path:
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        connection = sqlite3.connect(path, check_same_thread=False)
        checkpointer = SqliteSaver(connection)

    return g.compile(checkpointer=checkpointer)


graph = _make_graph()
""", encoding="utf-8")

    # tests
    (base / "tests" / "test_smoke.py").write_text(
        '''"""Smoke tests for the generated agent project."""

import importlib
from pathlib import Path

import pytest


def test_import_graph():
    module = importlib.import_module("src.agent.graph")
    assert hasattr(module, "graph")
    assert module.graph is not None


def test_state_structure():
    state_module = importlib.import_module("src.agent.state")
    assert hasattr(state_module, "AppState")
    state_keys = set(getattr(state_module, "AppState").__annotations__)
    expected_keys = {
        "architecture",
        "artifacts",
        "debug",
        "flags",
        "generated_files",
        "messages",
        "metrics",
        "passed",
        "plan",
        "replan",
        "report",
        "requirements",
        "review",
        "sandbox",
        "scaffold",
        "syntax_validation",
        "tests",
    }
    assert expected_keys.issubset(state_keys)


def test_graph_execution(tmp_path: Path):
    from src.agent.graph import _make_graph

    checkpoint_path = tmp_path / "checkpoints" / "graph.db"
    graph = _make_graph(str(checkpoint_path))
    result = graph.invoke(
        {"messages": []}, config={"configurable": {"thread_id": "test-smoke"}}
    )
    assert isinstance(result, dict)
    assert "messages" in result


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
''',
        encoding="utf-8",
    )

    # README
    (base / "README.md").write_text(
        f"""# {name}
Generated by Agentic-System-Builder MVP.

## Features
- Production-ready LangGraph scaffold with planner, executor, and state management modules.
- Prompt templates and configuration helpers for rapid iteration.
- Testing and reporting directories to validate agent behavior out of the box.

## Installation
1. `pip install -e . "langgraph-cli[inmem]"`
2. `cp .env.example .env`
3. Configure environment variables for your LLM provider.

## Usage

### Chat-style interaction
```bash
langgraph dev
```
Launch LangGraph Studio and iterate conversationally with the agent.

### Direct data invocation
```bash
langgraph run src.agent.graph:graph --input data.json
```
Supply structured payloads directly to the graph for batch workflows.

## Development
- Customize prompts in `prompts/` to refine planning behavior.
- Extend business logic under `src/agent/` and `src/llm/`.
- Update configuration defaults in `src/config/settings.py`.

## Testing
```bash
pytest -v
```
Run the automated suite before shipping changes.
""",
        encoding="utf-8",
    )

    state["scaffold"] = {"path": str(base), "ok": True}
    if missing_files:
        state["scaffold"]["missing"] = missing_files
    return state
