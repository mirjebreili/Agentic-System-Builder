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

    # langgraph.json
    (base / "langgraph.json").write_text(
        json.dumps({"graphs":{"agent":"src/agent/graph.py:graph"},
                    "dependencies":["."],"env":"./.env"}, indent=2), encoding="utf-8")

    # pyproject.toml
    (base / "pyproject.toml").write_text("""[project]
name = "generated-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "langgraph>=0.2.39,<0.3.0",
  "langchain-core>=0.2.28,<0.3.0",
  "langchain-openai>=0.2.4,<0.3.0",
  "pydantic>=2.7,<3",
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
    plan_nodes = {n["id"]: n for n in (state.get("plan") or {}).get("nodes", [])}
    (base / "src/agent/planner.py").write_text(f"""# generated
from pydantic import BaseModel, Field
from typing import Optional, List
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

def plan_node(state: dict) -> dict:
    llm = get_chat_model()
    goal = (state.get("messages") or [{{}}])[-1].get("content","Goal")
    user_prompt = USER_TMPL.replace("{{ user_goal }}", goal)
    nodes = [
        {{"id":"plan","type":"llm","prompt": {json.dumps(plan_nodes.get('plan',{}).get('prompt','Split into steps.'))} }},
        {{"id":"do","type":"llm","prompt": {json.dumps(plan_nodes.get('do',{}).get('prompt','Do next step; write ONLY DONE when done.'))} }},
        {{"id":"finish","type":"llm","prompt": {json.dumps(plan_nodes.get('finish',{}).get('prompt','Summarize briefly.'))} }}
    ]
    edges = [{{"from":"plan","to":"do"}},{{"from":"do","to":"do","if":"more_steps"}},{{"from":"do","to":"finish","if":"steps_done"}}]
    plan = Plan(goal=goal, nodes=[PlanNode(**n) for n in nodes], edges=[PlanEdge(**e) for e in edges], confidence=0.8).model_dump(by_alias=True)
    messages = list(state.get("messages") or []) + [{{"role":"assistant","content":"Planned workflow."}}]
    return {{"plan": plan, "messages": messages, "flags": {{"more_steps": True, "steps_done": False}}}}
""", encoding="utf-8")

    # child executor.py
    (base / "src/agent/executor.py").write_text("""# generated
from langchain_core.messages import HumanMessage
from llm.client import get_chat_model

_DONE = ("DONE","COMPLETED","FINISHED")

def _is_done(t:str)->bool: u=(t or "").upper(); return any(x in u for x in _DONE)

def execute(state: dict) -> dict:
    llm = get_chat_model()
    plan = state.get("plan", {})
    nodes = {n["id"]: n for n in plan.get("nodes", [])}
    msgs = list(state.get("messages") or [])
    out = llm.invoke([HumanMessage((nodes.get("plan") or {}).get("prompt","Split into steps."))]).content
    msgs.append({"role":"assistant","content":f"[plan]\\n{out}"})
    it=0
    while it<5:
        it+=1
        do = llm.invoke([HumanMessage((nodes.get("do") or {}).get("prompt","Do next step; write ONLY DONE when done."))]).content
        msgs.append({"role":"assistant","content":f"[do]\\n{do}"})
        if _is_done(do): break
    fin = llm.invoke([HumanMessage((nodes.get("finish") or {}).get("prompt","Summarize briefly."))]).content
    msgs.append({"role":"assistant","content":f"[finish]\\n{fin}"})
    return {"messages": msgs}
""", encoding="utf-8")

    # child graph.py
    (base / "src/agent/graph.py").write_text("""# generated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint import MemorySaver
from typing import Dict, Any
from .planner import plan_node
from .executor import execute

def _make_graph():
    g = StateGraph(dict)
    g.add_node("plan", plan_node)
    g.add_node("execute", execute)
    g.add_edge(START, "plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", END)
    return g.compile(checkpointer=MemorySaver())

graph = _make_graph()
""", encoding="utf-8")

    # tests
    (base / "tests" / "test_smoke.py").write_text("""def test_import_graph():
    from src.agent.graph import graph
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
