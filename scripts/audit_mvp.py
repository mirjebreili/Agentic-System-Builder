# scripts/audit_mvp.py
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def ok(flag, msg):
    print(("✅ " if flag else "❌ ") + msg)
    return flag

def must_exist(p):
    return ok(p.exists(), f"exists: {p}")

def must_contain(p, pat, label):
    if not p.exists():
        return ok(False, f"{label}: missing file {p}")
    text = p.read_text(encoding="utf-8", errors="ignore")
    return ok(bool(re.search(pat, text, re.S)), f"{label}: pattern found")

errors = 0

# Top-level
errors += not must_exist(ROOT/"langgraph.json")
if (ROOT/"langgraph.json").exists():
    try:
        lg = json.loads((ROOT/"langgraph.json").read_text())
        expect = lg.get("graphs",{}).get("agent","")
        ok(expect.endswith("src/agent/graph.py:graph"), "langgraph.json maps graphs.agent to src/agent/graph.py:graph")
        ok("." in (lg.get("dependencies") or []), "langgraph.json has dependencies=['.']")
        ok(lg.get("env","") == "./.env", "langgraph.json sets env ./ .env")
    except Exception as e:
        ok(False, f"langgraph.json parse error: {e}"); errors += 1

errors += not must_exist(ROOT/".env.example")
errors += not must_exist(ROOT/"pyproject.toml")
errors += not must_exist(ROOT/"prompts"/"plan_system.jinja")
errors += not must_exist(ROOT/"prompts"/"plan_user.jinja")

# Source layout
for rel in [
    "src/config/settings.py",
    "src/llm/client.py",
    "src/agent/state.py",
    "src/agent/planner.py",
    "src/agent/confidence.py",
    "src/agent/executor.py",
    "src/agent/hitl.py",
    "src/agent/tests_node.py",
    "src/agent/scaffold.py",
    "src/agent/sandbox.py",
    "src/agent/report.py",
    "src/agent/graph.py",
]:
    errors += not must_exist(ROOT/rel)

# Critical content checks
errors += not must_contain(ROOT/"src/agent/graph.py", r"SqliteSaver", "graph.py uses SQLite checkpointer")
errors += not must_contain(ROOT/"src/agent/graph.py", r"add_node\(\s*\"plan_tot\"", "graph has plan_tot")
errors += not must_contain(ROOT/"src/agent/hitl.py", r"INTERRUPT", "HITL interrupt used")
errors += not must_contain(ROOT/"src/agent/planner.py", r"PlanNode|PlanEdge|Plan", "planner defines Plan models")
errors += not must_contain(ROOT/"src/agent/planner.py", r"Selected ToT plan", "planner logs selection")
errors += not must_contain(ROOT/"src/agent/executor.py", r"DONE|FINISHED", "executor detects DONE tokens")

print("\n---- RESULT ----")
if errors:
    print(f"❌ {errors} issue(s) found. See ❌ lines above.")
    sys.exit(1)
else:
    print("✅ Looks good. Now try: pip install -e . \"langgraph-cli[inmem]\" && cp .env.example .env && langgraph dev")
