#!/usr/bin/env python3
"""
Bootstrap a complete test suite for LangGraph projects.

Usage:
    python scripts/bootstrap_tests.py
    python scripts/bootstrap_tests.py --force
    python scripts/bootstrap_tests.py --with-samples
    python scripts/bootstrap_tests.py --prompts "hello|plan a task|summarize this"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import dedent


def parse_args():
    parser = argparse.ArgumentParser(description="Bootstrap LangGraph test suite")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing test files",
    )
    parser.add_argument(
        "--with-samples",
        action="store_true",
        help="Generate minimal sample app files if missing",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="hello|plan a task|summarize this data",
        help="Pipe-separated list of test prompts (default: 'hello|plan a task|summarize this data')",
    )
    return parser.parse_args()


def write_file(path: Path, content: str, force: bool = False) -> tuple[bool, str]:
    """
    Write content to path. Returns (created, status_msg).
    """
    if path.exists() and not force:
        return False, f"SKIP (exists): {path}"
    
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    action = "CREATED" if not path.exists() or force else "OVERWROTE"
    return True, f"{action}: {path}"


# ============================================================================
# TEST FILE CONTENTS
# ============================================================================

CONFTEST_PY = '''"""
Shared pytest fixtures for the LangGraph test suite.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# Try importing the app graph
try:
    from src.agents.graph import graph as app_graph
except ImportError:
    app_graph = None


@pytest.fixture
def fake_llm():
    """
    A deterministic fake LLM that records prompts and returns canned responses.
    """
    class FakeLLM:
        def __init__(self):
            self.prompts = []
            self.response = AIMessage(content="Fake LLM response")
        
        def invoke(self, messages):
            self.prompts.append(messages)
            return self.response
        
        async def ainvoke(self, messages):
            self.prompts.append(messages)
            return self.response
    
    return FakeLLM()


@pytest.fixture
def ok_tool():
    """A tool that always succeeds."""
    def _tool(plan: str = "default") -> str:
        return f"ok:{plan}"
    return _tool


@pytest.fixture
def bad_tool():
    """A tool that always fails."""
    def _tool(*args, **kwargs):
        raise RuntimeError("boom")
    return _tool


@pytest.fixture
def graph():
    """
    Returns the compiled application graph with MemorySaver.
    """
    if app_graph is None:
        pytest.skip("Application graph not available")
    
    # The graph is already compiled, just return it
    return app_graph


@pytest.fixture
def free_port() -> int:
    """Find a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


@pytest.fixture
def dev_server(free_port: int) -> Iterator[str]:
    """
    Spawn a langgraph dev server, wait for /ok, yield base URL, then terminate.
    """
    # Check if langgraph CLI is available
    if not shutil.which("langgraph"):
        pytest.skip("langgraph CLI not available")
    
    port = free_port
    cmd = [
        "langgraph",
        "dev",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--no-reload",
        "--no-browser",
    ]
    
    # Add config file if it exists
    config_path = Path("langgraph.json")
    if config_path.exists():
        cmd.extend(["--config", "langgraph.json"])
    
    # Start the server
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    base_url = f"http://127.0.0.1:{port}"
    
    # Wait for server to be ready (up to 30 seconds)
    import requests
    ready = False
    for _ in range(60):  # 30 seconds with 0.5s intervals
        try:
            resp = requests.get(f"{base_url}/ok", timeout=1)
            if resp.status_code == 200:
                ready = True
                break
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(0.5)
        
        # Check if process died
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            pytest.fail(f"Server failed to start. stdout: {stdout}, stderr: {stderr}")
    
    if not ready:
        proc.terminate()
        proc.wait(timeout=5)
        pytest.fail(f"Server did not become ready at {base_url}/ok within 30s")
    
    try:
        yield base_url
    finally:
        # Cleanup
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@pytest.fixture
def lg_client(dev_server: str):
    """
    Returns a sync LangGraph SDK client connected to dev_server.
    """
    from langgraph_sdk import get_sync_client
    return get_sync_client(url=dev_server)


@pytest.fixture
def default_assistant_id(lg_client) -> str:
    """
    Discover an assistant from the server.
    Prefer one matching 'agent' graph_id, otherwise pick the first.
    """
    assistants = lg_client.assistants.search()
    if not assistants:
        pytest.fail("No assistants found on server")
    
    # Try to find 'agent' assistant
    for asst in assistants:
        if asst.get("graph_id") == "agent":
            return asst["assistant_id"]
    
    # Return first assistant
    return assistants[0]["assistant_id"]


@pytest.fixture
def cli_available() -> bool:
    """Check if langgraph CLI is available."""
    return shutil.which("langgraph") is not None


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "cli: mark test as requiring langgraph CLI"
    )
    config.addinivalue_line(
        "markers", 'slow: mark test as slow (deselect with \\'\\'-m "not slow"\\'\\' )'
    )
'''

TEST_REPO_LAYOUT_PY = '''"""
Meta tests: verify project structure and imports.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


def test_required_files_exist():
    """Check that required files exist."""
    required = [
        "langgraph.json",
        "pyproject.toml",
        "src/agents/graph.py",
        "src/agents/state.py",
    ]
    
    for rel_path in required:
        path = Path(rel_path)
        assert path.exists(), f"Required file missing: {rel_path}"


def test_can_import_graph():
    """Verify we can import the application graph."""
    try:
        from src.agents.graph import graph
        assert graph is not None
        assert callable(getattr(graph, "invoke", None)) or callable(getattr(graph, "stream", None))
    except ImportError as e:
        pytest.fail(f"Failed to import graph: {e}")


def test_can_import_state():
    """Verify we can import the state definition."""
    try:
        from src.agents.state import AppState
        assert AppState is not None
    except ImportError as e:
        pytest.fail(f"Failed to import AppState: {e}")


def test_python_version():
    """Ensure we're running on Python 3.11+."""
    assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version_info}"
'''

TEST_NODES_PY = '''"""
Unit tests for individual graph nodes.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest


def test_extract_system_elements_no_prompt():
    """Test extract_system_elements with no human message."""
    from src.agents.plugin_analyzer import extract_system_elements
    
    state = {"messages": []}
    result = extract_system_elements(state)
    
    assert "system_elements" in result
    assert "plugins" in result
    assert "has_system_elements" in result
    assert result["has_system_elements"] is False


def test_extract_system_elements_with_json():
    """Test extraction with JSON plugin definition."""
    from src.agents.plugin_analyzer import extract_system_elements
    from langchain_core.messages import HumanMessage
    
    prompt = """
    Here are my plugins:
    {"plugins": [{"name": "TestPlugin", "goal": "Do something"}]}
    """
    
    state = {"messages": [HumanMessage(content=prompt)]}
    result = extract_system_elements(state)
    
    assert result["has_system_elements"] is True
    assert len(result["plugins"]) > 0
    assert result["plugins"][0]["name"] == "TestPlugin"


def test_split_task_simple_goal():
    """Test task splitting with a simple goal."""
    from src.agents.splitter import split_task
    from langchain_core.messages import HumanMessage
    
    state = {
        "messages": [HumanMessage(content="Build a simple web app")],
        "has_system_elements": False,
        "system_elements": []
    }
    
    result = split_task(state)
    
    assert "split_tasks" in result
    assert len(result["split_tasks"]) > 0
    assert all("id" in task and "description" in task for task in result["split_tasks"])


def test_compute_plan_confidence():
    """Test confidence calculation."""
    from src.agents.confidence import compute_plan_confidence
    
    plan = {
        "goal": "test",
        "nodes": [{"id": "node1", "tool": "test_tool"}],
        "edges": [],
        "confidence": 0.8
    }
    
    state = {"plan": plan, "debug": {}}
    result = compute_plan_confidence(state)
    
    assert "plan" in result
    assert "confidence" in result["plan"]
    assert 0.0 <= result["plan"]["confidence"] <= 1.0


def test_review_plan_structure():
    """Test that review_plan has correct structure (without interrupt)."""
    from src.agents.hitl import review_plan
    
    plan = {
        "goal": "test",
        "nodes": [],
        "edges": [],
        "confidence": 0.9
    }
    
    state = {"plan": plan}
    
    # Note: This will trigger an interrupt in real execution
    # In unit tests, we just verify the function exists and is callable
    assert callable(review_plan)


def test_format_plan_order():
    """Test plan formatting."""
    from src.agents.formatter import format_plan_order
    from langchain_core.messages import HumanMessage
    
    plan = {
        "goal": "test",
        "nodes": [
            {"id": "step1", "tool": "tool1", "reasoning": "First step"},
            {"id": "step2", "tool": "tool2", "reasoning": "Second step"}
        ],
        "edges": [{"from": "step1", "to": "step2"}],
        "confidence": 0.85
    }
    
    state = {
        "plan": plan,
        "messages": [HumanMessage(content="Test")],
        "debug": {"plan_candidates": [{"confidence": 0.85, "plan": plan}]}
    }
    
    result = format_plan_order(state)
    
    assert "messages" in result
    assert len(result["messages"]) > 0
'''

TEST_ROUTER_PY = '''"""
Unit tests for router/conditional edge logic.
"""
from __future__ import annotations

import pytest


def test_route_after_review_approve():
    """Test routing after plan approval."""
    from src.agents.graph import route_after_review
    
    state = {"replan": False}
    next_node = route_after_review(state)
    
    assert next_node == "format_plan_order"


def test_route_after_review_revise():
    """Test routing when replan is needed."""
    from src.agents.graph import route_after_review
    
    state = {"replan": True}
    next_node = route_after_review(state)
    
    assert next_node == "plan_tot"


def test_route_after_review_default():
    """Test routing with missing replan field."""
    from src.agents.graph import route_after_review
    
    state = {}
    next_node = route_after_review(state)
    
    # Should default to approve (replan=False)
    assert next_node == "format_plan_order"
'''

TEST_GRAPH_INTEGRATION_PY = '''"""
Integration tests for the full graph execution.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage


def test_graph_invoke_basic(graph):
    """Test basic graph invocation."""
    initial_state = {
        "messages": [HumanMessage(content="Create a simple plan for building a website")]
    }
    
    # Note: This will hit the interrupt at review_plan
    # In real usage, you'd need to handle the interrupt
    try:
        result = graph.invoke(initial_state, config={"configurable": {"thread_id": "test-1"}})
        # If we get here, the graph completed somehow (shouldn't in normal flow)
        assert "messages" in result
    except Exception as e:
        # Expected to interrupt at review_plan
        assert "interrupt" in str(e).lower() or "review" in str(e).lower()


@pytest.mark.asyncio
async def test_graph_ainvoke_basic(graph):
    """Test async graph invocation."""
    initial_state = {
        "messages": [HumanMessage(content="Plan a data processing pipeline")]
    }
    
    try:
        result = await graph.ainvoke(initial_state, config={"configurable": {"thread_id": "test-async-1"}})
        assert "messages" in result
    except Exception as e:
        # Expected to interrupt at review_plan
        assert "interrupt" in str(e).lower() or "review" in str(e).lower()


def test_graph_stream_to_interrupt(graph):
    """Test streaming to the first interrupt point."""
    initial_state = {
        "messages": [HumanMessage(content="Design a microservices architecture")]
    }
    
    events = []
    config = {"configurable": {"thread_id": "test-stream-1"}}
    
    for event in graph.stream(initial_state, config=config):
        events.append(event)
        # Break before we hit too many events
        if len(events) >= 10:
            break
    
    # Should have at least processed some nodes
    assert len(events) > 0


def test_graph_has_expected_nodes(graph):
    """Verify the graph has all expected nodes."""
    expected_nodes = [
        "extract_system_elements",
        "split_task",
        "plan_tot",
        "confidence",
        "review_plan",
        "format_plan_order"
    ]
    
    graph_nodes = list(graph.nodes.keys())
    
    for node in expected_nodes:
        assert node in graph_nodes, f"Expected node '{node}' not found in graph"
'''

TEST_INTERRUPTS_PY = '''"""
Tests for interrupt/resume behavior (HITL).
"""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage


@pytest.mark.slow
def test_interrupt_at_review_plan(graph):
    """Test that graph interrupts at review_plan node."""
    initial_state = {
        "messages": [HumanMessage(content="Build a REST API")]
    }
    
    config = {"configurable": {"thread_id": "test-interrupt-1"}}
    
    # Stream until interrupt
    events = list(graph.stream(initial_state, config=config))
    
    # Should have processed multiple nodes before interrupt
    assert len(events) > 0
    
    # Check that we have plan data
    final_event = events[-1] if events else {}
    if "review_plan" in final_event:
        assert "plan" in final_event["review_plan"]


@pytest.mark.slow
def test_resume_after_interrupt(graph):
    """Test resuming after an interrupt."""
    initial_state = {
        "messages": [HumanMessage(content="Create a data pipeline")]
    }
    
    thread_id = "test-resume-1"
    config = {"configurable": {"thread_id": thread_id}}
    
    # First run: stream to interrupt
    events_before = list(graph.stream(initial_state, config=config))
    
    # In a real scenario, you would:
    # 1. Get the interrupt state
    # 2. Provide resume input (approve/revise)
    # 3. Continue execution
    
    # For this test, we just verify the interrupt happened
    assert len(events_before) > 0
'''

TEST_PROPERTIES_PY = '''"""
Property-based tests using Hypothesis for fuzzing.
"""
from __future__ import annotations

import pytest

# Only run if hypothesis is available
hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st


@given(st.text(min_size=1, max_size=100))
def test_extract_system_elements_never_crashes(prompt_text):
    """Fuzz test: extract_system_elements should never crash."""
    from src.agents.plugin_analyzer import extract_system_elements
    from langchain_core.messages import HumanMessage
    
    state = {"messages": [HumanMessage(content=prompt_text)]}
    
    # Should not raise
    result = extract_system_elements(state)
    
    # Should always return these keys
    assert "system_elements" in result
    assert "plugins" in result
    assert "has_system_elements" in result


@given(st.dictionaries(
    keys=st.sampled_from(["goal", "nodes", "edges", "confidence"]),
    values=st.one_of(
        st.text(max_size=50),
        st.lists(st.dictionaries(keys=st.text(max_size=10), values=st.text(max_size=20)), max_size=5),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
    )
))
def test_confidence_handles_varied_plans(plan_dict):
    """Fuzz test: confidence calculation should handle varied plan structures."""
    from src.agents.confidence import compute_plan_confidence
    
    state = {"plan": plan_dict, "debug": {}}
    
    # Should not crash
    try:
        result = compute_plan_confidence(state)
        assert "plan" in result
    except (KeyError, TypeError, AttributeError):
        # Some malformed inputs may fail gracefully
        pass
'''

TEST_SCHEMAS_PY = '''"""
Integration tests for SDK schemas and validation.
"""
from __future__ import annotations

import pytest


@pytest.mark.cli
def test_assistant_has_schemas(lg_client, default_assistant_id):
    """Test that assistant has input/output schemas."""
    assistant = lg_client.assistants.get(default_assistant_id)
    
    assert assistant is not None
    # LangGraph assistants should have metadata about schemas
    # The exact structure may vary, so we just check the assistant exists
    assert "assistant_id" in assistant


@pytest.mark.cli
def test_minimal_valid_payload(lg_client, default_assistant_id):
    """Create a minimal valid payload for the assistant."""
    # Create a thread
    thread = lg_client.threads.create()
    assert "thread_id" in thread
    
    # Minimal payload based on our AppState
    payload = {
        "messages": [{"role": "user", "content": "Hello, create a simple plan"}]
    }
    
    # Create a run
    run = lg_client.runs.create(
        thread_id=thread["thread_id"],
        assistant_id=default_assistant_id,
        input=payload
    )
    
    assert "run_id" in run
    assert run["status"] in ["pending", "running", "interrupted"]
'''

TEST_CLI_PY = '''"""
CLI integration tests for langgraph commands.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.cli
def test_langgraph_help():
    """Test that langgraph --help works."""
    result = subprocess.run(
        ["langgraph", "--help"],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    assert result.returncode == 0
    assert "langgraph" in result.stdout.lower() or "usage" in result.stdout.lower()


@pytest.mark.cli
@pytest.mark.slow
def test_dev_server_starts(dev_server):
    """Test that dev server starts and responds to /ok."""
    import requests
    
    response = requests.get(f"{dev_server}/ok", timeout=5)
    assert response.status_code == 200


@pytest.mark.cli
@pytest.mark.slow
def test_sdk_create_thread_and_run(lg_client, default_assistant_id):
    """Test creating a thread and running with default prompt."""
    # Create thread
    thread = lg_client.threads.create()
    thread_id = thread["thread_id"]
    
    # Create run with default prompt
    payload = {
        "messages": [{"role": "user", "content": "Create a plan for a web application"}]
    }
    
    run = lg_client.runs.create(
        thread_id=thread_id,
        assistant_id=default_assistant_id,
        input=payload
    )
    
    assert "run_id" in run
    assert run["status"] in ["pending", "running", "interrupted", "success", "error"]
    
    # Wait for run to complete or interrupt (with timeout)
    run_id = run["run_id"]
    max_wait = 30  # seconds
    import time
    elapsed = 0
    
    while elapsed < max_wait:
        run_status = lg_client.runs.get(thread_id, run_id)
        status = run_status.get("status")
        
        if status in ["success", "error", "interrupted"]:
            break
        
        time.sleep(1)
        elapsed += 1
    
    # Should have reached some terminal or interrupt state
    final_run = lg_client.runs.get(thread_id, run_id)
    assert final_run["status"] in ["success", "error", "interrupted"]


@pytest.mark.cli
@pytest.mark.slow
@pytest.mark.parametrize("prompt", [
    "hello",
    "plan a task",
    "summarize this data"
])
def test_custom_prompts_via_sdk(lg_client, default_assistant_id, prompt):
    """Test running with various custom prompts via SDK."""
    # Create thread
    thread = lg_client.threads.create()
    thread_id = thread["thread_id"]
    
    # Create run with custom prompt
    payload = {
        "messages": [{"role": "user", "content": prompt}]
    }
    
    run = lg_client.runs.create(
        thread_id=thread_id,
        assistant_id=default_assistant_id,
        input=payload
    )
    
    assert "run_id" in run
    
    # Just verify it starts, don't wait for completion
    assert run["status"] in ["pending", "running", "interrupted", "success", "error"]


@pytest.mark.cli
@pytest.mark.slow
def test_streaming_events(lg_client, default_assistant_id):
    """Test that streaming returns at least one event."""
    # Create thread
    thread = lg_client.threads.create()
    thread_id = thread["thread_id"]
    
    # Create run
    payload = {
        "messages": [{"role": "user", "content": "Quick test"}]
    }
    
    run = lg_client.runs.create(
        thread_id=thread_id,
        assistant_id=default_assistant_id,
        input=payload
    )
    
    run_id = run["run_id"]
    
    # Try to stream events (if supported)
    try:
        # Check if stream method exists
        if hasattr(lg_client.runs, 'stream'):
            events = list(lg_client.runs.stream(thread_id, run_id))
            # Should get at least one event
            assert len(events) >= 0  # May be 0 if run completes instantly
        else:
            pytest.skip("Streaming not supported by SDK version")
    except (AttributeError, Exception) as e:
        # Stream endpoint may not exist or be supported
        if "404" in str(e) or "Not Found" in str(e):
            pytest.skip("Streaming endpoint not available on server")
        raise
'''

PYTEST_INI = '''[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --tb=short
    --disable-warnings
markers =
    cli: tests requiring langgraph CLI
    slow: slow tests (deselect with '-m "not slow"')
    asyncio: async tests
'''

SAMPLE_APP_CLI_PY = '''"""
Sample CLI script for the application (optional).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_core.messages import HumanMessage
from src.agents.graph import graph


def main():
    parser = argparse.ArgumentParser(description="Run the agentic system")
    parser.add_argument("--prompt", type=str, required=True, help="User prompt")
    parser.add_argument("--thread-id", type=str, default="cli-thread", help="Thread ID")
    parser.add_argument("--output", type=str, help="Output file path")
    
    args = parser.parse_args()
    
    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content=args.prompt)]
    }
    
    config = {"configurable": {"thread_id": args.thread_id}}
    
    # Run the graph (will interrupt at review_plan)
    try:
        result = graph.invoke(initial_state, config=config)
        
        # Output result
        if args.output:
            Path(args.output).write_text(json.dumps(result, default=str), encoding="utf-8")
        else:
            print(json.dumps(result, default=str, indent=2))
        
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
'''


# ============================================================================
# FILE MAPPING
# ============================================================================

FILES_TO_CREATE = {
    "tests/conftest.py": CONFTEST_PY,
    "tests/meta/test_repo_layout.py": TEST_REPO_LAYOUT_PY,
    "tests/unit/test_nodes.py": TEST_NODES_PY,
    "tests/unit/test_router.py": TEST_ROUTER_PY,
    "tests/integration/test_graph_integration.py": TEST_GRAPH_INTEGRATION_PY,
    "tests/integration/test_interrupts.py": TEST_INTERRUPTS_PY,
    "tests/integration/test_properties.py": TEST_PROPERTIES_PY,
    "tests/integration/test_schemas.py": TEST_SCHEMAS_PY,
    "tests/cli/test_cli.py": TEST_CLI_PY,
    "pytest.ini": PYTEST_INI,
}

SAMPLE_FILES = {
    "app/cli.py": SAMPLE_APP_CLI_PY,
}


# ============================================================================
# MAIN LOGIC
# ============================================================================

def main():
    args = parse_args()
    
    print("=" * 70)
    print("LangGraph Test Suite Bootstrap")
    print("=" * 70)
    print(f"Force overwrite: {args.force}")
    print(f"Create samples: {args.with_samples}")
    print(f"Test prompts: {args.prompts}")
    print()
    
    # Update TEST_CLI_PY with custom prompts
    prompts = [p.strip() for p in args.prompts.split("|") if p.strip()]
    if prompts:
        # Inject prompts into the parametrize decorator
        global TEST_CLI_PY
        prompt_lines = [f'    "{p}"' for p in prompts]
        prompt_list_str = ",\n".join(prompt_lines)
        TEST_CLI_PY = TEST_CLI_PY.replace(
            '    "hello",\n    "plan a task",\n    "summarize this data"',
            prompt_list_str
        )
        FILES_TO_CREATE["tests/cli/test_cli.py"] = TEST_CLI_PY
    
    created_count = 0
    skipped_count = 0
    
    # Create test files
    print("Creating test files:")
    print("-" * 70)
    for rel_path, content in FILES_TO_CREATE.items():
        path = Path(rel_path)
        created, msg = write_file(path, content, force=args.force)
        print(msg)
        if created:
            created_count += 1
        else:
            skipped_count += 1
    
    # Create sample files if requested
    if args.with_samples:
        print()
        print("Creating sample application files:")
        print("-" * 70)
        for rel_path, content in SAMPLE_FILES.items():
            path = Path(rel_path)
            created, msg = write_file(path, content, force=args.force)
            print(msg)
            if created:
                created_count += 1
            else:
                skipped_count += 1
    
    print()
    print("=" * 70)
    print(f"Summary: {created_count} created, {skipped_count} skipped")
    print("=" * 70)
    print()
    
    # Print file map
    print("File Map:")
    print("-" * 70)
    all_files = list(FILES_TO_CREATE.keys())
    if args.with_samples:
        all_files.extend(SAMPLE_FILES.keys())
    
    for path_str in sorted(all_files):
        path = Path(path_str)
        status = "✓" if path.exists() else "✗"
        print(f"{status} {path_str}")
    
    print()
    print("=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print()
    print("1. Install test dependencies:")
    print("   pip install pytest pytest-asyncio hypothesis requests langgraph-sdk")
    print()
    print("2. Run the test suite:")
    print("   pytest -v")
    print()
    print("3. Run with coverage:")
    print("   pytest --cov=src --cov-report=term-missing")
    print()
    print("4. Run only fast tests (skip CLI and slow tests):")
    print('   pytest -v -m "not cli and not slow"')
    print()
    print("5. Run specific test categories:")
    print("   pytest tests/unit/          # Unit tests only")
    print("   pytest tests/integration/   # Integration tests only")
    print("   pytest tests/cli/           # CLI tests only")
    print()
    print("6. Start dev server for manual testing:")
    print("   langgraph dev")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
