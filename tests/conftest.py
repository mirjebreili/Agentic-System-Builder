"""
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
    from agents.graph import graph as app_graph
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
        "markers", 'slow: mark test as slow (deselect with \'-m "not slow"\')'
    )

