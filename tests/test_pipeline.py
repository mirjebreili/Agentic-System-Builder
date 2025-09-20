from __future__ import annotations
import pytest
from langgraph.graph import StateGraph, START, END
from asb.agent.state import AppState
from asb.agent.planner import plan_tot
from asb.agent.scaffold import scaffold_project
from asb.agent.sandbox import sandbox_smoke

@pytest.mark.asyncio
async def test_full_pipeline():
    """
    Tests the full pipeline from prompt to sandbox-verified project.
    """
    # 1. Define the graph
    g = StateGraph(AppState)
    g.add_node("plan_tot", plan_tot)
    g.add_node("scaffold_project", scaffold_project)
    g.add_node("sandbox_smoke", sandbox_smoke)
    g.add_edge(START, "plan_tot")
    g.add_edge("plan_tot", "scaffold_project")
    g.add_edge("scaffold_project", "sandbox_smoke")
    g.add_edge("sandbox_smoke", END)
    graph = g.compile()

    # 2. Define the initial state
    initial_state = AppState(
        messages=[{"role": "user", "content": "Create an agentic summarizer that takes input text and returns a summarized version."}]
    )

    # 3. Run the graph
    final_state = await graph.ainvoke(initial_state)

    # 4. Assert the result
    assert final_state is not None
    assert "sandbox" in final_state
    assert final_state["sandbox"].get("ok") is True, f"Sandbox failed. Log: {final_state['sandbox'].get('log_path')}"
