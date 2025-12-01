from src.agent.graph import graph

def test_runs_smoke():
    out = graph.invoke({"messages":[{"role":"human","content":"Create a tiny plan to write a hello-world script and list steps."}]})
    assert "messages" in out and isinstance(out["messages"], list)
