from src.agent.executor import execute_deep

def test_executor_finishes():
    demo = {"plan": {"goal":"demo",
                     "nodes":[
                       {"id":"plan","type":"llm","prompt":"List two steps then DONE."},
                       {"id":"do","type":"llm","prompt":"Say STEP 1, then STEP 2, then DONE."},
                       {"id":"finish","type":"llm","prompt":"Summarize in one line."}],
                     "edges":[
                       {"from":"plan","to":"do"},
                       {"from":"do","to":"do","if":"more_steps"},
                       {"from":"do","to":"finish","if":"steps_done"}]},
            "messages":[],"flags":{"more_steps":True,"steps_done":False}}
    out = execute_deep(demo)
    assert any(m.get("content","").startswith("[finish]") for m in out["messages"])
