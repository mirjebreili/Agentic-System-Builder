from src.agent.planner import plan_tot, Plan

def test_plan_tot_shape():
    state = {"messages":[{"role":"user","content":"Summarize 3 expenses by category."}]}
    out = plan_tot(state)
    plan = Plan.model_validate(out["plan"])
    nodes = [n.id for n in plan.nodes]
    assert nodes == ["plan","do","finish"]
    assert plan.confidence is None or (0.0 <= plan.confidence <= 1.0)
