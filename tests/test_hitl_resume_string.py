from asb.agent import hitl


def _run_review(monkeypatch, action: str):
    monkeypatch.setattr(hitl, "interrupt", lambda payload: action)
    state = {"plan": {}}
    return hitl.review_plan(state)


def test_review_plan_accepts_approve_string(monkeypatch):
    state = _run_review(monkeypatch, "approve")
    assert state["review"]["action"] == "approve"
    assert state["replan"] is False


def test_review_plan_accepts_revise_string(monkeypatch):
    state = _run_review(monkeypatch, "revise")
    assert state["review"]["action"] == "revise"
    assert state["replan"] is True

