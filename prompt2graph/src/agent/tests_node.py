from pydantic import BaseModel

# ## 4.5 Self-test results
# Schemas for the results of the agent's self-tests.

class PlannerTestResult(BaseModel):
    """Result of the planner dry-run test."""
    ok: bool
    reason: str

class ExecutorTestResult(BaseModel):
    """Result of the executor dry-run test."""
    ok: bool
    steps_used: int
    reason: str

class TestsSummary(BaseModel):
    """A summary of all self-test results."""
    planner: PlannerTestResult
    executor: ExecutorTestResult
    overall_ok: bool
