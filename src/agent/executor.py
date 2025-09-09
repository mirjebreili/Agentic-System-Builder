from pydantic import BaseModel
from typing import List, Literal

# ## 4.3 Artifacts (produced by `execute_deep`)
# Schemas for the artifacts that the Deep Executor produces during its dry run.

class AcceptanceTest(BaseModel):
    """A single acceptance test to be run on the generated agent's output."""
    name: str
    kind: Literal["contains", "regex", "equals"]
    expected: str

class Artifacts(BaseModel):
    """A collection of artifacts harvested from the executor's dry run."""
    plan_run_trace: List[str]  # textual log of steps taken
    sample_outputs: List[str]  # 1–2 concise outputs representative of final agent
    acceptance_tests: List[AcceptanceTest]
