# This file will contain the logic for generating the final report.
# It starts with the schemas for the report data.
from pydantic import BaseModel
from typing import Dict

# These imports will be circular if not handled carefully,
# but for schemas it's often fine. We will import the actual node functions
# inside the report generation function to avoid this.
from .planner import Plan
from .tests_node import TestsSummary


# ## 4.6 Report summary (`projects/<name>/reports/summary.json`)
# Schemas for the final summary report.

class ConfidenceDebug(BaseModel):
    """Debug information about the confidence score calculation."""
    self_score: float
    structural: float
    coverage: float
    prior: float
    final: float
    weights: Dict[str, float]

class ReportSummary(BaseModel):
    """The main schema for the summary.json report."""
    project_name: str
    goal: str
    plan: Plan
    confidence: float
    confidence_terms: ConfidenceDebug
    tests: TestsSummary
    executor_passed: bool
    sandbox_ok: bool
    sandbox_log_path: str
    satisfaction_score: float  # 0..1
    notes: str
