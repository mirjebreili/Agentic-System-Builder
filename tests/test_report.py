import json
from pathlib import Path

import pytest

from asb.agent.report import report


def test_report_creates_reports_directory(tmp_path: Path) -> None:
    state = {
        "scaffold": {"path": str(tmp_path)},
        "plan": {"goal": "Ship feature", "confidence": 0.5},
        "tests": ["pytest"],
        "passed": True,
    }

    result = report(state)

    summary_path = tmp_path / "reports" / "summary.json"
    assert summary_path.exists(), "Summary file should be written when scaffold path is provided"
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data["goal"] == "Ship feature"
    assert result["report"]["summary_path"] == str(summary_path)
    assert "summary" in result["report"]
    assert json.loads(result["final_response"]) == data


@pytest.mark.parametrize("state", [{}, {"plan": {"goal": "Fallback"}}])
def test_report_handles_missing_scaffold_path(state) -> None:
    result = report(state)

    assert "summary" in result["report"]
    assert "summary_path" not in result["report"]
    assert json.loads(result["final_response"]) == result["report"]["summary"]
