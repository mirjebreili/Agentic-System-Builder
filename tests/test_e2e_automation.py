import asyncio
from pathlib import Path

import pytest

from asb.testing.e2e_automation import E2ETestRunner, TEST_SCENARIOS


@pytest.mark.asyncio
@pytest.mark.slow
async def test_e2e_summarization_workflow(tmp_path: Path) -> None:
    """Test summarization workflow end-to-end."""

    runner = E2ETestRunner(tmp_path, timeout_seconds=120)
    scenario = next(s for s in TEST_SCENARIOS if s["name"] == "summarization_workflow")

    result = await runner.run_full_e2e_test(scenario)

    assert result.success, f"E2E test failed: {result.error}"
    assert result.api_result is not None
    assert result.api_result.processing_time < 30, "Processing took too long"
    assert "summary" in result.api_result.response_content.lower()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_all_e2e_scenarios(tmp_path: Path) -> None:
    """Run all E2E test scenarios."""

    runner = E2ETestRunner(tmp_path, timeout_seconds=180)
    report = await runner.run_all_e2e_tests()

    assert report.success_rate >= 0.8, f"Too many failures: {report.success_rate:.1%}"

    print("\n" + runner.generate_test_report(report))


if __name__ == "__main__":
    asyncio.run(test_all_e2e_scenarios(Path("/tmp/e2e-tests")))
