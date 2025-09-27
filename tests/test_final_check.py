from __future__ import annotations

import os
import pathlib

import pytest

from asb.agent.micro.final_check import final_check_node


@pytest.mark.skipif(os.getenv("CI") == "1", reason="needs local dev server")
def test_final_check_node_smoke():
    project_root = pathlib.Path(__file__).resolve().parents[1]
    initial_state = {"scaffold": {"project_root": str(project_root)}}

    result_state = final_check_node(initial_state)
    assert "final_check" in result_state

    final_result = result_state["final_check"]
    assert isinstance(final_result, dict)
    assert final_result.get("ok") in {True, False}

    if os.getenv("LANGGRAPH_API_KEY") and "used_api_key" in final_result:
        assert final_result.get("used_api_key") is True
