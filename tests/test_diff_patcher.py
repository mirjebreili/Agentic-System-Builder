from __future__ import annotations

from asb.agent.micro.diff_patcher import diff_patcher_node


def test_diff_patcher_applies_minimal_diff(tmp_path) -> None:
    project_root = tmp_path / "proj"
    target_dir = project_root / "src" / "agent"
    target_dir.mkdir(parents=True)
    file_path = target_dir / "sample.py"
    original = "from langchain_core.messages import AIMessage\n\nresponse = AIMessage(content=\"old\")\n"
    file_path.write_text(original, encoding="utf-8")

    variant_code = original.replace("old", "new")
    state = {
        "scaffold": {"path": str(project_root)},
        "scratch": {
            "selected_variants": [
                {"path": "src/agent/sample.py", "code": variant_code}
            ]
        },
    }

    result = diff_patcher_node(state)
    updated = file_path.read_text(encoding="utf-8")
    assert updated == variant_code
    applied = result["scratch"]["applied_variants"][0]
    diff_lines = applied["diff"]
    assert any(line.startswith("+") for line in diff_lines)
    assert any(line.startswith("-") for line in diff_lines)
