import pytest
from asb.agent.planner import PROMPTS_DIR, SYSTEM_PROMPT, USER_TMPL

def test_prompts_dir_exists():
    """
    Tests that the PROMPTS_DIR constant in the planner points to a real directory.
    """
    assert PROMPTS_DIR.exists(), f"Prompts directory not found at {PROMPTS_DIR}"
    assert PROMPTS_DIR.is_dir(), f"Prompts path {PROMPTS_DIR} is not a directory."

def test_prompt_files_are_readable():
    """
    Tests that the planner's prompt files can be read.
    """
    assert SYSTEM_PROMPT is not None
    assert len(SYSTEM_PROMPT) > 50
    assert "planner" in SYSTEM_PROMPT

    assert USER_TMPL is not None
    assert len(USER_TMPL) > 10
    assert "{{ user_goal }}" in USER_TMPL
