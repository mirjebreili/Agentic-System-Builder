import pytest
from asb.agent.prompts_util import find_prompts_dir


def test_find_prompts_dir_and_files_readable():
    prompts_dir = find_prompts_dir()
    assert prompts_dir.exists(), "Prompts directory not found"
    assert prompts_dir.is_dir(), "Prompts path is not a directory"

    system_file = prompts_dir / "plan_system.jinja"
    user_file = prompts_dir / "plan_user.jinja"

    for file_path in (system_file, user_file):
        assert file_path.exists(), f"Missing prompt file: {file_path}"
        content = file_path.read_text().strip()
        assert content, f"Prompt file {file_path} is empty"

