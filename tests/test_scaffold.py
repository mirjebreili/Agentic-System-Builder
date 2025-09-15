import pytest
from pathlib import Path
from asb.agent import scaffold


def test_missing_template_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(scaffold, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError) as exc:
        scaffold.scaffold_project({})
    expected = tmp_path / "src/config/settings.py"
    assert str(expected) in str(exc.value)
