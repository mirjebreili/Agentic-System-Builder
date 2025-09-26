from __future__ import annotations

from pathlib import Path

from asb.agent.micro.import_resolver import import_resolver_node


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_import_resolver_creates_stubs_and_logs_disallowed_imports(tmp_path: Path) -> None:
    project_root = tmp_path
    agent_dir = project_root / "src" / "agent"

    alpha_source = (
        "from __future__ import annotations\n\n"
        "import math\n"
        "import requests\n"
        "from .beta import helper\n"
        "from .gamma.delta import make\n"
    )
    _write(agent_dir / "alpha.py", alpha_source)

    pyproject = (
        "[project]\n"
        "name = \"demo\"\n"
        "version = \"0.0.1\"\n"
        "dependencies = [\"langgraph>=0.6\"]\n"
    )
    _write(project_root / "pyproject.toml", pyproject)

    state = {"project_root": str(project_root)}

    result = import_resolver_node(state)

    beta_stub = agent_dir / "beta.py"
    gamma_stub = agent_dir / "gamma" / "delta.py"

    assert beta_stub.exists()
    assert "def helper" in beta_stub.read_text(encoding="utf-8")
    assert gamma_stub.exists()
    assert "def make" in gamma_stub.read_text(encoding="utf-8")

    gamma_init = agent_dir / "gamma" / "__init__.py"
    agent_init = agent_dir / "__init__.py"
    assert gamma_init.exists()
    assert agent_init.exists()

    scratch = result.get("scratch", {})
    resolver_log = scratch.get("import_resolver", {})
    illegal = resolver_log.get("illegal_imports") or []
    assert any(entry.get("module") == "requests" for entry in illegal)

    artifacts = scratch.get("artifacts", {}).get("imports", {})
    stubs = artifacts.get("stubs", [])
    assert str(beta_stub) in stubs
    assert str(gamma_stub) in stubs

    errors = result.get("errors") or []
    assert any("requests" in message for message in errors)
