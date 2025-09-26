from __future__ import annotations

import sys
import types
from pathlib import Path


def _install_langfuse_stub(monkeypatch):
    langfuse_module = types.ModuleType("langfuse")

    class _Client:
        def auth_check(self) -> bool:  # pragma: no cover - trivial
            return False

    def get_client():
        return _Client()

    langfuse_module.get_client = get_client  # type: ignore[attr-defined]

    langchain_module = types.ModuleType("langfuse.langchain")

    class CallbackHandler:  # pragma: no cover - simple stub
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    langchain_module.CallbackHandler = CallbackHandler  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "langfuse", langfuse_module)
    monkeypatch.setitem(sys.modules, "langfuse.langchain", langchain_module)


def test_make_graph_langgraph_api(monkeypatch):
    _install_langfuse_stub(monkeypatch)
    import asb.agent.graph as graph_module

    monkeypatch.setattr(graph_module, "running_on_langgraph_api", lambda: True)
    sqlite_module = sys.modules.get("langgraph.checkpoint.sqlite")
    if sqlite_module is None:
        sqlite_module = types.ModuleType("langgraph.checkpoint.sqlite")
        sqlite_package = types.ModuleType("langgraph.checkpoint")
        sqlite_package.sqlite = sqlite_module  # type: ignore[attr-defined]
        langgraph_package = types.ModuleType("langgraph")
        langgraph_package.checkpoint = sqlite_package  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "langgraph", langgraph_package)
        monkeypatch.setitem(sys.modules, "langgraph.checkpoint", sqlite_package)
        monkeypatch.setitem(sys.modules, "langgraph.checkpoint.sqlite", sqlite_module)

    class RaiseSqliteSaver:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("SqliteSaver should not be constructed when using LangGraph Cloud")

    monkeypatch.setattr(sqlite_module, "SqliteSaver", RaiseSqliteSaver, raising=False)

    compile_kwargs: dict[str, object] = {}

    class DummyCompiled:
        def __init__(self) -> None:
            self.configs: list[dict[str, object]] = []

        def with_config(self, config: dict[str, object]):
            self.configs.append(config)
            return self

    sentinel = DummyCompiled()

    def fake_compile(self, *, checkpointer=None):  # type: ignore[override]
        compile_kwargs["checkpointer"] = checkpointer
        return sentinel

    monkeypatch.setattr(graph_module.StateGraph, "compile", fake_compile)

    def raise_connect(*_args, **_kwargs):
        raise AssertionError("SQLite should not be initialised when using LangGraph Cloud")

    monkeypatch.setattr(graph_module.sqlite3, "connect", raise_connect)

    result = graph_module._make_graph(path="/tmp/should_not_be_used.db")

    assert result is sentinel
    assert compile_kwargs["checkpointer"] is None


def test_make_graph_local_sqlite(monkeypatch, tmp_path):
    _install_langfuse_stub(monkeypatch)
    import asb.agent.graph as graph_module

    monkeypatch.setattr(graph_module, "running_on_langgraph_api", lambda: False)
    monkeypatch.delenv("ASB_DEV_SERVER", raising=False)

    compile_kwargs: dict[str, object] = {}

    class DummyCompiled:
        def __init__(self) -> None:
            self.configs: list[dict[str, object]] = []

        def with_config(self, config: dict[str, object]):
            self.configs.append(config)
            return self

    sentinel = DummyCompiled()

    def fake_compile(self, *, checkpointer=None):  # type: ignore[override]
        compile_kwargs["checkpointer"] = checkpointer
        return sentinel

    monkeypatch.setattr(graph_module.StateGraph, "compile", fake_compile)

    created_dirs: list[tuple[str, bool]] = []

    def fake_makedirs(path: str, exist_ok: bool):
        created_dirs.append((path, exist_ok))

    monkeypatch.setattr(graph_module.os, "makedirs", fake_makedirs)

    connections: dict[str, object] = {}
    dummy_conn = object()

    def fake_connect(path: str, check_same_thread: bool):
        connections["path"] = path
        connections["check_same_thread"] = check_same_thread
        return dummy_conn

    monkeypatch.setattr(graph_module.sqlite3, "connect", fake_connect)

    sqlite_module = sys.modules.get("langgraph.checkpoint.sqlite")
    if sqlite_module is None:
        sqlite_module = types.ModuleType("langgraph.checkpoint.sqlite")
        sqlite_package = types.ModuleType("langgraph.checkpoint")
        sqlite_package.sqlite = sqlite_module  # type: ignore[attr-defined]
        langgraph_package = types.ModuleType("langgraph")
        langgraph_package.checkpoint = sqlite_package  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "langgraph", langgraph_package)
        monkeypatch.setitem(sys.modules, "langgraph.checkpoint", sqlite_package)
        monkeypatch.setitem(sys.modules, "langgraph.checkpoint.sqlite", sqlite_module)

    class DummySaver:
        def __init__(self, conn):
            self.conn = conn

    monkeypatch.setattr(sqlite_module, "SqliteSaver", DummySaver, raising=False)

    db_path = tmp_path / "graph" / "state.db"
    result = graph_module._make_graph(path=str(db_path))

    assert result is sentinel
    checkpointer = compile_kwargs["checkpointer"]
    assert checkpointer is not None
    if isinstance(checkpointer, DummySaver):
        assert checkpointer.conn is dummy_conn
    assert connections == {"path": str(db_path), "check_same_thread": False}
    assert created_dirs == [(str(Path(db_path).parent), True)]
