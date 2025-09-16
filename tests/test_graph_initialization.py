from __future__ import annotations

from pathlib import Path


def test_make_graph_langgraph_api(monkeypatch):
    import asb.agent.graph as graph_module

    monkeypatch.setattr(graph_module, "running_on_langgraph_api", lambda: True)

    compile_kwargs: dict[str, object] = {}
    sentinel = object()

    def fake_compile(self, *, checkpointer=None):  # type: ignore[override]
        compile_kwargs["checkpointer"] = checkpointer
        return sentinel

    monkeypatch.setattr(graph_module.StateGraph, "compile", fake_compile)

    def raise_connect(*_args, **_kwargs):
        raise AssertionError("SQLite should not be initialised when using LangGraph Cloud")

    monkeypatch.setattr(graph_module.sqlite3, "connect", raise_connect)

    class RaiseSqliteSaver:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("SqliteSaver should not be constructed when using LangGraph Cloud")

    monkeypatch.setattr(graph_module, "SqliteSaver", RaiseSqliteSaver)

    result = graph_module._make_graph(path="/tmp/should_not_be_used.db")

    assert result is sentinel
    assert compile_kwargs["checkpointer"] is None


def test_make_graph_local_sqlite(monkeypatch, tmp_path):
    import asb.agent.graph as graph_module

    monkeypatch.setattr(graph_module, "running_on_langgraph_api", lambda: False)
    monkeypatch.delenv("ASB_DEV_SERVER", raising=False)

    compile_kwargs: dict[str, object] = {}
    sentinel = object()

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

    class DummySaver:
        def __init__(self, conn):
            self.conn = conn

    monkeypatch.setattr(graph_module, "SqliteSaver", DummySaver)

    db_path = tmp_path / "graph" / "state.db"
    result = graph_module._make_graph(path=str(db_path))

    assert result is sentinel
    checkpointer = compile_kwargs["checkpointer"]
    assert isinstance(checkpointer, DummySaver)
    assert checkpointer.conn is dummy_conn
    assert connections == {"path": str(db_path), "check_same_thread": False}
    assert created_dirs == [(str(Path(db_path).parent), True)]
