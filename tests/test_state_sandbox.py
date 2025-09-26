from __future__ import annotations

from typing import Annotated, get_args, get_type_hints

from asb.agent import state as state_module


def test_app_state_declares_sandbox_field() -> None:
    annotations = get_type_hints(state_module.AppState, include_extras=True)
    assert "sandbox" in annotations
    sandbox_type = annotations["sandbox"]
    _, aggregator = get_args(sandbox_type)
    import operator

    assert aggregator is operator.or_
