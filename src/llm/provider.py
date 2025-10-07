from __future__ import annotations

import os
from string import Template
from typing import Any, Dict, Optional


def _render(prompt: str, variables: Optional[Dict[str, Any]] = None) -> str:
    if not variables:
        return prompt
    return Template(prompt).safe_substitute({key: str(value) for key, value in variables.items()})


def call_llm(prompt: str, variables: Optional[Dict[str, Any]] = None, *, model: Optional[str] = None) -> Dict[str, Any]:
    """Minimal LLM adapter.

    The project defaults to a stubbed responder so tests remain deterministic. If an
    actual provider is requested via the ``LLM_PROVIDER`` environment variable the
    caller is instructed to configure their own integration.
    """

    provider = os.getenv("LLM_PROVIDER", "stub").lower()
    rendered_prompt = _render(prompt, variables)

    if provider in {"stub", "local"}:
        return {"content": rendered_prompt}

    raise RuntimeError(
        "LLM provider '%s' is not configured. Set LLM_PROVIDER=stub for the built-in "
        "heuristic responder or extend `call_llm` with the desired integration." % provider
    )
