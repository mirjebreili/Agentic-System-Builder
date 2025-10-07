from __future__ import annotations


from copy import deepcopy
from typing import Dict

from ..utils.types import Registry, ToolSpec
from .adapters.http_based_atlas_read_by_key import get_spec as get_http_reader
from .adapters.membased_atlas_key_stream_aggregator import (
    get_spec as get_stream_aggregator,
)


def _base_specs() -> Dict[str, ToolSpec]:
    specs = [get_http_reader(), get_stream_aggregator()]
    return {spec["name"]: deepcopy(spec) for spec in specs}


def build_registry(plugin_docs: Dict[str, str]) -> Registry:
    """Construct a registry by combining parsed docs with built-in adapters."""

    registry = _base_specs()

    alias_map: Dict[str, str] = {}
    for spec in registry.values():
        for alias in spec.get("metadata", {}).get("aliases", []):
            alias_map[alias.lower()] = spec["name"]

    for raw_name, documentation in plugin_docs.items():
        canonical = raw_name
        if canonical not in registry:
            canonical = alias_map.get(raw_name.lower(), raw_name)
        spec = registry.get(canonical)
        if spec:
            spec.setdefault("metadata", {})["documentation"] = documentation.strip()
        else:
            registry[canonical] = ToolSpec(
                name=canonical,
                description=documentation.strip(),
                inputs={},
                outputs={},
                role="mixed",
                metadata={"documentation": documentation.strip()},
            )

    return registry
