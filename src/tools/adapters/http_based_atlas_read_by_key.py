from __future__ import annotations

from ...utils.types import ToolSpec


TOOL_SPEC: ToolSpec = ToolSpec(
    name="HttpBasedAtlasReadByKey",
    description=(
        "Fetch Atlas records over HTTP given a namespace/project pair and optional "
        "search token. The operator yields JSON or stream payloads with ``data.result`` "
        "items containing key metadata."
    ),
    inputs={
        "args": {
            "namespace": "str (required)",
            "project": "str (optional)",
            "headers": "dict (optional)",
            "throwError": "bool (optional)",
        },
        "config": {
            "baseUrl": "HTTP endpoint for the Atlas service",
            "responseType": "json | stream (optional)",
        },
    },
    outputs={
        "type": "atlas_records",
        "shape": "data.result[*].keys -> List[str]",
    },
    role="producer",
    metadata={
        "produces": ["atlas_records"],
        "consumes": [],
        "aliases": ["partDeltaPluginHttpBasedAtlasReadByKey"],
    },
)


def get_spec() -> ToolSpec:
    return TOOL_SPEC
