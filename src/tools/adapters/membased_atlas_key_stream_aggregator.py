from __future__ import annotations

from ...utils.types import ToolSpec


TOOL_SPEC: ToolSpec = ToolSpec(
    name="membasedAtlasKeyStreamAggregator",
    description=(
        "Aggregate numeric suffixes from Atlas key streams. The operator consumes the "
        "HTTP reader output and sums trailing numeric values for keys filtered either "
        "by prefix (`name`) or index."
    ),
    inputs={
        "args": {
            "name": "str prefix (optional, required if index missing)",
            "index": "int (optional, required if name missing)",
        }
    },
    outputs={
        "type": "number",
        "shape": "sum of numeric suffixes",
    },
    role="consumer",
    metadata={
        "produces": ["numeric_sum"],
        "consumes": ["atlas_records"],
    },
)


def get_spec() -> ToolSpec:
    return TOOL_SPEC
