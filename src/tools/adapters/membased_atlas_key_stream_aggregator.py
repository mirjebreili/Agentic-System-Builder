from typing import Any, Dict, Literal

from src.utils.types import ToolSpec


def get_spec() -> ToolSpec:
    return {
        "name": "membasedAtlasKeyStreamAggregator",
        "description": "A consumer/transformer that aggregates a stream of keys, summing the numeric suffixes of keys that match a given prefix.",
        "role": "consumer",
        "inputs": {
            "name": {
                "type": "string",
                "description": "The prefix of the keys to aggregate.",
            },
            "index": {
                "type": "integer",
                "description": "The index of the keys to aggregate.",
            },
        },
        "outputs": {"type": "number"},
    }