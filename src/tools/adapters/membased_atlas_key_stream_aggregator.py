from typing import Literal, Dict, Any
from src.utils.types import ToolSpec

def get_spec() -> ToolSpec:
    """
    Returns the specification for the membasedAtlasKeyStreamAggregator tool.
    """
    return {
        "name": "membasedAtlasKeyStreamAggregator",
        "description": "A consumer/transformer that aggregates a stream of keys, summing the numeric suffixes of keys that match a given prefix.",
        "role": "consumer",
        "inputs": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The prefix of the keys to aggregate."},
                "index": {"type": "integer", "description": "The index of the keys to aggregate."}
            },
            "anyOf": [
                {"required": ["name"]},
                {"required": ["index"]}
            ]
        },
        "outputs": {
            "type": "number"
        }
    }