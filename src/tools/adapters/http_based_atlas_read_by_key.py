from typing import Any, Dict, Literal

from src.utils.types import ToolSpec


def get_spec() -> ToolSpec:
    return {
        "name": "HttpBasedAtlasReadByKey",
        "description": "A producer that reads from a remote service and yields a stream of objects.",
        "role": "producer",
        "inputs": {
            "key": {"type": "string", "description": "The key to read from the source."}
        },
        "outputs": {
            "type": "stream/json",
            "path": "data.result.*",
            "keys_field": "keys",
        },
    }