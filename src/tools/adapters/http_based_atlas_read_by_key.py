from typing import Literal, Dict, Any
from src.utils.types import ToolSpec

def get_spec() -> ToolSpec:
    """
    Returns the specification for the HttpBasedAtlasReadByKey tool.
    """
    return {
        "name": "HttpBasedAtlasReadByKey",
        "description": "A producer tool that reads data from an HTTP-based Atlas service by key.",
        "role": "producer",
        "inputs": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "The key to read from Atlas."}
            },
            "required": ["key"]
        },
        "outputs": {
            "type": "stream/json",
            "path": "data.result.*",
            "keys_field": "keys"
        }
    }