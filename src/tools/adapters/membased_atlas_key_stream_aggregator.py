# Static metadata for the membasedAtlasKeyStreamAggregator tool
METADATA = {
    "name": "membasedAtlasKeyStreamAggregator",
    "role": "Aggregates a stream of keys from a memory-based atlas.",
    "input_schema": {
        "type": "object",
        "properties": {
            "key_stream": {
                "type": "array",
                "items": {"type": "string"},
                "description": "A stream of keys to aggregate."
            }
        },
        "required": ["key_stream"]
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "aggregated_value": {
                "type": "string",
                "description": "The aggregated value from the key stream."
            }
        },
        "required": ["aggregated_value"]
    }
}