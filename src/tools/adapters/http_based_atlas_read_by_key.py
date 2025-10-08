# Static metadata for the HttpBasedAtlasReadByKey tool
METADATA = {
    "name": "HttpBasedAtlasReadByKey",
    "role": "Reads data from an HTTP-based atlas given a key.",
    "input_schema": {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "The key to read from the atlas."}
        },
        "required": ["key"]
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "value": {"type": "string", "description": "The value read from the atlas."}
        },
        "required": ["value"]
    }
}