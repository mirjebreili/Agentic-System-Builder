import json
from typing import Dict, Any

from src.utils.types import Registry
from src.utils.parsing import parse_first_message
from src.tools.adapters import (
    http_based_atlas_read_by_key,
    membased_atlas_key_stream_aggregator,
)


def build_registry(first_message: str) -> Registry:
    """
    Builds the registry by loading static tool adapters and merging them
    with any dynamically provided plugin specs from the user message.
    """
    # Start with the static adapters that are always available.
    registry: Registry = {
        "HttpBasedAtlasReadByKey": http_based_atlas_read_by_key.get_spec(),
        "membasedAtlasKeyStreamAggregator": membased_atlas_key_stream_aggregator.get_spec(),
    }

    # Parse the user message to get raw plugin docs.
    _, plugins_raw = parse_first_message(first_message)

    # Parse and merge the dynamic plugins.
    # We assume that the plugins_raw are JSON strings that represent ToolSpec.
    for plugin_str in plugins_raw:
        try:
            plugin_spec = json.loads(plugin_str)
            # Basic validation to ensure it's a valid spec.
            if "name" in plugin_spec and "description" in plugin_spec:
                name = plugin_spec["name"]
                registry[name] = plugin_spec
        except (json.JSONDecodeError, TypeError):
            # Silently ignore if a plugin string is not valid JSON.
            pass

    return registry


def get_question(first_message: str) -> str:
    """
    Extracts the question from the first message.
    """
    question, _ = parse_first_message(first_message)
    return question