from typing import Dict
from src.utils.types import Registry
from src.utils.parsing import parse_first_message, parse_plugin_docs
from src.tools.adapters import http_based_atlas_read_by_key, membased_atlas_key_stream_aggregator

def build_registry(first_message: str) -> Registry:
    """
    Builds the registry by loading static tool adapters and then
    updating them with information parsed from the first user message.
    """
    # 1. Load static adapters as the base
    registry: Registry = {
        "HttpBasedAtlasReadByKey": http_based_atlas_read_by_key.get_spec(),
        "membasedAtlasKeyStreamAggregator": membased_atlas_key_stream_aggregator.get_spec(),
    }

    # 2. Parse the first message to get user-provided plugin docs
    _, plugin_docs_raw = parse_first_message(first_message)
    parsed_plugins = parse_plugin_docs(plugin_docs_raw)

    # 3. Merge the parsed info into the registry
    # This assumes the user is providing updated descriptions for existing tools.
    for tool_name, description in parsed_plugins.items():
        if tool_name in registry:
            registry[tool_name]["description"] = description
        # Note: This simple logic doesn't handle creating new tools from user docs,
        # as the current task focuses on enriching existing, statically-defined tools.

    return registry

def get_question(first_message: str) -> str:
    """
    Extracts the question from the first message.
    """
    question, _ = parse_first_message(first_message)
    return question