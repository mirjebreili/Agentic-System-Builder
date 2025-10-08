from typing import Dict
from src.utils.types import Registry
from src.utils.parsing import parse_first_message
from src.tools.adapters import http_based_atlas_read_by_key, membased_atlas_key_stream_aggregator

def build_registry(first_message: str) -> Registry:
    """
    Builds the registry by parsing the first user message and merging
    with static tool adapters.
    """
    # For this implementation, we'll ignore the parsed plugins from the first message
    # and just use the static adapters.

    registry: Registry = {
        "HttpBasedAtlasReadByKey": http_based_atlas_read_by_key.get_spec(),
        "membasedAtlasKeyStreamAggregator": membased_atlas_key_stream_aggregator.get_spec(),
    }

    return registry

def get_question(first_message: str) -> str:
    """
    Extracts the question from the first message.
    """
    question, _ = parse_first_message(first_message)
    return question