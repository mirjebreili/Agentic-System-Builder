import re
from typing import Tuple, List, Dict

def parse_first_message(message: str) -> Tuple[str, List[str]]:
    """
    Parses the first user message to separate the question from plugin docs.
    The function handles both English and Persian text.
    """
    # A simple split based on a separator.
    parts = re.split(r'\n---\n', message, maxsplit=1)

    question = parts[0].strip()

    if len(parts) > 1:
        plugin_docs_raw = [line.strip() for line in parts[1].strip().split('\n') if line.strip()]
    else:
        plugin_docs_raw = []

    return question, plugin_docs_raw

def parse_plugin_docs(plugin_docs_raw: List[str]) -> Dict[str, str]:
    """
    Parses a list of raw plugin doc strings into a dictionary.
    Assumes the format "ToolName: Description".
    """
    parsed_plugins = {}
    for doc in plugin_docs_raw:
        match = re.match(r'(\w+):\s*(.*)', doc)
        if match:
            tool_name, description = match.groups()
            parsed_plugins[tool_name] = description.strip()
    return parsed_plugins