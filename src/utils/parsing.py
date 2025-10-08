import re
from typing import Tuple, List

def parse_first_message(message: str) -> Tuple[str, List[str]]:
    """
    Parses the first user message to separate the question from plugin docs.
    The function handles both English and Persian text.
    """
    # Use a regex to find the question part, which is assumed to be at the beginning of the message.
    # The plugin docs are assumed to be separated by a line of dashes or similar separator.
    parts = re.split(r'\n---\n', message, maxsplit=1)

    question = parts[0].strip()

    if len(parts) > 1:
        # The rest of the message contains plugin docs, separated by newlines.
        plugin_docs_raw = parts[1].strip().split('\n')
    else:
        plugin_docs_raw = []

    return question, plugin_docs_raw