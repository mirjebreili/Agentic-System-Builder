import re
from typing import List, Tuple


def parse_first_message(message: str) -> Tuple[str, List[str]]:
    """
    Parses the first user message to separate the question from plugin docs.
    The function handles both English and Persian text. It looks for a separator
    between the question and the plugin documentation.
    """
    # A separator is a line with 3 or more hyphens, asterisks, or equals signs,
    # a common convention in markdown for a horizontal rule.
    # We split on the first occurrence of such a separator.
    parts = re.split(r"\n\s*[-*=_]{3,}\s*\n", message, maxsplit=1)

    if len(parts) == 2:
        question, docs_str = parts
        question = question.strip()
        plugins_raw = [
            line.strip() for line in docs_str.strip().split("\n") if line.strip()
        ]
        return question, plugins_raw
    else:
        # If no separator is found, assume the entire message is the question.
        # This is a safe fallback.
        return message.strip(), []