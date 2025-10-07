from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .types import ParsedMessage


_PLUGIN_ALIASES: Dict[str, str] = {
    "partdeltapluginhttpbasedatlasreadbykey": "HttpBasedAtlasReadByKey",
    "httpbasedatlasreadbykey": "HttpBasedAtlasReadByKey",
    "membasedatlaskeystreamaggregator": "membasedAtlasKeyStreamAggregator",
}


_QUOTED_PATTERN = re.compile(r"[«\"“](.*?)[»\"”]")
_QUESTION_PREFIX = re.compile(r"^(?:سوال|سؤال|question|prompt)\s*[:：]\s*(.*)$", re.IGNORECASE | re.MULTILINE)


def _normalise(text: str) -> str:
    return " ".join(text.strip().split())


def _extract_question(section: str) -> str:
    """Pull the natural language question out of the leading section."""

    if not section:
        return ""

    prefix_match = _QUESTION_PREFIX.search(section)
    if prefix_match:
        extracted = prefix_match.group(1).strip()
        quotes = _QUOTED_PATTERN.search(extracted)
        if quotes:
            return _normalise(quotes.group(1))
        return _normalise(extracted)

    quotes = _QUOTED_PATTERN.search(section)
    if quotes:
        return _normalise(quotes.group(1))

    lines = [line.strip() for line in section.splitlines() if line.strip()]
    if lines:
        return _normalise(lines[-1])
    return ""


def _find_plugin_positions(text: str) -> List[Tuple[int, str, str]]:
    lowered = text.lower()
    positions: List[Tuple[int, str, str]] = []
    for alias, canonical in _PLUGIN_ALIASES.items():
        idx = lowered.find(alias)
        if idx != -1:
            positions.append((idx, alias, canonical))
    positions.sort(key=lambda item: item[0])
    return positions


def parse_first_message(message: str) -> ParsedMessage:
    """Split the first user message into question text and plugin documentation."""

    raw_text = message.strip()
    if not raw_text:
        return ParsedMessage(question="", plugin_docs={}, raw=message)

    positions = _find_plugin_positions(raw_text)
    first_plugin_idx = positions[0][0] if positions else len(raw_text)
    question_block = raw_text[:first_plugin_idx].strip()
    question = _extract_question(question_block)

    plugin_docs: Dict[str, str] = {}
    for current, alias, canonical in positions:
        next_idx_candidates = [pos for pos, *_ in positions if pos > current]
        next_idx = min(next_idx_candidates) if next_idx_candidates else len(raw_text)
        snippet = raw_text[current:next_idx].strip()
        plugin_docs[canonical] = snippet

    return ParsedMessage(question=question, plugin_docs=plugin_docs, raw=message)
