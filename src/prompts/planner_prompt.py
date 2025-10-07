"""User-facing planner prompt template."""

from __future__ import annotations

PLANNER_PROMPT = """You must design a plugin execution plan that will help a downstream agent complete the user request.

# User Question
{user_question}

# Available Plugins
{plugin_descriptions}

# Plan Format
Return a JSON array where each entry has:
- "plugin": the name of the plugin to call.
- "description": a short justification for why it is needed.
- "args": a JSON object containing the arguments to supply.

Ensure the steps are ordered sequentially and omit any plugins that are not required."""
