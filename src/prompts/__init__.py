"""Prompt templates shared across the planner and executor."""

from __future__ import annotations

PLANNER_SYSTEM_PROMPT = (
    "You are a planning assistant that decides which plugins to run and in what order. "
    "Produce a concise ordered plan that can be handed to an automated executor."
)

PLANNER_USER_PROMPT_TEMPLATE = """\
You must design a plugin execution plan that will help a downstream agent complete the
user request.

# Task
{input_text}

# Available Plugins
{available_plugins}

# Plan Format
Return a JSON array following this template:
{plan_format}

Each entry must include the plugin name to call, a short justification, and the
arguments to supply. Ensure the steps are in the order they should execute.
"""

PLANNER_PLAN_FORMAT = """\
[
  {{
    "plugin": "<plugin_name>",
    "description": "<why this plugin is used>",
    "args": {{}}
  }}
]
"""

