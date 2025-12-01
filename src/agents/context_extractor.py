"""
Context Extractor Node - Plan B Refactoring

This node merges three previous nodes into one efficient operation:
- extract_goal (extracts user's objective)
- recognize_plugin_pattern (detects format)
- extract_system_elements (parses plugins/components)

Handles both concrete mode (with system elements) and abstract mode (goal only).
"""

from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Tuple, Optional
from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.prompt_manager import get_prompt_manager
from src.utils.retry import invoke_llm_with_retry
from src.utils.logger import get_logger, log_node_execution, PerformanceLogger
from src.llm.client import get_chat_model

_logger = get_logger(__name__)
pm = get_prompt_manager()

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def _extract_json(text: str) -> str:
    """Extract JSON from markdown code blocks."""
    m = _JSON_BLOCK.search(text or "")
    return m.group(1).strip() if m else (text or "").strip()


def _extract_last_message_content(messages) -> str:
    """Extract content from the last human message."""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content
        elif isinstance(msg, dict) and msg.get("type") == "human":
            return msg.get("content", "")
    return ""


def _detect_format(text: str) -> Dict[str, Any]:
    """
    Quickly detect the format type without LLM call.
    
    Returns:
        Dict with format_type, has_plugins, extraction_strategy
    """
    text_lower = text.lower()
    
    # Check for JSON
    has_json_structure = False
    try:
        if "{" in text and "}" in text:
            # Try to find JSON block
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                json.loads(json_match.group())
                has_json_structure = True
    except:
        pass
    
    # Check for common plugin indicators
    has_plugin_keywords = any(kw in text_lower for kw in [
        "plugin", "api", "function", "component", "tool", "service", "module"
    ])
    
    # Detect format type
    if has_json_structure:
        if "plugins" in text_lower or "components" in text_lower:
            format_type = "JSON_ARRAY"
        else:
            format_type = "JSON_OBJECT"
    elif re.search(r'^\s*[-*]\s+', text, re.MULTILINE):
        format_type = "MARKDOWN"
    elif has_plugin_keywords:
        format_type = "PLAIN_TEXT"
    else:
        format_type = "NONE"
    
    has_plugins = has_json_structure or has_plugin_keywords
    
    return {
        "format_type": format_type,
        "has_plugins": has_plugins,
        "extraction_strategy": "json_parse" if has_json_structure else "llm_semantic"
    }


def _parse_json_plugins(text: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Extract plugins from JSON format.
    
    Returns:
        Tuple of (plugins list, goal string)
    """
    plugins = []
    goal = ""
    
    try:
        # Find JSON in text
        json_match = re.search(r'\{[\s\S]*\}', text)
        if not json_match:
            return [], text
        
        data = json.loads(json_match.group())
        
        # Extract plugins
        if "plugins" in data:
            plugins_data = data["plugins"]
            if isinstance(plugins_data, list):
                for p in plugins_data:
                    if isinstance(p, dict):
                        plugins.append({
                            "name": p.get("name", "unknown"),
                            "goal": p.get("goal", p.get("description", "")),
                            "dependencies": p.get("dependencies", [])
                        })
        
        # Extract goal - look for text outside JSON or in "goal" field
        if "goal" in data:
            goal = data["goal"]
        else:
            # Extract text before/after JSON
            json_start = json_match.start()
            json_end = json_match.end()
            before = text[:json_start].strip()
            after = text[json_end:].strip()
            goal = (before + " " + after).strip()
        
        if not goal:
            goal = "Process using the provided plugins"
        
        _logger.info(f"JSON parsing extracted {len(plugins)} plugins")
        
    except json.JSONDecodeError as e:
        _logger.warning(f"JSON parsing failed: {e}")
        return [], text
    
    return plugins, goal


def _parse_markdown_plugins(text: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Extract plugins from Markdown list format.
    
    Returns:
        Tuple of (plugins list, goal string)
    """
    plugins = []
    goal_lines = []
    
    lines = text.split('\n')
    in_list = False
    current_plugin = None
    
    for line in lines:
        stripped = line.strip()
        
        # Check if line is a list item
        if re.match(r'^[-*]\s+', stripped):
            in_list = True
            # Extract plugin name
            name = re.sub(r'^[-*]\s+', '', stripped)
            # Remove markdown formatting
            name = re.sub(r'\*\*(.+?)\*\*', r'\1', name)
            name = re.sub(r'`(.+?)`', r'\1', name)
            
            if current_plugin:
                plugins.append(current_plugin)
            
            current_plugin = {
                "name": name.split(':')[0].strip() if ':' in name else name,
                "goal": name.split(':')[1].strip() if ':' in name else name,
                "dependencies": []
            }
        elif not stripped:
            if current_plugin:
                plugins.append(current_plugin)
                current_plugin = None
            in_list = False
        elif not in_list and stripped:
            goal_lines.append(stripped)
    
    if current_plugin:
        plugins.append(current_plugin)
    
    goal = ' '.join(goal_lines) if goal_lines else "Process the listed items"
    
    _logger.info(f"Markdown parsing extracted {len(plugins)} plugins")
    
    return plugins, goal


def _extract_with_llm(text: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Use LLM to extract plugins and goal from plain text.
    
    Returns:
        Tuple of (plugins list, goal string)
    """
    llm = get_chat_model()
    
    system_prompt = pm.render("plugin_extraction_system")
    user_prompt = pm.render("plugin_extraction_user", user_input=text)
    
    try:
        with PerformanceLogger(_logger, "extract_context_llm_call"):
            response = invoke_llm_with_retry(
                llm, 
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
        
        result_text = response.content
        json_str = _extract_json(result_text)
        data = json.loads(json_str)
        
        plugins = data.get("plugins", [])
        goal = data.get("goal", text)
        
        _logger.info(f"LLM extraction found {len(plugins)} plugins")
        
        return plugins, goal
        
    except Exception as e:
        _logger.warning(f"LLM extraction failed: {e}", exc_info=True)
        return [], text


@log_node_execution("extract_context")
def extract_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract user goal and system elements (plugins/components) in one efficient pass.
    
    This node replaces:
    - extract_goal
    - recognize_plugin_pattern  
    - extract_system_elements
    
    It handles both modes:
    - Concrete: User provides system elements (JSON/Markdown/Text)
    - Abstract: User provides only a goal
    
    Args:
        state: Application state with messages
        
    Returns:
        Updated state with:
        - goal: User's objective
        - plugins: List of plugin dictionaries
        - system_elements: List of element names
        - has_system_elements: Boolean flag
    """
    messages = state.get("messages", [])
    
    # Extract initial prompt
    initial_prompt = _extract_last_message_content(messages)
    
    if not initial_prompt:
        _logger.warning("No initial prompt found")
        return {
            "goal": "",
            "plugins": [],
            "system_elements": [],
            "has_system_elements": False
        }
    
    _logger.info("Starting context extraction", extra={
        "prompt_length": len(initial_prompt),
        "prompt_preview": initial_prompt[:100]
    })
    
    # Detect format quickly (no LLM call)
    format_info = _detect_format(initial_prompt)
    
    _logger.info("Format detected", extra={
        "format_type": format_info["format_type"],
        "has_plugins": format_info["has_plugins"]
    })
    
    # Extract based on detected format
    plugins = []
    goal = initial_prompt
    
    if format_info["format_type"] == "NONE":
        # Pure abstract mode - no plugins
        _logger.info("Abstract mode: no system elements detected")
        goal = initial_prompt.strip()
        
    elif format_info["format_type"] in ["JSON_ARRAY", "JSON_OBJECT"]:
        # Try JSON parsing
        plugins, extracted_goal = _parse_json_plugins(initial_prompt)
        if extracted_goal:
            goal = extracted_goal
        
    elif format_info["format_type"] == "MARKDOWN":
        # Try markdown parsing
        plugins, extracted_goal = _parse_markdown_plugins(initial_prompt)
        if extracted_goal:
            goal = extracted_goal
    
    # Fallback to LLM if no plugins found but format suggests there should be
    if not plugins and format_info["has_plugins"]:
        _logger.info("Falling back to LLM extraction")
        plugins, extracted_goal = _extract_with_llm(initial_prompt)
        if extracted_goal:
            goal = extracted_goal
    
    # Clean up goal
    goal = goal.strip()
    if not goal:
        goal = "Complete the task using available resources"
    
    # Extract system element names
    system_elements = [p.get("name", f"element_{i}") for i, p in enumerate(plugins)]
    has_system_elements = len(plugins) > 0
    
    _logger.info("Context extraction complete", extra={
        "goal_length": len(goal),
        "plugins_count": len(plugins),
        "has_system_elements": has_system_elements,
        "system_elements": system_elements[:5]  # Log first 5
    })
    
    return {
        "goal": goal,
        "plugins": plugins,
        "system_elements": system_elements,
        "has_system_elements": has_system_elements
    }
