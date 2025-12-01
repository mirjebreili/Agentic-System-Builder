from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Tuple
from langchain_core.messages import HumanMessage, SystemMessage

# New infrastructure imports
from src.utils.prompt_manager import get_prompt_manager
from src.utils.retry import invoke_llm_with_retry
from src.utils.logger import get_logger, log_node_execution, PerformanceLogger
from src.utils.metrics import get_metrics_collector
from src.config.app_settings import settings
from src.llm.client import get_chat_model

# Get structured logger, metrics, and prompt manager
_logger = get_logger(__name__)
metrics = get_metrics_collector()
pm = get_prompt_manager()


@log_node_execution("extract_system_elements")
def extract_system_elements(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract system elements (plugins, APIs, components) from the initial user prompt.
    Uses pattern information from recognize_plugin_pattern node to guide extraction strategy.
    
    The extraction method is chosen based on the detected pattern:
    - JSON formats: Direct JSON parsing
    - Markdown: Regex pattern matching
    - Plain text: LLM semantic understanding
    - Mixed: Hybrid approach
    
    Returns updated state with:
    - system_elements: List of extracted element descriptions
    - plugins: List of plugin dictionaries with name, goal, etc.
    - has_system_elements: Boolean flag indicating if concrete system was provided
    """
    messages = state.get("messages", [])
    
    # Get pattern information from previous node
    pattern_info = state.get("plugin_definition_pattern", {})
    pattern_confidence = state.get("pattern_confidence", 0.5)
    
    _logger.info("Starting plugin extraction with pattern guidance", extra={
        "format_type": pattern_info.get("format_type", "unknown"),
        "confidence": pattern_confidence,
        "has_plugins_hint": pattern_info.get("has_plugins", False)
    })
    
    # Find the first human message (initial prompt)
    initial_prompt = None
    for msg in messages:
        # Handle both LangChain message objects and dict messages
        if hasattr(msg, "type") and msg.type == "human":
            initial_prompt = msg.content
            break
        elif isinstance(msg, dict) and msg.get("type") == "human":
            initial_prompt = msg.get("content")
            break
    
    if not initial_prompt:
        _logger.info("No initial prompt found, will use abstract planning")
        return {
            "system_elements": [],
            "plugins": [],
            "has_system_elements": False
        }
    
    # Check if pattern indicates no plugins
    if pattern_info.get("format_type") == "NONE" or not pattern_info.get("has_plugins", True):
        _logger.info("Pattern recognition indicates no plugins present")
        return {
            "system_elements": [],
            "plugins": [],
            "has_system_elements": False
        }
    
    # Extract using pattern-guided approach
    extracted_plugins = []
    system_elements = []
    
    format_type = pattern_info.get("format_type", "MIXED")
    extraction_strategy = pattern_info.get("extraction_strategy", "hybrid")
    
    _logger.info(f"Using extraction strategy: {extraction_strategy} for format: {format_type}")
    
    try:
        # Choose extraction method based on detected pattern
        if format_type in ["JSON_ARRAY", "JSON_OBJECT", "JSON_INLINE"] and extraction_strategy in ["json_parse", "json_extract_then_parse"]:
            # Try JSON parsing first for JSON formats
            _logger.info("Attempting direct JSON extraction")
            json_plugins, json_elements = _extract_json_plugins(initial_prompt, pattern_info)
            if json_plugins:
                extracted_plugins.extend(json_plugins)
                system_elements.extend(json_elements)
                _logger.info("JSON extraction successful", 
                           extra={"plugins_count": len(json_plugins)})
        
        # If JSON didn't work or format is not JSON, try LLM extraction
        if not extracted_plugins and extraction_strategy in ["llm_semantic", "hybrid"]:
            _logger.info("Attempting LLM-based semantic extraction")
            llm_plugins, llm_elements = _extract_plugins_with_llm(initial_prompt, pattern_info)
            if llm_plugins:
                extracted_plugins.extend(llm_plugins)
                system_elements.extend(llm_elements)
                _logger.info("LLM extraction successful", 
                           extra={"plugins_count": len(llm_plugins)})
        
        # Fallback to regex pattern matching for structured text
        if not extracted_plugins and format_type in ["MARKDOWN_LIST", "MARKDOWN_TABLE"]:
            _logger.info("Attempting regex pattern matching for markdown")
            components = _extract_component_list(initial_prompt)
            if components:
                system_elements.extend(components)
                _logger.info("Pattern matching found components", 
                           extra={"components_count": len(components)})
    
    except Exception as e:
        _logger.error("Error during guided extraction", 
                     extra={"error": str(e)}, exc_info=True)
    
    # Final fallback if all methods failed
    if not extracted_plugins and not system_elements:
        _logger.warning("All extraction methods failed, trying final fallback")
        try:
            # Try brute-force JSON extraction
            json_plugins, json_elements = _extract_json_plugins(initial_prompt, {})
            if json_plugins:
                extracted_plugins.extend(json_plugins)
                system_elements.extend(json_elements)
            
            # Try LLM without pattern hints
            if not extracted_plugins:
                llm_plugins, llm_elements = _extract_plugins_with_llm(initial_prompt, {})
                if llm_plugins:
                    extracted_plugins.extend(llm_plugins)
                    system_elements.extend(llm_elements)
        except Exception as e:
            _logger.error("Final fallback also failed", extra={"error": str(e)})
    
    has_elements = len(extracted_plugins) > 0 or len(system_elements) > 0
    
    _logger.info("System elements extraction complete", extra={
        "plugins_count": len(extracted_plugins),
        "elements_count": len(system_elements),
        "has_system_elements": has_elements
    })
    
    if not has_elements:
        _logger.warning("NO PLUGINS OR SYSTEM ELEMENTS FOUND", 
                      extra={"prompt_length": len(initial_prompt)})
        _logger.debug("Prompt preview for failed extraction", 
                    extra={"preview": initial_prompt[:500]})
    
    return {
        "system_elements": system_elements,
        "plugins": extracted_plugins,
        "has_system_elements": has_elements
    }


def _extract_plugins_with_llm(prompt: str, pattern_info: Dict[str, Any] = None) -> Tuple[List[Dict], List[str]]:
    """
    Use an LLM agent to extract plugins from the prompt.
    Uses pattern information to provide better context to the LLM.
    
    Args:
        prompt: The user's input text
        pattern_info: Pattern analysis from recognize_plugin_pattern node
    
    Returns:
        Tuple of (plugins, system_elements)
        - plugins: List of dicts with 'name', 'goal', 'type'
        - system_elements: List of string descriptions
    """
    pattern_info = pattern_info or {}
    
    # Render prompts using PromptManager with pattern context
    system_prompt = pm.render("plugin_extraction_system")
    
    # Add pattern hints to user prompt if available
    pattern_hints = ""
    if pattern_info:
        format_type = pattern_info.get("format_type", "unknown")
        language = pattern_info.get("language", "unknown")
        pattern_hints = f"\n\nPATTERN HINTS: Format detected as {format_type}, language: {language}"
        
        if pattern_info.get("extraction_hints"):
            hints = pattern_info["extraction_hints"]
            if hints.get("key_markers"):
                pattern_hints += f"\nKey markers: {', '.join(hints['key_markers'])}"
    
    user_prompt = pm.render("plugin_extraction_user", prompt_text=prompt) + pattern_hints
    
    # Call LLM with retry logic
    llm = get_chat_model(temperature=0.1)  # Low temperature for consistent extraction
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    with PerformanceLogger(_logger, "plugin_extraction_llm_call"):
        ai_message = invoke_llm_with_retry(llm, messages)
        response = ai_message.content
    
    
    _logger.debug("LLM response received", 
                extra={"response_length": len(response), "preview": response[:300]})
    
    # Parse LLM response (expecting JSON)
    plugins = []
    elements = []
    
    try:
        # Try to extract JSON from the response
        # LLM might wrap it in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to parse the whole response as JSON
            json_str = response.strip()
        
        data = json.loads(json_str)
        
        _logger.debug("Parsed JSON data", 
                    extra={"keys": list(data.keys()) if isinstance(data, dict) else "not a dict"})
        
        if data.get("has_plugins") and "plugins" in data:
            for plugin in data["plugins"]:
                if isinstance(plugin, dict) and plugin.get("name"):
                    plugin_info = {
                        "name": plugin.get("name", ""),
                        "goal": plugin.get("goal", "") or plugin.get("description", ""),
                        "type": plugin.get("type", "plugin")
                    }
                    plugins.append(plugin_info)
                    elements.append(f"{plugin_info['name']}: {plugin_info['goal']}")
        
        _logger.info("LLM successfully parsed plugins from JSON response", 
                   extra={"plugins_count": len(plugins)})
        
    except json.JSONDecodeError as e:
        _logger.warning("Failed to parse LLM response as JSON", 
                      extra={"error": str(e), "position": e.pos})
        _logger.debug("JSON parse error context", 
                    extra={"context": response[max(0, e.pos-50):e.pos+50] if e.pos else ""})
        _logger.debug("Full LLM response preview", extra={"preview": response[:500]})
    except Exception as e:
        _logger.warning("Error processing LLM response", 
                      extra={"error": str(e)}, exc_info=True)
    
    return plugins, elements


def _extract_json_plugins(prompt: str, pattern_info: Dict[str, Any] = None) -> Tuple[List[Dict], List[str]]:
    """
    Extract plugins from JSON structures in the prompt.
    Uses pattern information to optimize JSON extraction.
    
    Args:
        prompt: The user's input text
        pattern_info: Pattern analysis with extraction hints
    """
    pattern_info = pattern_info or {}
    plugins = []
    elements = []
    
    # Try to use JSONPath hint if available
    extraction_hints = pattern_info.get("extraction_hints", {})
    json_path = extraction_hints.get("json_path")
    
    _logger.debug("JSON extraction", extra={
        "json_path_hint": json_path,
        "pattern_structure": pattern_info.get("structure_pattern")
    })
    
    # First, try to find JSON array at the beginning (common case)
    if prompt.strip().startswith('['):
        try:
            # Try parsing entire prompt as JSON array
            data = json.loads(prompt)
            if isinstance(data, list):
                _logger.info("Successfully parsed prompt as direct JSON array")
                for item in data:
                    if isinstance(item, dict) and item.get("name"):
                        plugin_info = {
                            "name": item.get("name", ""),
                            "goal": item.get("goal", "") or item.get("description", ""),
                            "type": item.get("type", "plugin")
                        }
                        plugins.append(plugin_info)
                        elements.append(f"{plugin_info['name']}: {plugin_info['goal']}")
                return plugins, elements
        except json.JSONDecodeError as e:
            _logger.debug("Failed to parse as direct array", extra={"error": str(e)})
    
    # Find JSON blocks in text
    start_idx = prompt.find('{')
    if start_idx < 0:
        return plugins, elements
        
    # Find matching closing brace using stack-based approach
    brace_stack = []
    for i in range(start_idx, len(prompt)):
        if prompt[i] == '{':
            brace_stack.append(i)
        elif prompt[i] == '}':
            if brace_stack:
                brace_stack.pop()
            if not brace_stack:  # Found matching closing brace
                json_str = prompt[start_idx:i+1]
                try:
                    data = json.loads(json_str)
                    
                    # Look for plugins in various structures
                    plugin_sources = []
                    if "delta_Rules" in data and "plugins" in data["delta_Rules"]:
                        plugin_sources = data["delta_Rules"]["plugins"]
                    elif "plugins" in data:
                        plugin_sources = data["plugins"]
                    
                    # Extract plugin info
                    for plugin in plugin_sources:
                        if isinstance(plugin, dict) and plugin.get("name"):
                            plugin_info = {
                                "name": plugin.get("name", ""),
                                "goal": plugin.get("goal", "") or plugin.get("description", ""),
                                "type": "plugin"
                            }
                            plugins.append(plugin_info)
                            elements.append(f"{plugin_info['name']}: {plugin_info['goal']}")
                    
                    if plugins:
                        break
                        
                except json.JSONDecodeError:
                    continue
    
    return plugins, elements


def _extract_component_list(prompt: str) -> List[str]:
    """Extract component/service names from bullet points or lists."""
    components = []
    lines = prompt.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Look for bullet points, numbered lists, or plugin patterns
        match = re.match(r'^[-•*\d]+[.)]\s*(.+)$', line)
        if not match:
            # Try plugin pattern like @namespace/plugin - description
            match = re.match(r'^[@-]?\s*(@?\w+[/\w-]*)\s*[-:]\s*(.+)$', line)
        
        if match:
            component = match.group(1).strip()
            # Filter out obvious non-components
            if 3 < len(component) < 100:
                components.append(component)
    
    return components
