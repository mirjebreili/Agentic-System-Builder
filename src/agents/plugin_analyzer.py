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
    Uses an LLM agent to intelligently detect and extract plugins from various formats:
    - JSON structures
    - Markdown lists (bullet points, numbered)
    - Simple text descriptions
    - Plugin patterns like @namespace/plugin-name
    
    Falls back to pattern matching if LLM extraction fails.
    
    Returns updated state with:
    - system_elements: List of extracted element descriptions
    - plugins: List of plugin dictionaries with name, goal, etc.
    - has_system_elements: Boolean flag indicating if concrete system was provided
    """
    messages = state.get("messages", [])
    
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
        _logger.debug("Checked messages for initial prompt", 
                    extra={"messages_count": len(messages)})
        return {
            "system_elements": [],
            "plugins": [],
            "has_system_elements": False
        }
    
    # Try LLM-based extraction first
    extracted_plugins = []
    system_elements = []
    
    try:
        _logger.info("Attempting LLM-based plugin extraction", 
                   extra={"prompt_length": len(initial_prompt)})
        _logger.debug("Prompt preview", extra={"preview": initial_prompt[:200]})
        llm_plugins, llm_elements = _extract_plugins_with_llm(initial_prompt)
        if llm_plugins:
            extracted_plugins.extend(llm_plugins)
            system_elements.extend(llm_elements)
            _logger.info("LLM extracted plugins successfully", 
                       extra={"plugins_count": len(llm_plugins), 
                              "plugin_names": [p['name'] for p in llm_plugins]})
        else:
            _logger.info("LLM extraction returned no plugins")
    except Exception as e:
        _logger.warning("LLM extraction failed, falling back to pattern matching", 
                      extra={"error": str(e)}, exc_info=True)
    
    # Fallback to pattern matching if LLM didn't find anything
    if not extracted_plugins:
        try:
            _logger.info("LLM found no plugins, trying pattern matching fallback")
            # Method 1: Look for JSON structures with plugins
            json_plugins, json_elements = _extract_json_plugins(initial_prompt)
            if json_plugins:
                extracted_plugins.extend(json_plugins)
                system_elements.extend(json_elements)
                _logger.info("Pattern matching extracted plugins from JSON", 
                           extra={"plugins_count": len(json_plugins),
                                  "plugin_names": [p['name'] for p in json_plugins]})
            else:
                _logger.info("JSON pattern matching found no plugins")
            
            # Method 2: Look for component lists
            if not system_elements:
                components = _extract_component_list(initial_prompt)
                if components:
                    system_elements.extend(components)
                    _logger.info("Pattern matching extracted system components", 
                               extra={"components_count": len(components), "components": components})
                else:
                    _logger.info("Component list extraction found no elements")
            
        except Exception as e:
            _logger.warning("Error in pattern matching fallback", 
                          extra={"error": str(e)}, exc_info=True)
    
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


def _extract_plugins_with_llm(prompt: str) -> Tuple[List[Dict], List[str]]:
    """
    Use an LLM agent to extract plugins from the prompt.
    The LLM can recognize plugins in various formats: JSON, markdown, plain text.
    
    Returns:
        Tuple of (plugins, system_elements)
        - plugins: List of dicts with 'name', 'goal', 'type'
        - system_elements: List of string descriptions
    """
    # Render prompts using PromptManager
    system_prompt = pm.render("plugin_extraction_system")
    user_prompt = pm.render("plugin_extraction_user", prompt_text=prompt)
    
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


def _extract_json_plugins(prompt: str) -> Tuple[List[Dict], List[str]]:
    """Extract plugins from JSON structures in the prompt."""
    plugins = []
    elements = []
    
    # Find JSON blocks
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
