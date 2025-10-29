from __future__ import annotations
import json
import re
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def extract_system_elements(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract system elements (plugins, APIs, components) from the initial user prompt.
    Looks for JSON structures, plugin catalogues, API definitions, etc.
    
    Returns updated state with:
    - system_elements: List of extracted element descriptions
    - plugins: List of plugin dictionaries with name, goal, etc.
    - has_system_elements: Boolean flag indicating if concrete system was provided
    """
    messages = state.get("messages", [])
    
    # Find the first human message (initial prompt)
    initial_prompt = None
    for msg in messages:
        if hasattr(msg, "type") and msg.type == "human":
            initial_prompt = msg.content
            break
    
    if not initial_prompt:
        logger.info("No initial prompt found, will use abstract planning")
        return {
            "system_elements": [],
            "plugins": [],
            "has_system_elements": False
        }
    
    # Extract system elements
    extracted_plugins = []
    system_elements = []
    
    try:
        # Method 1: Look for JSON structures with plugins
        json_plugins, json_elements = _extract_json_plugins(initial_prompt)
        if json_plugins:
            extracted_plugins.extend(json_plugins)
            system_elements.extend(json_elements)
            logger.info(f"Extracted {len(json_plugins)} plugins from JSON")
        
        # Method 2: Look for component lists
        if not system_elements:
            components = _extract_component_list(initial_prompt)
            if components:
                system_elements.extend(components)
                logger.info(f"Extracted {len(components)} system components")
        
    except Exception as e:
        logger.warning(f"Error extracting system elements: {str(e)}")
    
    has_elements = len(extracted_plugins) > 0 or len(system_elements) > 0
    
    logger.info(f"System elements extraction complete: {len(extracted_plugins)} plugins, "
                f"{len(system_elements)} elements, has_system_elements={has_elements}")
    
    return {
        "system_elements": system_elements,
        "plugins": extracted_plugins,
        "has_system_elements": has_elements
    }


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
