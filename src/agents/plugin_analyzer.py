from __future__ import annotations
import json
import logging
from typing import Any, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path

from agents.prompts_util import find_prompts_dir
from llm.client import get_chat_model
from utils.message_utils import extract_last_message_content

logger = logging.getLogger(__name__)

def analyze_plugins(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze Persian question and determine plugin sequence."""
    
    messages = state.get("messages", [])
    question = state.get("question", "")
    plugins = state.get("plugins", [])
    
    if not question and messages:
        question = extract_last_message_content(messages, "")
    
    if not question:
        state["error"] = "No question provided for plugin analysis"
        return state
        
    if not plugins:
        state["error"] = "No plugins provided for analysis"
        return state
    
    try:
        # Load prompts
        prompts_dir = find_prompts_dir()
        
        with open(prompts_dir / "plugin_analysis_system.jinja", "r", encoding="utf-8") as f:
            system_prompt = f.read()
            
        with open(prompts_dir / "plugin_analysis_user.jinja", "r", encoding="utf-8") as f:
            user_prompt_template = f.read()
        
        # Create plugin list for template
        plugin_names = []
        if isinstance(plugins, list):
            for plugin in plugins:
                if isinstance(plugin, dict):
                    plugin_names.append(plugin.get('name', str(plugin)))
                else:
                    plugin_names.append(str(plugin))
        
        # Render user prompt using simple template replacement
        user_prompt = user_prompt_template.replace("{{question}}", question)
        plugins_text = "\n".join([f"- {plugin}" for plugin in plugin_names])
        user_prompt = user_prompt.replace("{% for plugin in plugins %}\n- {{plugin}}\n{% endfor %}", plugins_text)
        
        # Create messages
        llm_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        # Get LLM response
        llm = get_chat_model()
        response = llm.invoke(llm_messages)
        response_content = response.content.strip()
        
        # Extract plugin sequence from response
        plugin_sequence = extract_plugin_sequence(response_content)
        
        state["plugin_sequence"] = plugin_sequence
        state["plan"] = {
            "type": "plugin_sequence",
            "sequence": plugin_sequence,
            "analysis": response_content,
            "question": question,
            "plugins_used": len(plugins)
        }
        
        logger.info(f"Determined plugin sequence: {plugin_sequence}")
        
    except Exception as e:
        state["error"] = f"Plugin analysis failed: {str(e)}"
        logger.error(f"Plugin analysis error: {str(e)}")
    
    return state

def extract_plugin_sequence(response_content: str) -> str:
    """Extract plugin sequence from LLM response."""
    lines = response_content.split('\n')
    
    for line in lines:
        line = line.strip()
        if '-->' in line:
            # Found a line with plugin sequence
            # Clean up the line to extract just the sequence
            if ':' in line:
                # Format like "Sequence: Plugin1 --> Plugin2"
                sequence = line.split(':', 1)[1].strip()
            else:
                sequence = line
            
            # Remove any markdown formatting
            sequence = sequence.replace('`', '').strip()
            return sequence
    
    # If no --> found, return the response as is
    return response_content.strip()

def parse_plugin_descriptions(plugin_texts: list) -> list:
    """Parse plugin text descriptions into structured format."""
    plugins = []
    
    for plugin_text in plugin_texts:
        lines = plugin_text.strip().split('\n')
        if not lines:
            continue
            
        name = lines[0].strip()
        description = '\n'.join(lines[1:]).strip()
        
        plugins.append({
            "name": name,
            "description": description
        })
    
    return plugins