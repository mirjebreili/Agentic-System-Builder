from __future__ import annotations
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

def analyze_plugins_simple(state: Dict[str, Any]) -> Dict[str, Any]:
    """Simple rule-based plugin sequencing for testing without LLM."""
    
    question = state.get("question", "")
    plugins = state.get("plugins", [])
    
    if not question:
        state["error"] = "No question provided for plugin analysis"
        return state
        
    if not plugins:
        state["error"] = "No plugins provided for analysis"
        return state
    
    try:
        # Simple rule-based logic for Persian questions
        question_lower = question.lower()
        
        # Check if it's asking for sum/total
        is_sum_question = any(word in question for word in ["مجموع", "جمع", "total", "sum"])
        
        # Check if it mentions keys with prefix
        has_prefix = any(word in question for word in ["شروع", "prefix", "با"])
        
        # Check if it mentions project/namespace (implies need to read data first)
        needs_data_read = any(word in question for word in ["پروژه", "project", "namespace", "فضای نام"])
        
        sequence = ""
        
        if is_sum_question:
            if needs_data_read or not has_specific_data_already(state):
                # Need to read data first, then aggregate
                sequence = "HttpBasedAtlasReadByKey --> membasedAtlasKeyStreamAggregator"
            else:
                # Data already available, just aggregate
                sequence = "membasedAtlasKeyStreamAggregator"
        else:
            # Default to reading data
            sequence = "HttpBasedAtlasReadByKey"
        
        state["plugin_sequence"] = sequence
        state["plan"] = {
            "type": "plugin_sequence", 
            "sequence": sequence,
            "analysis": f"Rule-based analysis for: {question}",
            "question": question,
            "plugins_used": len(plugins)
        }
        
        logger.info(f"Determined plugin sequence: {sequence}")
        
    except Exception as e:
        state["error"] = f"Plugin analysis failed: {str(e)}"
        logger.error(f"Plugin analysis error: {str(e)}")
    
    return state

def has_specific_data_already(state: Dict[str, Any]) -> bool:
    """Check if the state already contains specific data to work with."""
    return bool(state.get("data") or state.get("stream_data"))

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