from __future__ import annotations
import json
import re
from typing import Any, Dict, Optional
from langchain_core.messages import HumanMessage, SystemMessage

# Infrastructure imports
from src.utils.prompt_manager import get_prompt_manager
from src.utils.retry import invoke_llm_with_retry
from src.utils.logger import get_logger, log_node_execution, PerformanceLogger
from src.utils.metrics import get_metrics_collector
from src.llm.client import get_chat_model

# Get structured logger, metrics, and prompt manager
_logger = get_logger(__name__)
metrics = get_metrics_collector()
pm = get_prompt_manager()

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def _extract_json(text: str) -> str:
    """Extract JSON from markdown code blocks."""
    m = _JSON_BLOCK.search(text or "")
    return m.group(1).strip() if m else (text or "").strip()


@log_node_execution("recognize_plugin_pattern")
def recognize_plugin_pattern(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the user's input to determine the format pattern of plugin/component definitions.
    
    This node runs BEFORE extract_system_elements to intelligently detect:
    - Format type (JSON, Markdown, Plain Text, etc.)
    - Structure pattern (array, nested object, list, etc.)
    - Language used (English, Persian, Mixed)
    - Best extraction strategy
    
    The detected pattern is passed to extract_system_elements to guide its extraction logic.
    
    Args:
        state: Current AppState containing messages
        
    Returns:
        Updated state with:
        - plugin_definition_pattern: Dict containing pattern analysis
        - pattern_confidence: Float indicating confidence in pattern detection
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
        _logger.info("No initial prompt found, skipping pattern recognition")
        return {
            "plugin_definition_pattern": {
                "format_type": "NONE",
                "has_plugins": False,
                "confidence": 1.0,
                "extraction_strategy": "none"
            },
            "pattern_confidence": 1.0
        }
    
    _logger.info("Starting plugin pattern recognition", 
                extra={
                    "prompt_length": len(initial_prompt),
                    "prompt_preview": initial_prompt[:100]
                })
    
    try:
        # Use LLM to analyze the pattern
        pattern_analysis = _analyze_pattern_with_llm(initial_prompt)
        
        if pattern_analysis:
            confidence = pattern_analysis.get("confidence", 0.5)
            format_type = pattern_analysis.get("format_type", "UNKNOWN")
            has_plugins = pattern_analysis.get("has_plugins", False)
            
            _logger.info("Pattern recognition completed", extra={
                "format_type": format_type,
                "confidence": confidence,
                "has_plugins": has_plugins,
                "language": pattern_analysis.get("language", "unknown"),
                "extraction_strategy": pattern_analysis.get("extraction_strategy", "unknown")
            })
            
            # Log detailed characteristics
            if pattern_analysis.get("characteristics"):
                _logger.debug("Pattern characteristics", 
                            extra={"characteristics": pattern_analysis["characteristics"]})
            
            return {
                "plugin_definition_pattern": pattern_analysis,
                "pattern_confidence": confidence
            }
        else:
            _logger.warning("Pattern analysis returned no results")
            return _get_fallback_pattern()
            
    except Exception as e:
        _logger.error("Error in pattern recognition, using fallback", 
                     extra={"error": str(e)}, exc_info=True)
        return _get_fallback_pattern()


def _analyze_pattern_with_llm(prompt: str) -> Optional[Dict[str, Any]]:
    """
    Use an LLM agent to analyze the pattern of plugin definitions.
    
    Args:
        prompt: The user's initial prompt text
        
    Returns:
        Dictionary containing pattern analysis or None if failed
    """
    try:
        # Render prompts using PromptManager
        system_prompt = pm.render("pattern_recognition_system")
        user_prompt = pm.render("pattern_recognition_user", prompt_text=prompt)
        
        # Call LLM with low temperature for consistent analysis
        llm = get_chat_model(temperature=0.1)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        _logger.debug("Calling LLM for pattern analysis")
        
        with PerformanceLogger(_logger, "pattern_recognition_llm_call"):
            response = invoke_llm_with_retry(llm, messages)
        
        response_text = response.content
        _logger.debug("LLM response received", 
                     extra={"response_length": len(response_text)})
        
        # Extract and parse JSON response
        json_str = _extract_json(response_text)
        pattern_data = json.loads(json_str)
        
        # Validate required fields
        required_fields = ["format_type", "has_plugins", "confidence", "extraction_strategy"]
        if not all(field in pattern_data for field in required_fields):
            _logger.warning("Pattern response missing required fields", 
                          extra={"present_fields": list(pattern_data.keys())})
            return None
        
        # Validate format_type is one of expected values
        valid_formats = [
            "JSON_ARRAY", "JSON_OBJECT", "JSON_INLINE", 
            "MARKDOWN_LIST", "MARKDOWN_TABLE", 
            "PLAIN_TEXT", "MIXED", "NONE"
        ]
        if pattern_data["format_type"] not in valid_formats:
            _logger.warning("Invalid format_type", 
                          extra={"format_type": pattern_data["format_type"]})
            pattern_data["format_type"] = "MIXED"  # Default to MIXED
        
        # Ensure confidence is in valid range
        confidence = float(pattern_data.get("confidence", 0.5))
        pattern_data["confidence"] = max(0.0, min(1.0, confidence))
        
        return pattern_data
        
    except json.JSONDecodeError as e:
        _logger.error("Failed to parse pattern analysis JSON", 
                     extra={"error": str(e), "response_preview": response_text[:200]})
        return None
    except Exception as e:
        _logger.error("Unexpected error in LLM pattern analysis", 
                     extra={"error": str(e)}, exc_info=True)
        return None


def _get_fallback_pattern() -> Dict[str, Any]:
    """
    Return a safe fallback pattern when analysis fails.
    Assumes mixed format with low confidence, requiring hybrid extraction.
    """
    return {
        "plugin_definition_pattern": {
            "format_type": "MIXED",
            "structure_pattern": "unknown",
            "language": "mixed",
            "extraction_strategy": "hybrid",
            "confidence": 0.3,
            "characteristics": ["pattern detection failed", "using fallback strategy"],
            "has_plugins": True,  # Assume plugins might exist
            "estimated_plugin_count": "unknown",
            "recommended_parser": {
                "primary_method": "llm_semantic",
                "fallback_method": "regex_pattern",
                "special_handling": "try multiple extraction methods"
            },
            "extraction_hints": {
                "json_path": None,
                "regex_pattern": None,
                "key_markers": [],
                "delimiter": None
            }
        },
        "pattern_confidence": 0.3
    }


def get_extraction_guidance(pattern: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert pattern analysis into actionable extraction guidance.
    This is a helper function that extract_system_elements can use.
    
    Args:
        pattern: The plugin_definition_pattern from state
        
    Returns:
        Dictionary with extraction guidance including methods and priorities
    """
    format_type = pattern.get("format_type", "MIXED")
    extraction_strategy = pattern.get("extraction_strategy", "hybrid")
    
    # Define extraction method priorities based on format
    method_priorities = {
        "JSON_ARRAY": ["json_parse", "json_extract_then_parse", "llm_semantic"],
        "JSON_OBJECT": ["json_parse", "json_extract_then_parse", "llm_semantic"],
        "JSON_INLINE": ["json_extract_then_parse", "json_parse", "llm_semantic"],
        "MARKDOWN_LIST": ["regex_pattern", "llm_semantic"],
        "MARKDOWN_TABLE": ["regex_pattern", "llm_semantic"],
        "PLAIN_TEXT": ["llm_semantic", "regex_pattern"],
        "MIXED": ["hybrid", "llm_semantic", "json_extract_then_parse", "regex_pattern"],
        "NONE": []
    }
    
    methods = method_priorities.get(format_type, ["hybrid", "llm_semantic"])
    
    return {
        "primary_method": methods[0] if methods else "llm_semantic",
        "fallback_methods": methods[1:] if len(methods) > 1 else ["llm_semantic"],
        "extraction_hints": pattern.get("extraction_hints", {}),
        "recommended_parser": pattern.get("recommended_parser", {}),
        "confidence": pattern.get("confidence", 0.5),
        "language": pattern.get("language", "mixed"),
        "structure_pattern": pattern.get("structure_pattern", "unknown")
    }
