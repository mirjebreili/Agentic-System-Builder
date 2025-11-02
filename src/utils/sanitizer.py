"""
Input Sanitization and Security Validation.

This module provides utilities to validate and sanitize user inputs
to prevent security issues like prompt injection.
"""

from typing import Any, Dict, List, Optional
import re
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Suspicious patterns that might indicate injection attempts
SUSPICIOUS_PATTERNS = [
    r"ignore\s+(previous|all)\s+instructions",
    r"system\s*:\s*you\s+are",
    r"new\s+instructions",
    r"disregard\s+(previous|all)",
    r"<\s*script",
    r"javascript\s*:",
    r"eval\s*\(",
    r"__import__",
    r"exec\s*\(",
    r"subprocess",
]


def contains_injection_attempt(text: str) -> bool:
    """
    Check if text contains potential injection attempts.
    
    Args:
        text: Input text to check
        
    Returns:
        True if suspicious patterns detected
    """
    text_lower = text.lower()
    
    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            logger.warning("suspicious_pattern_detected", pattern=pattern)
            return True
    
    return False


def sanitize_input(text: str, max_length: int = 10000) -> str:
    """
    Sanitize user input by removing potentially harmful content.
    
    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
    """
    # Truncate to max length
    if len(text) > max_length:
        logger.warning("input_truncated", original_length=len(text), max_length=max_length)
        text = text[:max_length]
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def validate_json_structure(data: Any, max_depth: int = 10, current_depth: int = 0) -> bool:
    """
    Validate JSON structure to prevent deeply nested attacks.
    
    Args:
        data: Data structure to validate
        max_depth: Maximum nesting depth
        current_depth: Current recursion depth
        
    Returns:
        True if valid, False otherwise
    """
    if current_depth > max_depth:
        logger.warning("json_max_depth_exceeded", depth=current_depth)
        return False
    
    if isinstance(data, dict):
        if len(data) > 1000:  # Too many keys
            logger.warning("json_too_many_keys", count=len(data))
            return False
        
        for value in data.values():
            if not validate_json_structure(value, max_depth, current_depth + 1):
                return False
    
    elif isinstance(data, list):
        if len(data) > 1000:  # Too many items
            logger.warning("json_too_many_items", count=len(data))
            return False
        
        for item in data:
            if not validate_json_structure(item, max_depth, current_depth + 1):
                return False
    
    return True


def sanitize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize user inputs in state.
    
    Args:
        state: Application state
        
    Returns:
        Updated state with validation flags
    """
    messages = state.get("messages", [])
    
    # Check the last user message for suspicious content
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            content = msg.content if hasattr(msg, "content") else ""
            
            if contains_injection_attempt(content):
                logger.error("injection_attempt_detected", content_preview=content[:100])
                from langchain_core.messages import AIMessage
                return {
                    "messages": [AIMessage(content="⚠️ Invalid input detected. Please rephrase your request.")],
                    "error": "Input validation failed",
                    "input_validated": False
                }
            
            # Validate JSON if present
            if "{" in content and "}" in content:
                try:
                    import json
                    # Try to find and parse JSON
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        json_data = json.loads(json_match.group())
                        if not validate_json_structure(json_data):
                            logger.error("invalid_json_structure")
                            from langchain_core.messages import AIMessage
                            return {
                                "messages": [AIMessage(content="⚠️ Invalid data structure. Please simplify your input.")],
                                "error": "Invalid JSON structure",
                                "input_validated": False
                            }
                except json.JSONDecodeError:
                    # Not valid JSON, that's okay
                    pass
            
            break
    
    return {"input_validated": True}


def validate_plugin_input(plugin: Dict[str, Any]) -> List[str]:
    """
    Validate plugin definition from user.
    
    Args:
        plugin: Plugin dictionary
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Check required fields
    if "name" not in plugin:
        errors.append("Plugin missing 'name' field")
    else:
        name = plugin["name"]
        # Validate name format
        if not isinstance(name, str) or not name.strip():
            errors.append("Plugin name must be non-empty string")
        elif not re.match(r'^[a-zA-Z0-9_-]+$', name):
            errors.append(f"Plugin name '{name}' contains invalid characters")
    
    # Validate optional fields
    if "dependencies" in plugin:
        deps = plugin["dependencies"]
        if not isinstance(deps, (list, str)):
            errors.append("Plugin dependencies must be a list or string")
    
    if "tools" in plugin:
        tools = plugin["tools"]
        if not isinstance(tools, list):
            errors.append("Plugin tools must be a list")
    
    return errors


def rate_limit_check(state: Dict[str, Any], max_requests: int = 100) -> bool:
    """
    Check if rate limit is exceeded.
    
    Args:
        state: Application state
        max_requests: Maximum requests allowed
        
    Returns:
        True if within limit, False if exceeded
    """
    # This is a simple implementation
    # In production, use Redis or similar for distributed rate limiting
    
    request_count = state.get("_request_count", 0)
    
    if request_count >= max_requests:
        logger.warning("rate_limit_exceeded", count=request_count, limit=max_requests)
        return False
    
    return True
