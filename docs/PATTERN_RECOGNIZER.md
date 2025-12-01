# Pattern Recognizer Agent - Documentation

## Overview

The **Pattern Recognizer** is a specialized LLM-based agent that analyzes user input to detect **HOW** plugins/components are defined, **BEFORE** attempting to extract them.

## Position in the Workflow

```
User Input
    ↓
[extract_goal]
    ↓
[recognize_plugin_pattern] ← NEW NODE (detects format)
    ↓
[extract_system_elements] ← Uses pattern info
    ↓
[resolve_dependencies]
    ↓
[split_task]
    ↓
[plan_tot]
    ↓
...
```

## What It Detects

### 1. Format Types
- **JSON_ARRAY**: `[{...}, {...}]`
- **JSON_OBJECT**: `{"plugins": [...]}`
- **JSON_INLINE**: Text with embedded JSON
- **MARKDOWN_LIST**: Bullet/numbered lists
- **MARKDOWN_TABLE**: Structured tables
- **PLAIN_TEXT**: Natural language descriptions
- **MIXED**: Combination of formats
- **NONE**: No plugins detected

### 2. Structure Patterns
- `flat_array`: Direct array
- `nested_plugins_key`: Under "plugins" key
- `bullet_list`: Markdown bullets
- `inline_description`: Narrative format

### 3. Language Detection
- English
- Persian/Farsi
- Mixed
- Multilingual

### 4. Extraction Strategy
- `json_parse`: Direct JSON parsing
- `json_extract_then_parse`: Extract then parse
- `regex_pattern`: Regex extraction
- `llm_semantic`: LLM understanding
- `hybrid`: Multiple methods

## State Fields Added

```python
class AppState(TypedDict):
    # ... existing fields ...
    
    # Pattern recognition (NEW)
    plugin_definition_pattern: Dict[str, Any]  # Pattern analysis
    pattern_confidence: float  # Confidence score
```

## Pattern Analysis Output

```json
{
  "format_type": "JSON_ARRAY",
  "structure_pattern": "flat_array",
  "language": "english",
  "extraction_strategy": "json_parse",
  "confidence": 0.95,
  "characteristics": [
    "pure JSON array",
    "plugin objects with name and goal"
  ],
  "has_plugins": true,
  "estimated_plugin_count": 10,
  "recommended_parser": {
    "primary_method": "json.loads",
    "fallback_method": "json_extract_then_parse",
    "special_handling": "none"
  },
  "extraction_hints": {
    "json_path": "$[*]",
    "regex_pattern": null,
    "key_markers": ["name", "goal"],
    "delimiter": null
  }
}
```

## How Extract System Elements Uses Pattern Info

### Before Pattern Recognizer
```python
# Old approach: Try all methods blindly
try:
    llm_plugins = extract_with_llm(prompt)
    if not llm_plugins:
        json_plugins = extract_json(prompt)
    if not json_plugins:
        regex_plugins = extract_regex(prompt)
```

### After Pattern Recognizer
```python
# New approach: Guided by pattern analysis
pattern = state["plugin_definition_pattern"]

if pattern["format_type"] == "JSON_ARRAY":
    # We know it's JSON, parse directly
    plugins = json.loads(prompt)
elif pattern["format_type"] == "PLAIN_TEXT":
    # We know it's plain text, use LLM
    plugins = extract_with_llm(prompt, pattern["language"])
elif pattern["format_type"] == "MARKDOWN_LIST":
    # We know it's markdown, use regex
    plugins = extract_with_regex(prompt)
```

## Benefits

1. **Efficiency**: No wasted attempts on wrong methods
2. **Accuracy**: Right tool for the right format
3. **Context**: Pattern hints improve LLM extraction
4. **Language Support**: Persian/Farsi detection
5. **Debugging**: Clear visibility into why extraction failed

## Example Use Cases

### Case 1: Large JSON Array
**Input**: Your 50+ plugin JSON array

**Pattern Detection**:
```json
{
  "format_type": "JSON_ARRAY",
  "confidence": 0.95,
  "extraction_strategy": "json_parse",
  "has_plugins": true,
  "estimated_plugin_count": 50
}
```

**Result**: Direct JSON parsing, no LLM call needed, instant extraction

### Case 2: Persian Natural Language
**Input**: `میخوام از اطلس بخونم با استفاده از کلید`

**Pattern Detection**:
```json
{
  "format_type": "PLAIN_TEXT",
  "language": "persian",
  "confidence": 0.7,
  "extraction_strategy": "llm_semantic",
  "has_plugins": false
}
```

**Result**: Skips plugin extraction, proceeds to abstract planning

### Case 3: Mixed Format
**Input**: 
```
I have these plugins:
[{"name": "A"}, {"name": "B"}]

And also these components:
- ComponentX
- ComponentY
```

**Pattern Detection**:
```json
{
  "format_type": "MIXED",
  "confidence": 0.6,
  "extraction_strategy": "hybrid",
  "has_plugins": true
}
```

**Result**: Uses hybrid approach (JSON + regex)

## Testing

Run the test suite:
```bash
python tests/unit/test_pattern_recognizer.py
```

## Implementation Files

- `src/agents/pattern_recognizer.py` - Main agent logic
- `src/prompts/pattern_recognition_system.jinja` - System prompt
- `src/prompts/pattern_recognition_user.jinja` - User prompt
- `src/agents/plugin_analyzer.py` - Updated to use patterns
- `src/agents/graph.py` - Updated workflow
- `src/agents/state.py` - New state fields

## Future Enhancements

1. **Pattern Library**: Cache common patterns for faster detection
2. **Multi-Stage Extraction**: For very complex mixed formats
3. **Pattern Validation**: Verify extracted data matches pattern
4. **Auto-Correction**: Fix common format issues (missing commas, etc.)
5. **Pattern Metrics**: Track which patterns are most common

## Debug Tips

Enable detailed logging:
```python
import logging
logging.getLogger("src.agents.pattern_recognizer").setLevel(logging.DEBUG)
```

Check pattern in state:
```python
pattern = state.get("plugin_definition_pattern", {})
print(f"Detected: {pattern.get('format_type')}")
print(f"Confidence: {pattern.get('confidence')}")
```
