#!/usr/bin/env python3
"""Test script for plugin sequencing system."""

import sys
import os
sys.path.insert(0, 'src')

from agents.plugin_analyzer_simple import analyze_plugins_simple, parse_plugin_descriptions

def test_plugin_sequencing():
    """Test the plugin sequencing with provided examples."""
    
    # Sample plugin descriptions from the user
    plugin_texts = [
        """partDeltaPluginHttpBasedAtlasReadByKey
        
Reads data from Atlas service using HTTP requests. Can search by keys/tokens and returns structured data with keys and values. Good for reading data from database/service.

Parameters:
- atlasRequest: Search parameters including searchToken, include fields
- args: project, namespace, headers, throwError
- configs: baseUrl, responseType

Output: JSON data with result array containing objects with id and keys.""",

        """membasedAtlasKeyStreamAggregator

Aggregates/sums numeric values from keys in Atlas stream data. Can sum by key prefix (name parameter) or by key index (index parameter). Good for calculating totals, sums, aggregations.

Parameters:
- atlasStream: Input stream of Atlas data
- args: index (for position-based) or name (for prefix-based filtering)
- config: inputType should be "stream"

Output: Numeric sum of extracted values from matching keys."""
    ]
    
    # Parse plugins
    plugins = parse_plugin_descriptions(plugin_texts)
    
    # Test questions
    test_questions = [
        "مجموع مقادیر انتهایی کلیدهایی که با price_ شروع میشن رو بده",
        "مجموع هزینه‌ی انجام شده در پروژه‌ی target_project و فضای نام target_namespace", 
        "مجموع هزینه های انجام شده"
    ]
    
    for question in test_questions:
        print(f"\n🔍 Question: {question}")
        
        # Create state
        state = {
            "question": question,
            "plugins": plugins,
            "goal": "plugin sequencing task"
        }
        
        # Analyze plugins
        result_state = analyze_plugins_simple(state)
        
        if "error" in result_state:
            print(f"❌ Error: {result_state['error']}")
        else:
            sequence = result_state.get("plugin_sequence", "No sequence determined")
            print(f"✅ Plugin Sequence: {sequence}")

if __name__ == "__main__":
    test_plugin_sequencing()