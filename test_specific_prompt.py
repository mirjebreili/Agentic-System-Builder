#!/usr/bin/env python3
"""
Test script to validate the spec        if actual_sequence == expected_sequence:
            print("✅ SUCCESS: Got exactly the expected sequence!")
            print(f"   Expected: {expected_sequence}")
            print(f"   Actual:   {actual_sequence}")
            success_flag = True
        else:
            print("❌ MISMATCH: Sequence doesn't match expected result")
            print(f"   Expected: {expected_sequence}")
            print(f"   Actual:   {actual_sequence}")
            success_flag = False
            
        print()
        print("🧠 Detailed Analysis:")
        print(f"  Plan Type: {result.get('plan', {}).get('type', 'N/A')}")
        print(f"  Analysis: {result.get('plan', {}).get('analysis', 'N/A')}")
        print(f"  Plugins Used: {result.get('plan', {}).get('plugins_used', 'N/A')}")
        
        return success_flagompt:
"مجموع مقادیر انتهایی کلیدهایی که با price_ شروع میشن رو بده"

This should produce exactly: HttpBasedAtlasReadByKey --> membasedAtlasKeyStreamAggregator
"""

import os
import sys

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agents.plugin_analyzer_simple import analyze_plugins_simple

def test_specific_prompt():
    """Test the specific Persian prompt with our plugin analyzer"""
    
    # The specific prompt to test
    test_prompt = "مجموع مقادیر انتهایی کلیدهایی که با price_ شروع میشن رو بده"
    
    # Available plugins with their descriptions
    plugins = {
        "partDeltaPluginHttpBasedAtlasReadByKey": {
            "name": "HttpBasedAtlasReadByKey",
            "description": "Plugin for reading data from Atlas service by key. Handles HTTP requests to Atlas readByKey API. Used for data retrieval operations with search tokens.",
            "capabilities": ["data_reading", "http_requests", "atlas_service", "key_search"]
        },
        "membasedAtlasKeyStreamAggregator": {
            "name": "membasedAtlasKeyStreamAggregator", 
            "description": "Plugin for aggregating numeric values from Atlas keys in memory. Processes streams of data and sums numeric suffixes from keys based on prefix or index.",
            "capabilities": ["data_aggregation", "stream_processing", "numeric_calculation", "key_filtering"]
        }
    }
    
    print("🧪 Testing Specific Persian Prompt")
    print("=" * 50)
    print(f"📝 Prompt: {test_prompt}")
    print()
    
    # Create state object for the analyzer
    state = {
        "question": test_prompt,
        "plugins": list(plugins.keys())
    }
    
    # Analyze the prompt
    try:
        result = analyze_plugins_simple(state)
        
        print("🔍 Analysis Results:")
        print(f"  Plugin Sequence: {result.get('plugin_sequence', 'N/A')}")
        print(f"  Plan: {result.get('plan', {})}")
        print(f"  Error: {result.get('error', 'None')}")
        print()
        
        # Check if we get the expected result
        expected_sequence = "HttpBasedAtlasReadByKey --> membasedAtlasKeyStreamAggregator"
        actual_sequence = result.get('plugin_sequence', '')
        
        if actual_sequence == expected_sequence:
            print("✅ SUCCESS: Got exactly the expected sequence!")
            print(f"   Expected: {expected_sequence}")
            print(f"   Actual:   {actual_sequence}")
            success_flag = True
        else:
            print("❌ MISMATCH: Sequence doesn't match expected result")
            print(f"   Expected: {expected_sequence}")
            print(f"   Actual:   {actual_sequence}")
            success_flag = False
            
        print()
        print("🧠 Detailed Analysis:")
        print(f"  Plan Type: {result.get('plan', {}).get('type', 'N/A')}")
        print(f"  Analysis: {result.get('plan', {}).get('analysis', 'N/A')}")
        print(f"  Plugins Used: {result.get('plan', {}).get('plugins_used', 'N/A')}")
        
        return success_flag
        
    except Exception as e:
        print(f"❌ ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Plugin Analyzer Test - Specific Persian Prompt")
    print("=" * 60)
    
    success = test_specific_prompt()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 TEST PASSED: The system produces the exact expected result!")
    else:
        print("⚠️  TEST FAILED: The system output doesn't match expectations")
    
    return success

if __name__ == "__main__":
    main()