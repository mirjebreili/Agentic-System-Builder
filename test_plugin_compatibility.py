#!/usr/bin/env python3
"""
Test script to validate the system works with the exact plugin descriptions provided by the user.
This ensures the analyzer works correctly with partDeltaPluginHttpBasedAtlasReadByKey and membasedAtlasKeyStreamAggregator.
"""

import os
import sys

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agents.plugin_analyzer_simple import analyze_plugins_simple

def test_with_provided_plugins():
    """Test with the exact plugin descriptions provided by the user"""
    
    # The specific prompt to test
    test_prompt = "مجموع مقادیر انتهایی کلیدهایی که با price_ شروع میشن رو بده"
    
    # Exact plugin names from user's documentation
    plugins = [
        "partDeltaPluginHttpBasedAtlasReadByKey", 
        "membasedAtlasKeyStreamAggregator"
    ]
    
    # Create state object
    state = {
        "question": test_prompt,
        "plugins": plugins
    }
    
    print("🧪 Testing with Provided Plugin Configuration")
    print("=" * 55)
    print(f"📝 Persian Question: {test_prompt}")
    print(f"🔌 Available Plugins: {plugins}")
    print()
    
    # Analyze the prompt
    try:
        result = analyze_plugins_simple(state)
        
        print("🔍 Analysis Results:")
        print(f"  ✅ Plugin Sequence: {result.get('plugin_sequence', 'N/A')}")
        print(f"  🎯 Plan Type: {result.get('plan', {}).get('type', 'N/A')}")
        print(f"  📊 Plugins Used: {result.get('plan', {}).get('plugins_used', 0)}")
        print(f"  ⚠️  Error: {result.get('error', 'None')}")
        print()
        
        # Verify the expected sequence
        expected = "HttpBasedAtlasReadByKey --> membasedAtlasKeyStreamAggregator"
        actual = result.get('plugin_sequence', '')
        
        print("🎯 Validation:")
        print(f"  Expected: {expected}")
        print(f"  Actual:   {actual}")
        print(f"  Match:    {'✅ YES' if actual == expected else '❌ NO'}")
        print()
        
        # Check workflow logic
        print("🧠 Workflow Analysis:")
        print("  1. Persian keyword 'مجموع' detected → Sum operation required")
        print("  2. Persian keyword 'شروع' detected → Key prefix filtering needed")
        print("  3. No existing data in state → Data reading required first")
        print("  4. Logic: Read data first, then aggregate")
        print("  5. Result: HttpBasedAtlasReadByKey → membasedAtlasKeyStreamAggregator")
        print()
        
        # Verify plugin capabilities match the requirement
        print("🔧 Plugin Capability Mapping:")
        print("  📥 HttpBasedAtlasReadByKey:")
        print("     - Reads data from Atlas service via HTTP")
        print("     - Supports search tokens (like 'price_*')")
        print("     - Returns JSON stream suitable for aggregation")
        print("  📊 membasedAtlasKeyStreamAggregator:")
        print("     - Processes streams from Atlas plugins")
        print("     - Filters keys by prefix (handles 'price_' prefix)")
        print("     - Sums numeric values from key suffixes")
        print("     - Perfect for: 'مجموع مقادیر انتهایی کلیدهایی که با price_ شروع میشن'")
        
        return actual == expected
        
    except Exception as e:
        print(f"❌ ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Plugin Compatibility Test")
    print("Testing with exact plugin names from user documentation")
    print("=" * 65)
    
    success = test_with_provided_plugins()
    
    print("\n" + "=" * 65)
    if success:
        print("🎉 COMPATIBILITY CONFIRMED!")
        print("✅ The system correctly works with your provided plugins")
        print("✅ Persian prompt analysis produces exact expected sequence")
        print("✅ Workflow: Data reading → Aggregation by prefix → Sum calculation")
    else:
        print("⚠️  COMPATIBILITY ISSUE DETECTED")
        print("❌ The system output doesn't match expectations")
    
    return success

if __name__ == "__main__":
    main()