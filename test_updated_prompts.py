#!/usr/bin/env python3
"""
Comprehensive test to validate the updated prompts for Persian question analysis.
Focus: "مجموع مقادیر انتهایی کلیدهایی که با price_ شروع میشن رو بده"
Expected: HttpBasedAtlasReadByKey --> membasedAtlasKeyStreamAggregator
"""

import os
import sys

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agents.plugin_analyzer_simple import analyze_plugins_simple

def test_updated_prompts():
    """Test the updated prompts with various Persian questions"""
    
    print("🧪 Testing Updated Prompts for Persian Question Analysis")
    print("=" * 60)
    
    test_cases = [
        {
            "question": "مجموع مقادیر انتهایی کلیدهایی که با price_ شروع میشن رو بده",
            "description": "Sum of final values for keys starting with price_",
            "expected": "HttpBasedAtlasReadByKey --> membasedAtlasKeyStreamAggregator"
        },
        {
            "question": "مجموع مقادیر کلیدهایی که با cost_ شروع میشوند",
            "description": "Sum of values for keys starting with cost_",
            "expected": "HttpBasedAtlasReadByKey --> membasedAtlasKeyStreamAggregator"
        },
        {
            "question": "مجموع هزینه‌های پروژه target_project",
            "description": "Sum of project costs for target_project",
            "expected": "HttpBasedAtlasReadByKey --> membasedAtlasKeyStreamAggregator"
        },
        {
            "question": "مجموع expense_100 و expense_200",
            "description": "Sum of specific expense values",
            "expected": "HttpBasedAtlasReadByKey --> membasedAtlasKeyStreamAggregator"
        }
    ]
    
    # Available plugins matching your exact setup
    plugins = [
        "partDeltaPluginHttpBasedAtlasReadByKey",
        "membasedAtlasKeyStreamAggregator"
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}:")
        print(f"   Persian: {test_case['question']}")
        print(f"   English: {test_case['description']}")
        print(f"   Expected: {test_case['expected']}")
        
        # Create state for analyzer
        state = {
            "question": test_case['question'],
            "plugins": plugins
        }
        
        # Analyze
        result = analyze_plugins_simple(state)
        actual = result.get('plugin_sequence', '')
        
        if actual == test_case['expected']:
            print(f"   ✅ PASS: {actual}")
        else:
            print(f"   ❌ FAIL: Expected {test_case['expected']}, got {actual}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Updated prompts correctly handle Persian questions")
        print("✅ Plugin sequencing works for price_ prefix aggregation")
    else:
        print("⚠️  Some tests failed - need prompt adjustments")
    
    return all_passed

def demonstrate_plugin_mapping():
    """Demonstrate how the plugin mapping works for the specific use case"""
    
    print("\n🔧 Plugin Mapping Demonstration")
    print("=" * 60)
    
    question = "مجموع مقادیر انتهایی کلیدهایی که با price_ شروع میشن رو بده"
    
    print(f"📝 Persian Question: {question}")
    print()
    
    print("🧠 Analysis Breakdown:")
    print("   1. 'مجموع' (sum) → Aggregation needed")
    print("   2. 'مقادیر انتهایی' (final values) → Extract numeric suffixes")
    print("   3. 'کلیدهایی که با price_ شروع' → Keys starting with price_")
    print("   4. 'رو بده' (give/return) → Output requirement")
    print()
    
    print("🔌 Plugin Workflow:")
    print("   Step 1: partDeltaPluginHttpBasedAtlasReadByKey")
    print("          - Search Atlas DB with searchToken: 'price_*'")
    print("          - Return JSON data with keys like ['price_100', 'price_50']")
    print("          - Configure stream output for aggregator")
    print()
    print("   Step 2: membasedAtlasKeyStreamAggregator")
    print("          - Process stream from Step 1")
    print("          - Filter keys by name parameter: 'price_'")
    print("          - Extract numeric suffixes: 100, 50, etc.")
    print("          - Return sum: 150 (if price_100 + price_50)")
    print()
    
    print("🎯 Result: HttpBasedAtlasReadByKey --> membasedAtlasKeyStreamAggregator")
    print()
    print("💡 This matches exactly what your plugins do:")
    print("   - HttpBasedAtlasReadByKey: Fetches data via Atlas readByKey API")
    print("   - membasedAtlasKeyStreamAggregator: Sums values by key prefix")

if __name__ == "__main__":
    print("🚀 Updated Prompts Validation Test")
    print("=" * 60)
    
    # Test updated prompts
    success = test_updated_prompts()
    
    # Demonstrate the specific use case
    demonstrate_plugin_mapping()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 VALIDATION COMPLETE: Updated prompts work perfectly!")
        print("✅ Your Persian question produces exact expected sequence")
        print("✅ Plugin documentation properly integrated")
        print("✅ Ready for production use")
    else:
        print("⚠️  Validation needs adjustment")
    
    print("=" * 60)