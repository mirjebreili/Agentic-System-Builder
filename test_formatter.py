"""
Simple test script to verify the formatter node works correctly.
Run this after setting up your environment variables.
"""

import json
from agents.formatter import format_plan_order, _fallback_format

# Sample plan for testing
sample_plan = {
    "goal": "Sum values of keys starting with price_*",
    "nodes": [
        {
            "id": "HttpBasedAtlasReadByKey",
            "tool": "HttpBasedAtlasReadByKey",
            "prompt": "read keys: price_*"
        },
        {
            "id": "membasedAtlasKeyStreamAggregator",
            "tool": "membasedAtlasKeyStreamAggregator",
            "prompt": "sum numeric values"
        }
    ],
    "edges": [
        {
            "from": "HttpBasedAtlasReadByKey",
            "to": "membasedAtlasKeyStreamAggregator",
            "if": None
        }
    ],
    "confidence": 0.95
}

# Test state
test_state = {
    "plan": sample_plan,
    "messages": [],
    "scratch": {}
}

print("=" * 80)
print("Testing Formatter Node")
print("=" * 80)

# Test fallback formatter first
print("\n1. Testing fallback formatter (no LLM):")
print("-" * 80)
fallback_output = _fallback_format(sample_plan)
print(fallback_output)

# Test full LLM-based formatter
print("\n\n2. Testing LLM-based formatter:")
print("-" * 80)
print("Invoking the formatter node...")

try:
    result = format_plan_order(test_state)
    formatted_text = result.get("scratch", {}).get("formatted_plan_order", "")
    
    if formatted_text:
        print("\nFormatted Output:")
        print("-" * 80)
        print(formatted_text)
        print("\n✓ Formatter executed successfully!")
    else:
        print("\n✗ No formatted output was generated")
        
except Exception as e:
    print(f"\n✗ Error during formatting: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Test Complete")
print("=" * 80)
