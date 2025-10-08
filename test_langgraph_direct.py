#!/usr/bin/env python3
"""
Test the LangGraph system directly with the Persian message.
This bypasses the CLI and tests the graph directly.
"""

import os
import sys
import asyncio

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agents.graph import _make_graph
from agents.state import AppState
from langchain_core.messages import HumanMessage

async def test_persian_message():
    """Test the system with the Persian message"""
    
    print("🚀 Testing LangGraph System with Persian Message")
    print("=" * 60)
    
    # The Persian message from the user
    persian_message = "مجموع مقادیر انتهایی کلیدهایی که با price_ شروع میشن رو بده"
    
    print(f"📝 Input Message: {persian_message}")
    print()
    
    # Create the initial state
    initial_state: AppState = {
        "messages": [HumanMessage(content=persian_message)],
        "question": persian_message,
        "plugins": [
            "partDeltaPluginHttpBasedAtlasReadByKey",
            "membasedAtlasKeyStreamAggregator"
        ]
    }
    
    # Create the graph
    try:
        print("🔧 Creating LangGraph...")
        graph = _make_graph()
        print("✅ Graph created successfully")
        print()
        
        # Run the graph
        print("▶️  Running the graph...")
        result = await graph.ainvoke(initial_state)
        
        print("✅ Graph execution completed!")
        print()
        
        # Display results
        print("📊 Results:")
        print("=" * 30)
        
        # Check for plugin sequence
        if "plugin_sequence" in result:
            print(f"🔌 Plugin Sequence: {result['plugin_sequence']}")
        
        # Check for plan
        if "plan" in result:
            plan = result['plan']
            print(f"📋 Plan Type: {plan.get('type', 'N/A')}")
            print(f"🎯 Sequence: {plan.get('sequence', 'N/A')}")
            print(f"📝 Analysis: {plan.get('analysis', 'N/A')}")
        
        # Check final messages
        if "messages" in result and result["messages"]:
            last_message = result["messages"][-1]
            print(f"💬 Last Message: {last_message.content[:200]}...")
        
        # Check for errors
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        
        print()
        print("🎯 Expected Sequence: HttpBasedAtlasReadByKey --> membasedAtlasKeyStreamAggregator")
        
        # Validate result
        expected = "HttpBasedAtlasReadByKey --> membasedAtlasKeyStreamAggregator"
        actual = result.get('plugin_sequence', '')
        
        if actual == expected:
            print("✅ SUCCESS: Got exactly the expected sequence!")
        else:
            print(f"❌ MISMATCH: Expected {expected}, got {actual}")
            
        return result
        
    except Exception as e:
        print(f"❌ Error running graph: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function"""
    print("🚀 LangGraph Direct Test")
    print("=" * 60)
    
    try:
        result = asyncio.run(test_persian_message())
        
        print("\n" + "=" * 60)
        if result:
            print("🎉 Test completed successfully!")
        else:
            print("⚠️  Test failed")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()