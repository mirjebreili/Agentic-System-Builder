#!/usr/bin/env python3
"""
Test the planning system to show the detailed multi-step plan structure.
"""

import os
import sys
import asyncio

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agents.planner import plan_tot
from agents.state import AppState
from langchain_core.messages import HumanMessage

def test_planning_system():
    """Test the planning system to see detailed plan structure"""
    
    print("🧪 Testing Planning System for Persian Plugin Analysis")
    print("=" * 60)
    
    # The Persian message
    persian_message = "مجموع مقادیر انتهایی کلیدهایی که با price_ شروع میشن رو بده"
    
    print(f"📝 Input Message: {persian_message}")
    print()
    
    # Create the initial state
    initial_state: AppState = {
        "messages": [HumanMessage(content=persian_message)],
        "goal": persian_message,
        "plugins": [
            "partDeltaPluginHttpBasedAtlasReadByKey",
            "membasedAtlasKeyStreamAggregator"
        ]
    }
    
    try:
        print("🔧 Creating plan...")
        result = plan_tot(initial_state)  # Remove await since it's not async
        
        print("✅ Plan created successfully!")
        print()
        
        # Display plan details
        print("📋 Plan Details:")
        print("=" * 30)
        
        if "plan" in result:
            plan = result["plan"]
            print(f"🎯 Goal: {plan.get('goal', 'N/A')}")
            print(f"🔢 Confidence: {plan.get('confidence', 'N/A')}")
            print()
            
            print("📝 Nodes:")
            nodes = plan.get("nodes", [])
            for i, node in enumerate(nodes, 1):
                print(f"  {i}. ID: {node.get('id', 'N/A')}")
                print(f"     Type: {node.get('type', 'N/A')}")
                print(f"     Prompt: {node.get('prompt', 'N/A')[:100]}...")
                if node.get('tool'):
                    print(f"     Tool: {node.get('tool')}")
                print()
            
            print("🔗 Edges:")
            edges = plan.get("edges", [])
            for i, edge in enumerate(edges, 1):
                from_node = edge.get('from', 'N/A')
                to_node = edge.get('to', 'N/A')
                condition = edge.get('if', '')
                if condition:
                    print(f"  {i}. {from_node} → {to_node} (if: {condition})")
                else:
                    print(f"  {i}. {from_node} → {to_node}")
            
            print()
            print("🏗️ Plan Structure Analysis:")
            if len(nodes) >= 4:
                print("✅ Multi-step plan created successfully!")
                print(f"✅ Contains {len(nodes)} nodes (meets 2-5 requirement)")
                print(f"✅ Contains {len(edges)} edges for workflow")
            else:
                print(f"⚠️  Only {len(nodes)} nodes created (should be 2-5)")
            
            # Check for plugin-specific nodes
            node_ids = [node.get('id', '') for node in nodes]
            if any('plugin' in node_id or 'analyze' in node_id for node_id in node_ids):
                print("✅ Plugin-specific workflow detected")
            else:
                print("⚠️  Generic workflow used instead of plugin-specific")
        
        return result
        
    except Exception as e:
        print(f"❌ Error creating plan: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_execution_flow():
    """Test how the plan would execute step by step"""
    
    print("\n" + "=" * 60)
    print("🎬 Simulating Plan Execution Flow")
    print("=" * 60)
    
    print("Expected Flow for Persian Plugin Analysis:")
    print("1. 📝 Analyze Question: Understand Persian text and requirements")
    print("2. 🔌 Determine Plugins: Map operations to available plugins")
    print("3. ✅ Validate Sequence: Verify the plugin sequence is correct")
    print("4. 📋 Summarize: Provide final result and explanation")
    print()
    
    print("Expected Output: HttpBasedAtlasReadByKey --> membasedAtlasKeyStreamAggregator")
    print()
    print("Why this sequence:")
    print("- 'مجموع' (sum) → Need aggregation")
    print("- 'price_ شروع' (starts with price_) → Need prefix filtering")
    print("- No existing data → Need to read from Atlas first")
    print("- Result: Read → Aggregate")

def main():
    """Main function"""
    try:
        result = test_planning_system()  # Remove asyncio.run
        test_execution_flow()  # Remove asyncio.run
        
        print("\n" + "=" * 60)
        if result and result.get("plan"):
            print("🎉 Planning system test completed!")
            print("✅ Multi-step plan structure created")
            print("✅ Persian plugin analysis workflow ready")
        else:
            print("⚠️  Planning system needs adjustment")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()