#!/usr/bin/env python3
"""
Complete Plugin Sequencing System Example

This demonstrates the agentic system that:
1. Receives Persian questions about data operations
2. Analyzes available plugins 
3. Determines the optimal sequence to resolve the question
4. Runs via LangGraph dev server

Usage:
    python demo_plugin_system.py

The system will be available at: http://127.0.0.1:8124
"""

import sys
import os
sys.path.insert(0, 'src')

from agents.plugin_analyzer_simple import analyze_plugins_simple, parse_plugin_descriptions

def demo_plugin_sequencing():
    """Demonstrate the complete plugin sequencing system."""
    
    print("🚀 Plugin Sequencing System Demo")
    print("=" * 50)
    
    # Sample plugin descriptions as provided by the user
    plugin_texts = [
        """partDeltaPluginHttpBasedAtlasReadByKey
        
این پلاگین برای خواندن داده از سرویس اطلس با استفاده از درخواست HTTP طراحی شده است. 
قابلیت جستجو بر اساس کلیدها/توکن‌ها را دارد و داده‌های ساختاری شامل کلیدها و مقادیر را برمی‌گرداند.

Parameters:
- atlasRequest: پارامترهای جستجو شامل searchToken و فیلدهای include
- args: project, namespace, headers, throwError
- configs: baseUrl, responseType

Output: داده JSON با آرایه result شامل اشیاء با id و keys""",

        """membasedAtlasKeyStreamAggregator

این پلاگین برای جمع‌آوری/جمع مقادیر عددی از کلیدهای داده‌های stream اطلس طراحی شده است.
قابلیت جمع بر اساس پیشوند کلید (پارامتر name) یا بر اساس اندیس کلید (پارامتر index) را دارد.

Parameters:
- atlasStream: stream ورودی داده‌های اطلس
- args: index (برای فیلتر بر اساس موقعیت) یا name (برای فیلتر بر اساس پیشوند)
- config: inputType باید "stream" باشد

Output: مجموع عددی مقادیر استخراج شده از کلیدهای منطبق"""
    ]
    
    # Parse plugins
    plugins = parse_plugin_descriptions(plugin_texts)
    print(f"📦 Loaded {len(plugins)} plugins:")
    for i, plugin in enumerate(plugins, 1):
        print(f"   {i}. {plugin['name']}")
    
    print()
    
    # Test questions from the user
    test_questions = [
        "مجموع مقادیر انتهایی کلیدهایی که با price_ شروع میشن رو بده",
        "مجموع هزینه‌ی انجام شده در پروژه‌ی target_project و فضای نام target_namespace", 
        "مجموع هزینه های انجام شده"
    ]
    
    print("🧠 Testing Plugin Sequencing Analysis:")
    print("-" * 40)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. سوال: {question}")
        
        # Create state for analysis
        state = {
            "question": question,
            "plugins": plugins,
            "goal": "plugin sequencing task"
        }
        
        # Analyze plugins to determine sequence
        result_state = analyze_plugins_simple(state)
        
        if "error" in result_state:
            print(f"   ❌ خطا: {result_state['error']}")
        else:
            sequence = result_state.get("plugin_sequence", "هیچ ترتیبی تعین نشد")
            print(f"   ✅ ترتیب پلاگین‌ها: {sequence}")
            
            # Show analysis details
            plan = result_state.get("plan", {})
            analysis = plan.get("analysis", "")
            if analysis:
                print(f"   💡 تحلیل: {analysis}")
    
    print()
    print("🎯 Expected Results:")
    print("   All questions should result in:")
    print("   HttpBasedAtlasReadByKey --> membasedAtlasKeyStreamAggregator")
    print()
    print("📋 System Architecture:")
    print("   1. Question Analysis (Persian NLP)")
    print("   2. Plugin Matching (Rule-based)")
    print("   3. Sequence Generation")
    print("   4. HITL Review Process")
    print()
    print("🌐 LangGraph Server:")
    print("   Run: langgraph dev")
    print("   API: http://127.0.0.1:8124")
    print("   Graph: agent")

if __name__ == "__main__":
    demo_plugin_sequencing()