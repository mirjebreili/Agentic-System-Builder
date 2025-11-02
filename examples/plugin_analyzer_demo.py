#!/usr/bin/env python3
"""
Demo script showing the plugin analyzer extracting plugins from various formats.

Run with: python examples/plugin_analyzer_demo.py
"""
from __future__ import annotations
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.plugin_analyzer import extract_system_elements
from langchain_core.messages import HumanMessage
import json


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def print_results(result: dict):
    """Print extraction results in a readable format."""
    print(f"\n✓ Has system elements: {result['has_system_elements']}")
    print(f"✓ Plugins found: {len(result['plugins'])}")
    print(f"✓ System elements: {len(result['system_elements'])}")
    
    if result['plugins']:
        print("\nExtracted Plugins:")
        for i, plugin in enumerate(result['plugins'], 1):
            print(f"  {i}. {plugin['name']}")
            print(f"     Goal: {plugin['goal']}")
            print(f"     Type: {plugin.get('type', 'plugin')}")
    
    if result['system_elements']:
        print("\nSystem Elements:")
        for i, elem in enumerate(result['system_elements'], 1):
            print(f"  {i}. {elem}")


def demo_json_format():
    """Demo: Extract from JSON format."""
    print_section("Demo 1: JSON Format")
    
    prompt = """
    I have these plugins defined:
    {
      "plugins": [
        {"name": "UserAuthPlugin", "goal": "Handle user authentication and authorization"},
        {"name": "DataStoragePlugin", "goal": "Manage database operations and data persistence"},
        {"name": "EmailServicePlugin", "goal": "Send email notifications to users"}
      ]
    }
    
    Build a user management system using these plugins.
    """
    
    print(f"Prompt:\n{prompt}")
    
    state = {"messages": [HumanMessage(content=prompt)]}
    result = extract_system_elements(state)
    
    print_results(result)


def demo_markdown_format():
    """Demo: Extract from markdown bullet list."""
    print_section("Demo 2: Markdown Bullet List")
    
    prompt = """
    I have the following microservices:
    - PaymentGateway: Processes credit card transactions and refunds
    - InventoryManager: Tracks product stock levels
    - ShippingTracker: Monitors package delivery status
    - NotificationHub: Sends SMS and email alerts
    
    Create an e-commerce backend using these services.
    """
    
    print(f"Prompt:\n{prompt}")
    
    state = {"messages": [HumanMessage(content=prompt)]}
    result = extract_system_elements(state)
    
    print_results(result)


def demo_plain_text_format():
    """Demo: Extract from natural language."""
    print_section("Demo 3: Plain Text / Natural Language")
    
    prompt = """
    I have a SearchEngine plugin that provides full-text search capabilities,
    a CacheManager plugin for managing Redis cache,
    and a LogAggregator plugin that collects and analyzes logs.
    
    Build a monitoring dashboard.
    """
    
    print(f"Prompt:\n{prompt}")
    
    state = {"messages": [HumanMessage(content=prompt)]}
    result = extract_system_elements(state)
    
    print_results(result)


def demo_numbered_list():
    """Demo: Extract from numbered list."""
    print_section("Demo 4: Numbered List")
    
    prompt = """
    Available plugins:
    1. AuthService - JWT-based authentication
    2. FileStorage - S3-compatible file storage
    3. WebSocket - Real-time bidirectional communication
    4. RateLimiter - API rate limiting and throttling
    """
    
    print(f"Prompt:\n{prompt}")
    
    state = {"messages": [HumanMessage(content=prompt)]}
    result = extract_system_elements(state)
    
    print_results(result)


def demo_plugin_pattern():
    """Demo: Extract from @namespace/plugin pattern."""
    print_section("Demo 5: Plugin Pattern (@namespace/name)")
    
    prompt = """
    Use these plugins:
    @acme/authentication - User login and session management
    @acme/database-orm - Object-relational mapping for PostgreSQL
    @acme/message-queue - RabbitMQ integration
    """
    
    print(f"Prompt:\n{prompt}")
    
    state = {"messages": [HumanMessage(content=prompt)]}
    result = extract_system_elements(state)
    
    print_results(result)


def demo_no_plugins():
    """Demo: No plugins mentioned."""
    print_section("Demo 6: No Plugins (Abstract Mode)")
    
    prompt = """
    Build a todo list application with user accounts,
    task management, and notifications.
    """
    
    print(f"Prompt:\n{prompt}")
    
    state = {"messages": [HumanMessage(content=prompt)]}
    result = extract_system_elements(state)
    
    print_results(result)
    print("\n→ This will trigger abstract planning mode (no concrete plugins)")


def demo_mixed_format():
    """Demo: Mixed formats in one prompt."""
    print_section("Demo 7: Mixed Formats")
    
    prompt = """
    I have these core plugins:
    {"plugins": [{"name": "APIGateway", "goal": "Route and manage API requests"}]}
    
    Plus these additional services:
    - DatabasePool: Connection pooling for MySQL
    - CDNManager: Cloudflare CDN integration
    
    And also a @internal/metrics-collector for monitoring.
    """
    
    print(f"Prompt:\n{prompt}")
    
    state = {"messages": [HumanMessage(content=prompt)]}
    result = extract_system_elements(state)
    
    print_results(result)


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("  PLUGIN ANALYZER DEMO")
    print("  Extracting plugins from various formats")
    print("="*60)
    
    demos = [
        demo_json_format,
        demo_markdown_format,
        demo_plain_text_format,
        demo_numbered_list,
        demo_plugin_pattern,
        demo_no_plugins,
        demo_mixed_format,
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\n✗ Error in demo: {e}")
    
    print("\n" + "="*60)
    print("  Demo Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
