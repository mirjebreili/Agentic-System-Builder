"""
Test the plugin analyzer with various input formats.
Run with: pytest tests/unit/test_plugin_analyzer.py -v
"""
from __future__ import annotations
import pytest
from langchain_core.messages import HumanMessage
from agents.plugin_analyzer import extract_system_elements


class TestPluginAnalyzer:
    """Test plugin extraction from various formats."""
    
    def test_no_plugins(self):
        """Test when no plugins are mentioned."""
        state = {
            "messages": [HumanMessage(content="Build a simple calculator app")]
        }
        result = extract_system_elements(state)
        
        assert result["has_system_elements"] is False
        assert len(result["plugins"]) == 0
    
    def test_json_format_simple(self):
        """Test extraction from simple JSON format."""
        prompt = """
        I have these plugins:
        {"plugins": [
            {"name": "UserAuth", "goal": "Handle user authentication"},
            {"name": "DataStore", "goal": "Manage database operations"}
        ]}
        """
        state = {"messages": [HumanMessage(content=prompt)]}
        result = extract_system_elements(state)
        
        assert result["has_system_elements"] is True
        assert len(result["plugins"]) == 2
        assert result["plugins"][0]["name"] == "UserAuth"
        assert result["plugins"][1]["name"] == "DataStore"
    
    def test_json_format_nested(self):
        """Test extraction from nested JSON format (delta_Rules)."""
        prompt = """
        Configuration:
        {"delta_Rules": {"plugins": [
            {"name": "EmailService", "goal": "Send emails to users"}
        ]}}
        """
        state = {"messages": [HumanMessage(content=prompt)]}
        result = extract_system_elements(state)
        
        assert result["has_system_elements"] is True
        assert len(result["plugins"]) >= 1
        plugin_names = [p["name"] for p in result["plugins"]]
        assert "EmailService" in plugin_names
    
    def test_markdown_format_bullets(self):
        """Test extraction from markdown bullet list."""
        prompt = """
        I have the following plugins:
        - UserAuth: Handles user authentication and sessions
        - PaymentGateway: Processes credit card payments
        - NotificationService: Sends push notifications
        
        Build a system using these.
        """
        state = {"messages": [HumanMessage(content=prompt)]}
        result = extract_system_elements(state)
        
        # Should extract at least some elements
        assert result["has_system_elements"] is True
        assert len(result["system_elements"]) > 0 or len(result["plugins"]) > 0
    
    def test_markdown_format_numbered(self):
        """Test extraction from numbered list."""
        prompt = """
        Plugins available:
        1. SearchPlugin - Handles full-text search
        2. CachePlugin - Manages caching layer
        3. LoggerPlugin - Centralized logging
        """
        state = {"messages": [HumanMessage(content=prompt)]}
        result = extract_system_elements(state)
        
        assert result["has_system_elements"] is True
        assert len(result["system_elements"]) > 0 or len(result["plugins"]) > 0
    
    def test_plain_text_format(self):
        """Test extraction from plain text description."""
        prompt = """
        I have a UserAuth plugin that handles user authentication,
        a DataStore plugin for managing data persistence,
        and an EmailService plugin that sends notifications.
        """
        state = {"messages": [HumanMessage(content=prompt)]}
        result = extract_system_elements(state)
        
        # LLM should be able to extract from natural language
        # If LLM fails, pattern matching might still catch some
        assert result is not None
        assert "plugins" in result
        assert "system_elements" in result
    
    def test_plugin_pattern_format(self):
        """Test extraction from @namespace/plugin pattern."""
        prompt = """
        Use these plugins:
        @myorg/auth-service - Authentication
        @myorg/data-store - Data persistence
        @myorg/notification - Send notifications
        """
        state = {"messages": [HumanMessage(content=prompt)]}
        result = extract_system_elements(state)
        
        assert result["has_system_elements"] is True
        assert len(result["system_elements"]) > 0 or len(result["plugins"]) > 0
    
    def test_mixed_format(self):
        """Test extraction when plugins are in mixed formats."""
        prompt = """
        I have these components:
        
        {"plugins": [{"name": "CoreAPI", "goal": "Main REST API"}]}
        
        Plus these additional services:
        - CacheService: Redis caching layer
        - QueueService: Message queue handler
        """
        state = {"messages": [HumanMessage(content=prompt)]}
        result = extract_system_elements(state)
        
        assert result["has_system_elements"] is True
        # Should extract from both JSON and markdown
        assert len(result["plugins"]) > 0 or len(result["system_elements"]) > 0
    
    def test_empty_message(self):
        """Test with empty messages list."""
        state = {"messages": []}
        result = extract_system_elements(state)
        
        assert result["has_system_elements"] is False
        assert len(result["plugins"]) == 0
    
    def test_no_human_message(self):
        """Test when messages don't contain human message."""
        from langchain_core.messages import AIMessage
        
        state = {"messages": [AIMessage(content="Some AI response")]}
        result = extract_system_elements(state)
        
        assert result["has_system_elements"] is False
        assert len(result["plugins"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
