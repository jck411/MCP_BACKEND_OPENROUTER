#!/usr/bin/env python3
"""Test script to verify configurable timeout and retry logic."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import yaml
from pathlib import Path

from src.config import Configuration
from src.main import MCPClient

# Set up logging
logging.basicConfig(level=logging.INFO)


def create_test_config(config_overrides: dict) -> Path:
    """Create a temporary YAML config file with specific values."""
    base_config = {
        "mcp": {
            "config_file": "servers_config.json",
            "connection": {
                "max_reconnect_attempts": 5,
                "initial_reconnect_delay": 1.0,
                "max_reconnect_delay": 30.0,
                "connection_timeout": 30.0,
                "ping_timeout": 10.0,
            }
        },
        "chat": {
            "websocket": {"host": "localhost", "port": 8000},
            "service": {"streaming": {"enabled": True}}
        },
        "llm": {
            "active": "openai",
            "providers": {"openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4"}}
        },
        "logging": {"level": "INFO"}
    }
    
    # Apply overrides
    def deep_update(base: dict, update: dict) -> dict:
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                deep_update(base[key], value)
            else:
                base[key] = value
        return base
    
    config = deep_update(base_config, config_overrides)
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(config, temp_file, default_flow_style=False)
    temp_file.close()
    
    return Path(temp_file.name)


async def test_configuration_loading():
    """Test that configuration values are properly loaded and applied."""
    print("🧪 Testing MCP Client Configuration Loading")
    
    # Test custom configuration
    config_overrides = {
        "mcp": {
            "connection": {
                "max_reconnect_attempts": 3,
                "initial_reconnect_delay": 0.5,
                "max_reconnect_delay": 10.0,
                "connection_timeout": 15.0,
                "ping_timeout": 5.0,
            }
        }
    }
    
    # Create temporary config file
    config_file = create_test_config(config_overrides)
    
    try:
        # Monkey patch the config loading to use our test file
        original_load_yaml = Configuration._load_yaml_config
        Configuration._load_yaml_config = lambda self: yaml.safe_load(config_file.read_text())
        
        config = Configuration()
        mcp_config = config.get_mcp_connection_config()
        
        # Verify configuration values
        assert mcp_config["max_reconnect_attempts"] == 3
        assert mcp_config["initial_reconnect_delay"] == 0.5
        assert mcp_config["max_reconnect_delay"] == 10.0
        assert mcp_config["connection_timeout"] == 15.0
        assert mcp_config["ping_timeout"] == 5.0
        
        print("  ✅ Custom configuration values loaded correctly")
        
        # Test MCPClient initialization with custom config
        server_config = {
            "command": "echo",
            "args": ["hello"],
            "enabled": True
        }
        
        client = MCPClient("test_client", server_config, mcp_config)
        
        # Verify the client has the right configuration
        assert client._max_reconnect_attempts == 3
        assert client._initial_reconnect_delay == 0.5
        assert client._max_reconnect_delay == 10.0
        assert client._connection_timeout == 15.0
        assert client._ping_timeout == 5.0
        
        print("  ✅ MCPClient initialized with custom configuration")
        
        # Restore original method
        Configuration._load_yaml_config = original_load_yaml
        
    finally:
        # Clean up temp file
        config_file.unlink()


async def test_validation():
    """Test configuration validation."""
    print("🧪 Testing Configuration Validation")
    
    # Test invalid max_reconnect_attempts
    config_overrides = {
        "mcp": {"connection": {"max_reconnect_attempts": 0}}
    }
    
    config_file = create_test_config(config_overrides)
    
    try:
        original_load_yaml = Configuration._load_yaml_config
        Configuration._load_yaml_config = lambda self: yaml.safe_load(config_file.read_text())
        
        config = Configuration()
        
        try:
            config.get_mcp_connection_config()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "max_reconnect_attempts must be at least 1" in str(e)
            print("  ✅ Invalid max_reconnect_attempts properly rejected")
        
        Configuration._load_yaml_config = original_load_yaml
        
    finally:
        config_file.unlink()
    
    # Test invalid delays
    config_overrides = {
        "mcp": {"connection": {"initial_reconnect_delay": 10.0, "max_reconnect_delay": 5.0}}
    }
    
    config_file = create_test_config(config_overrides)
    
    try:
        Configuration._load_yaml_config = lambda self: yaml.safe_load(config_file.read_text())
        
        config = Configuration()
        
        try:
            config.get_mcp_connection_config()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "max_reconnect_delay must be >= initial_reconnect_delay" in str(e)
            print("  ✅ Invalid delay configuration properly rejected")
        
        Configuration._load_yaml_config = original_load_yaml
        
    finally:
        config_file.unlink()


async def test_exponential_backoff():
    """Test that exponential backoff works correctly."""
    print("🧪 Testing Exponential Backoff Logic")
    
    server_config = {
        "command": "nonexistent_command_that_will_fail",
        "args": [],
        "enabled": True
    }
    
    connection_config = {
        "max_reconnect_attempts": 3,
        "initial_reconnect_delay": 0.1,  # Very short for testing
        "max_reconnect_delay": 1.0,
        "connection_timeout": 1.0,
        "ping_timeout": 1.0,
    }
    
    client = MCPClient("test_client", server_config, connection_config)
    
    # This should fail after 3 attempts
    start_time = asyncio.get_event_loop().time()
    
    try:
        await client.connect()
        assert False, "Connection should have failed"
    except Exception:
        pass  # Expected to fail
    
    end_time = asyncio.get_event_loop().time()
    elapsed = end_time - start_time
    
    # Should have taken some time due to delays (at least initial + 2*initial)
    # 0.1 + 0.2 = 0.3 seconds minimum, but let's be generous for timing variations
    assert elapsed >= 0.25, f"Elapsed time {elapsed} seems too short for exponential backoff"
    
    print(f"  ✅ Exponential backoff working (took {elapsed:.2f}s for 3 attempts)")
    
    # Verify delay was reset
    assert client._reconnect_delay == client._initial_reconnect_delay
    print("  ✅ Reconnect delay properly reset after failed connection")


async def main():
    """Run all tests."""
    print("🔬 Testing Configurable Timeout and Retry Logic")
    print("=" * 50)
    
    await test_configuration_loading()
    await test_validation()
    await test_exponential_backoff()
    
    print("\n🎉 All configuration tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
