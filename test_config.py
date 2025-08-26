#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set environment variables that might be needed
os.environ.setdefault('LOG_LEVEL', 'INFO')

from config import Configuration

def test_mcp_config():
    print("Testing MCP logging configuration...")
    config = Configuration()

    # Get the current config
    current_config = config._get_current_config()
    print(f"Current config logging section: {current_config.get('logging', {})}")

    # Test the get_mcp_logging_config method
    mcp_config = config.get_mcp_logging_config()
    print(f"MCP logging config result: {mcp_config}")

    # Check specific values
    print(f"tool_arguments enabled: {mcp_config.get('tool_arguments', 'NOT_FOUND')}")

    # Test the should_log_feature function
    import logging
    from src.chat.logging_utils import should_log_feature

    print("\nTesting should_log_feature function:")
    print(f"should_log_feature('mcp', 'tool_arguments'): {should_log_feature('mcp', 'tool_arguments')}")
    print(f"should_log_feature('chat', 'tool_arguments'): {should_log_feature('chat', 'tool_arguments')}")

    # Check if logging._module_features exists
    if hasattr(logging, '_module_features'):
        print(f"logging._module_features: {logging._module_features}")
    else:
        print("logging._module_features does not exist")

if __name__ == "__main__":
    test_mcp_config()
