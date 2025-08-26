#!/usr/bin/env python3
"""
Simple test script to verify tool argument logging
"""
import asyncio
import json
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Configuration

def test_logging_config():
    """Test that the logging configuration is properly set up"""
    print("Testing logging configuration...")

    config = Configuration()

    # Get MCP logging configuration
    mcp_config = config.get_mcp_logging_config()
    print(f"MCP Logging Config: {json.dumps(mcp_config, indent=2)}")

    # Manually configure logging like main.py does
    import logging
    from typing import Dict, Any

    def _configure_advanced_logging(logging_config: Dict[str, Any]) -> None:
        """Copy of the logging configuration from main.py"""
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }

        global_level = logging_config.get("level", "WARNING")
        logging.getLogger().setLevel(level_map.get(global_level, logging.WARNING))

        module_logger_map = {
            "chat": {
                "loggers": ["src.chat"],
                "default_level": "INFO",
                "features": ["llm_replies", "system_prompt", "tool_execution", "tool_results"]
            },
            "connection_pool": {
                "loggers": ["src.clients"],
                "default_level": "INFO",
                "features": ["connection_events", "pool_stats", "http_requests"]
            },
            "mcp": {
                "loggers": ["mcp", "src.clients.mcp_client"],
                "default_level": "INFO",
                "features": ["connection_attempts", "health_checks", "tool_calls", "tool_arguments", "tool_results"]
            }
        }

        modules_config = logging_config.get("modules", {})

        for module_name, module_config in modules_config.items():
            if not isinstance(module_config, dict):
                continue

            module_level = module_config.get("level", module_logger_map.get(module_name, {}).get("default_level", global_level))
            level_value = level_map.get(module_level, logging.WARNING)
            print(f"Module {module_name}: config_level={module_config.get('level')}, default_level={module_logger_map.get(module_name, {}).get('default_level')}, global_level={global_level}, final_level={module_level} ({level_value})")

            for logger_name in module_logger_map.get(module_name, {}).get("loggers", []):
                logger = logging.getLogger(logger_name)
                logger.setLevel(level_value)
                print(f"Set {logger_name} logger level to {level_value}")

            if not hasattr(logging, '_module_features'):
                logging._module_features = {}
            logging._module_features[module_name] = module_config.get("enable_features", {})

    # Apply logging configuration
    logging_config = config.get_logging_config()
    _configure_advanced_logging(logging_config)

    print(f"Applied logging configuration with global level: {logging_config.get('level', 'WARNING')}")

    # Now test the should_log_feature function
    from src.chat.logging_utils import should_log_feature

    # This should return True based on our configuration
    tool_args_enabled = should_log_feature("mcp", "tool_arguments")
    tool_calls_enabled = should_log_feature("mcp", "tool_calls")
    tool_results_enabled = should_log_feature("mcp", "tool_results")

    print(f"Tool arguments logging enabled: {tool_args_enabled}")
    print(f"Tool calls logging enabled: {tool_calls_enabled}")
    print(f"Tool results logging enabled: {tool_results_enabled}")

    # Test the log_tool_arguments function
    from src.chat.logging_utils import log_tool_arguments

    # Set up console logging to see the output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    # Debug: Check logger levels
    print(f"Root logger level: {logging.getLogger().getEffectiveLevel()}")
    print(f"src.chat logger level: {logging.getLogger('src.chat').getEffectiveLevel()}")
    print(f"src.chat.logging_utils logger level: {logging.getLogger('src.chat.logging_utils').getEffectiveLevel()}")

    test_args = {"param1": "value1", "param2": {"nested": "data"}}
    print("\n--- Testing log_tool_arguments function ---")
    log_tool_arguments("test_tool", test_args, "test_context")
    print("--- End test ---\n")

    print("Test completed successfully!")

if __name__ == "__main__":
    test_logging_config()
