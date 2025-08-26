#!/usr/bin/env python3
"""
Test to demonstrate the difference between feature-controlled LLM logging
and regular operational WebSocket logging
"""
import logging
import sys
sys.path.insert(0, 'src')

from src.config import Configuration
from src.chat.logging_utils import log_llm_reply, should_log_feature

def test_llm_logging():
    """Test the difference between feature-controlled and regular logging"""

    # Set up basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config = Configuration()

    # Apply logging configuration
    from src.main import _configure_advanced_logging
    logging_config = config.get_logging_config()
    _configure_advanced_logging(logging_config)

    print("=== LLM Logging Test ===\n")

    print("Current chat configuration:")
    chat_config = config.get_chat_service_config()
    print(f"llm_replies enabled: {chat_config.get('logging', {}).get('llm_replies', 'not found')}")
    print(f"Feature check result: {should_log_feature('chat', 'llm_replies')}\n")

    # Test 1: Feature-controlled LLM reply logging
    print("Test 1: Feature-controlled LLM reply logging")
    test_reply = {
        "message": {
            "content": "This is a test LLM response that should NOT appear if llm_replies is disabled",
            "role": "assistant"
        },
        "model": "test-model"
    }

    print("Calling log_llm_reply()...")
    log_llm_reply(test_reply, "Test response", chat_config)
    print("log_llm_reply() completed\n")

    # Test 2: Regular operational logging (like WebSocket messages)
    print("Test 2: Regular operational logging (WebSocket style)")
    logger = logging.getLogger('src.websocket_server')

    print("These are regular logger.info() calls, NOT controlled by feature flags:")
    logger.info("Sending WebSocket message: type=text, content=This WebSocket message...")
    logger.info("← LLM: streaming completed (hop 1), finish_reason=stop")
    logger.info("← Frontend: streaming response completed")

    print("\n" + "="*50)
    print("SUMMARY:")
    print("- Feature-controlled logging (log_llm_reply): respects your llm_replies: false setting")
    print("- Regular operational logging: always appears when logger level allows")
    print("- The 'Sending WebSocket message' logs are operational, not LLM content logging")

if __name__ == "__main__":
    test_llm_logging()
