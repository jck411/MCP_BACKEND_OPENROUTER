#!/usr/bin/env python3
"""
Test script to verify resource cleanup functionality.

This script tests the graceful shutdown and resource cleanup mechanisms
we've implemented for the MCP platform.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

from src.chat_service import ChatService
from src.config import Configuration
from src.history.chat_store import JsonlRepo

# Configure logging for the test
logging.basicConfig(level=logging.INFO)


async def test_chat_service_cleanup():
    """Test that ChatService properly closes LLM client during cleanup."""
    print("Testing ChatService cleanup...")
    
    # Create mock LLM client
    mock_llm_client = AsyncMock()
    mock_llm_client.close = AsyncMock()
    mock_llm_client.config = {"model": "test-model", "provider": "test"}
    
    # Create mock MCP clients
    mock_mcp_client = AsyncMock()
    mock_mcp_client.close = AsyncMock()
    mock_mcp_client.name = "test-client"
    
    # Create mock tool manager
    mock_tool_mgr = MagicMock()
    mock_tool_mgr.clients = [mock_mcp_client]
    
    # Create test configuration
    config = {"chat": {"service": {"system_prompt": "Test prompt"}}}
    test_config = Configuration()
    repo = JsonlRepo("test_events.jsonl")
    
    # Create ChatService
    service_config = ChatService.ChatServiceConfig(
        clients=[mock_mcp_client],
        llm_client=mock_llm_client,
        config=config,
        repo=repo,
        configuration=test_config,
    )
    
    chat_service = ChatService(service_config)
    
    # Set the tool manager manually (normally done in initialize)
    chat_service.tool_mgr = mock_tool_mgr
    
    # Test cleanup
    await chat_service.cleanup()
    
    # Verify both MCP client and LLM client were closed
    mock_mcp_client.close.assert_called_once()
    mock_llm_client.close.assert_called_once()
    
    print("✅ ChatService cleanup test passed - both MCP and LLM clients closed")


async def test_llm_client_context_manager():
    """Test that LLMClient properly closes HTTP connections as context manager."""
    from src.main import LLMClient
    
    print("Testing LLMClient context manager...")
    
    # Create test config
    config = {"base_url": "https://api.test.com", "model": "test-model"}
    api_key = "test-key"
    
    # Test that context manager properly closes HTTP client
    async with LLMClient(config, api_key) as llm_client:
        # Verify client is created
        assert llm_client.client is not None
        assert not llm_client.client.is_closed
    
    # After context manager exit, HTTP client should be closed
    assert llm_client.client.is_closed
    
    print("✅ LLMClient context manager test passed - HTTP client properly closed")


async def main():
    """Run all resource cleanup tests."""
    print("🔧 Testing Resource Cleanup Implementation")
    print("=" * 50)
    
    try:
        await test_chat_service_cleanup()
        await test_llm_client_context_manager()
        
        print("=" * 50)
        print("🎉 All resource cleanup tests passed!")
        print("\nKey improvements implemented:")
        print("1. ChatService.cleanup() now closes LLM HTTP client")
        print("2. WebSocket server has improved error handling and cleanup")
        print("3. Main function has signal handling for graceful shutdown")
        print("4. Proper async context manager usage ensures resource cleanup")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
