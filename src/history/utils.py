#!/usr/bin/env python3
"""
Chat History Utilities

Helper functions and utilities for the chat history system.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid

from .models import ChatEvent, ToolCall
from .repository import ChatRepository, visible_to_llm
from .sqlite_repo import SQLiteRepo

logger = logging.getLogger(__name__)


async def demo_basic_storage():
    """Test basic storage and retrieval functionality."""
    # Use a separate test database to avoid polluting production data
    test_db = "test_chat_events.db"
    repo: ChatRepository = SQLiteRepo(test_db)
    conv_id = str(uuid.uuid4())

    logger.info("=== Testing Basic Storage ===")

    # Test user message
    user_ev = ChatEvent(
        conversation_id=conv_id,
        type="user_message",
        role="user",
        content="Hello!",
        extra={"request_id": "user_001"},
    )
    success = await repo.add_event(user_ev)
    logger.info(f"User message added: {success}")

    # Test assistant message with tool calls
    tool_call = ToolCall(
        id="call_123", name="get_weather", arguments={"location": "San Francisco"}
    )
    asst_ev = ChatEvent(
        conversation_id=conv_id,
        type="assistant_message",
        role="assistant",
        content="Let me check the weather for you.",
        tool_calls=[tool_call],
        provider="test_provider",
        model="test-model",
        extra={"request_id": "assistant_001", "user_request_id": "user_001"},
    )
    success = await repo.add_event(asst_ev)
    logger.info(f"Assistant message added: {success}")

    # Test tool result
    tool_result_ev = ChatEvent(
        conversation_id=conv_id,
        type="tool_result",
        content="The weather in San Francisco is 72°F and sunny.",
        extra={"tool_call_id": "call_123", "request_id": "tool_001"},
    )
    success = await repo.add_event(tool_result_ev)
    logger.info(f"Tool result added: {success}")

    # Test duplicate prevention
    duplicate = await repo.add_event(user_ev)
    logger.info(f"Duplicate prevention works: {not duplicate}")

    # Show conversation history
    logger.info(f"\n=== Conversation {conv_id[:8]}... ===")
    events = await repo.get_events(conv_id)
    for ev in events:
        content_preview = str(ev.content)[:50] + "..." if ev.content else "None"
        logger.info(f"- seq={ev.seq} {ev.type} {ev.role} content={content_preview}")

    # Test LLM context filtering
    logger.info("\n=== LLM Context (filtered) ===")
    context_events = await repo.get_conversation_history(conv_id)
    for ev in context_events:
        visible = visible_to_llm(ev)
        logger.info(f"- {ev.type} {ev.role} visible_to_llm={visible}")

    # Cleanup test database
    if os.path.exists(test_db):
        os.remove(test_db)
        logger.info(f"Cleaned up test database: {test_db}")


async def main():
    """Run development tests and demos."""
    try:
        await demo_basic_storage()
        logger.info("\n✅ All tests passed!")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    import logging

    # Configure basic logging for standalone execution
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    asyncio.run(main())
