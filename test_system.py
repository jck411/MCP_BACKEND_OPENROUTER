#!/usr/bin/env python3
"""
MCP Backend System Test

This script tests the core functionality of the MCP backend system:
- ChatEvent model and validation
- Repository implementations (InMemory and JSONL)
- Prompt history filtering (_visible_to_llm)
- Sequence number assignment
- Token counting

Run this script to verify everything is working correctly.

This is a test file - print statements are intentional for test output.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import uuid
from pathlib import Path

from src.history.chat_store import (
    ChatEvent, InMemoryRepo, JsonlRepo, _visible_to_llm, Usage
)


def test_chat_event_model():
    """Test ChatEvent model and seq behavior."""
    print("🧪 Testing ChatEvent model...")
    
    # Test that seq defaults to None
    event = ChatEvent(
        conversation_id="test",
        type="user_message",
        role="user",
        content="Hello"
    )
    assert event.seq is None, f"Expected seq=None, got {event.seq}"
    print("  ✅ seq defaults to None")
    
    # Test token counting
    event.compute_and_cache_tokens()
    assert event.token_count is not None and event.token_count > 0
    print(f"  ✅ Token counting works: {event.token_count} tokens")


def test_visibility_filter():
    """Test the _visible_to_llm filter function."""
    print("🧪 Testing _visible_to_llm filter...")
    
    test_cases = [
        (ChatEvent(conversation_id="test", type="user_message", role="user", content="Hi"), True),
        (ChatEvent(conversation_id="test", type="assistant_message", role="assistant", content="Hello"), True),
        (ChatEvent(conversation_id="test", type="tool_result", content="Tool output"), True),
        (ChatEvent(conversation_id="test", type="meta", extra={"kind": "assistant_delta"}), False),
        (ChatEvent(conversation_id="test", type="tool_call"), False),
        (ChatEvent(conversation_id="test", type="system_update", content="System message"), False),
        (ChatEvent(conversation_id="test", type="system_update", content="System message", extra={"visible_to_model": True}), True),
        (ChatEvent(conversation_id="test", type="system_update", content="System message", extra={"visible_to_model": False}), False),
    ]
    
    for event, expected in test_cases:
        result = _visible_to_llm(event)
        assert result == expected, f"Expected {expected} for {event.type}, got {result}"
        status = "✅" if result == expected else "❌"
        print(f"  {status} {event.type} -> {result}")


async def test_in_memory_repo():
    """Test InMemoryRepo functionality."""
    print("🧪 Testing InMemoryRepo...")
    
    repo = InMemoryRepo()
    conv_id = str(uuid.uuid4())
    
    # Test adding events and seq assignment
    user_event = ChatEvent(
        conversation_id=conv_id,
        type="user_message",
        role="user",
        content="Hello, world!"
    )
    
    # Before adding - seq should be None
    assert user_event.seq is None
    
    # Add event
    success = await repo.add_event(user_event)
    assert success, "Failed to add event"
    assert user_event.seq == 1, f"Expected seq=1, got {user_event.seq}"
    print("  ✅ Event added and seq assigned correctly")
    
    # Add assistant event
    asst_event = ChatEvent(
        conversation_id=conv_id,
        type="assistant_message",
        role="assistant",
        content="Hi there!"
    )
    await repo.add_event(asst_event)
    assert asst_event.seq == 2, f"Expected seq=2, got {asst_event.seq}"
    
    # Add a meta event that should be filtered
    meta_event = ChatEvent(
        conversation_id=conv_id,
        type="meta",
        extra={"kind": "assistant_delta", "user_request_id": "test123"}
    )
    await repo.add_event(meta_event)
    assert meta_event.seq == 3
    
    # Test last_n_tokens filtering
    filtered_events = await repo.last_n_tokens(conv_id, 1000)
    assert len(filtered_events) == 2, f"Expected 2 visible events, got {len(filtered_events)}"
    
    # Should only contain user_message and assistant_message (meta filtered out)
    types = [ev.type for ev in filtered_events]
    assert "user_message" in types and "assistant_message" in types
    assert "meta" not in types
    print("  ✅ Filtering works correctly in last_n_tokens")
    
    # Test duplicate prevention
    duplicate = ChatEvent(
        conversation_id=conv_id,
        type="user_message",
        role="user",
        content="Duplicate",
        extra={"request_id": "unique123"}
    )
    success1 = await repo.add_event(duplicate)
    assert success1, "First event should succeed"
    
    duplicate2 = ChatEvent(
        conversation_id=conv_id,
        type="user_message", 
        role="user",
        content="Different content",
        extra={"request_id": "unique123"}  # Same request_id
    )
    success2 = await repo.add_event(duplicate2)
    assert not success2, "Duplicate should be rejected"
    print("  ✅ Duplicate detection works")


async def test_jsonl_repo():
    """Test JsonlRepo functionality."""
    print("🧪 Testing JsonlRepo...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name
    
    try:
        repo = JsonlRepo(temp_path)
        conv_id = str(uuid.uuid4())
        
        # Test adding events
        user_event = ChatEvent(
            conversation_id=conv_id,
            type="user_message",
            role="user",
            content="Testing JSONL repo"
        )
        
        success = await repo.add_event(user_event)
        assert success, "Failed to add event to JSONL repo"
        assert user_event.seq == 1, f"Expected seq=1, got {user_event.seq}"
        
        # Add assistant event  
        asst_event = ChatEvent(
            conversation_id=conv_id,
            type="assistant_message",
            role="assistant", 
            content="JSONL response"
        )
        await repo.add_event(asst_event)
        assert asst_event.seq == 2
        
        # Test persistence by creating new repo instance
        repo2 = JsonlRepo(temp_path)
        events = await repo2.get_events(conv_id)
        assert len(events) == 2, f"Expected 2 persisted events, got {len(events)}"
        
        # Test filtering
        filtered = await repo2.last_n_tokens(conv_id, 1000)
        assert len(filtered) == 2, "All events should be visible"
        print("  ✅ JSONL persistence and filtering works")
        
    finally:
        Path(temp_path).unlink(missing_ok=True)


async def test_compact_deltas():
    """Test delta compaction functionality."""
    print("🧪 Testing compact_deltas...")
    
    repo = InMemoryRepo()
    conv_id = str(uuid.uuid4())
    user_request_id = str(uuid.uuid4())
    
    # Add some delta events
    for i in range(3):
        delta_event = ChatEvent(
            conversation_id=conv_id,
            type="meta",
            extra={
                "kind": "assistant_delta",
                "user_request_id": user_request_id,
                "delta": f"chunk {i}"
            }
        )
        await repo.add_event(delta_event)
    
    # Compact deltas
    final_content = "This is the final message"
    usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    
    compact_event = await repo.compact_deltas(
        conv_id, user_request_id, final_content, usage, "test-model"
    )
    
    assert compact_event.type == "assistant_message"
    assert compact_event.content == final_content
    assert compact_event.usage == usage
    assert compact_event.seq is not None
    print(f"  ✅ Delta compaction works, final seq: {compact_event.seq}")
    
    # Verify deltas were removed and final message is visible
    filtered = await repo.last_n_tokens(conv_id, 1000)
    assert len(filtered) == 1
    assert filtered[0].type == "assistant_message"
    assert filtered[0].content == final_content
    print("  ✅ Deltas filtered out, final message visible")


async def main():
    """Run all tests."""
    print("🚀 Starting MCP Backend System Tests\n")
    
    try:
        test_chat_event_model()
        print()
        
        test_visibility_filter()
        print()
  # noqa: W293
        await test_in_memory_repo()
        print()
  # noqa: W293
        await test_jsonl_repo()
        print()
  # noqa: W293
        await test_compact_deltas()
        print()
        
        print("🎉 All tests passed! System is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
