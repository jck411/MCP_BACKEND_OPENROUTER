#!/usr/bin/env python3
"""
Test the request ID indexing optimization functionality.
"""
import asyncio
import tempfile
import uuid
import os
from src.history.chat_store import ChatEvent, InMemoryRepo, JsonlRepo


async def test_request_id_indexing():
    """Test that request ID indexing works correctly for both repositories."""
    print("Testing Request ID Indexing Optimization")
    print("=" * 50)
    
    # Test with InMemoryRepo
    print("\n📝 Testing InMemoryRepo...")
    await test_repo_indexing(InMemoryRepo(), "InMemoryRepo")
    
    # Test with JsonlRepo
    print("\n📝 Testing JsonlRepo...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name
    
    try:
        await test_repo_indexing(JsonlRepo(temp_path), "JsonlRepo")
    finally:
        os.unlink(temp_path)
    
    print("\n✅ All indexing tests passed!")


async def test_repo_indexing(repo, repo_name):
    """Test request ID indexing for a specific repository implementation."""
    conv_id = str(uuid.uuid4())
    
    # Create test events with request IDs
    events_data = [
        ("user_message", "user", "Hello", "req_001"),
        ("assistant_message", "assistant", "Hi there!", "req_002"),
        ("user_message", "user", "How are you?", "req_003"),
        ("assistant_message", "assistant", "I'm doing well", "req_004"),
        ("tool_call", None, None, "req_005"),
        ("tool_result", None, "Tool executed successfully", "req_006"),
    ]
    
    # Add events to repository
    added_events = []
    for msg_type, role, content, request_id in events_data:
        event = ChatEvent(
            conversation_id=conv_id,
            type=msg_type,
            role=role,
            content=content,
            extra={"request_id": request_id}
        )
        success = await repo.add_event(event)
        assert success, f"Failed to add event with request_id {request_id}"
        added_events.append(event)
    
    # Test O(1) lookups for all events
    print(f"  🔍 Testing lookups for {len(events_data)} events...")
    for i, (_, _, _, request_id) in enumerate(events_data):
        found_event = await repo.get_event_by_request_id(conv_id, request_id)
        assert found_event is not None, f"Failed to find event with request_id {request_id}"
        assert found_event.extra.get("request_id") == request_id, f"Wrong event returned for {request_id}"
        assert found_event.seq == i + 1, f"Wrong sequence number for {request_id}"
    
    # Test lookup for non-existent request ID
    missing_event = await repo.get_event_by_request_id(conv_id, "req_999")
    assert missing_event is None, "Should return None for non-existent request ID"
    
    # Test lookup in non-existent conversation
    missing_conv_event = await repo.get_event_by_request_id("missing_conv", "req_001")
    assert missing_conv_event is None, "Should return None for non-existent conversation"
    
    # Test duplicate prevention (should still work with indexing)
    duplicate_event = ChatEvent(
        conversation_id=conv_id,
        type="system_update",
        content="This is a duplicate",
        extra={"request_id": "req_001"}  # Same as first event
    )
    success = await repo.add_event(duplicate_event)
    assert not success, "Duplicate event should be rejected"
    
    # Verify original event is still findable
    original_event = await repo.get_event_by_request_id(conv_id, "req_001")
    assert original_event is not None, "Original event should still exist"
    assert original_event.type == "user_message", "Original event should be unchanged"
    
    print(f"  ✅ {repo_name} indexing works correctly")


if __name__ == "__main__":
    asyncio.run(test_request_id_indexing())
