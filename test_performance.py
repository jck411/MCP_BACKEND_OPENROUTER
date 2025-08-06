#!/usr/bin/env python3
"""
Performance Test for Request ID Indexing

This script demonstrates the performance improvement achieved by switching from
linear scan to O(1) indexing for duplicate event checking and request ID lookups.
"""
import asyncio
import time
import uuid
from src.history.chat_store import ChatEvent, InMemoryRepo, JsonlRepo


async def test_lookup_performance():
    """Test performance of request ID lookups with many events."""
    print("🚀 Testing Request ID Lookup Performance")
    print("=" * 50)
    
    # Create test repository
    repo = InMemoryRepo()
    conv_id = str(uuid.uuid4())
    
    # Add many events to simulate a large conversation history
    num_events = 10000
    print(f"📊 Creating {num_events} events...")
    
    event_request_ids = []
    start_time = time.time()
    
    for i in range(num_events):
        request_id = f"req_{i:06d}"
        event_request_ids.append(request_id)
        
        event = ChatEvent(
            conversation_id=conv_id,
            type="user_message" if i % 2 == 0 else "assistant_message",
            role="user" if i % 2 == 0 else "assistant",
            content=f"Test message {i}",
            extra={"request_id": request_id}
        )
        await repo.add_event(event)
    
    creation_time = time.time() - start_time
    print(f"✅ Created {num_events} events in {creation_time:.3f}s")
    
    # Test lookup performance - lookup events from different parts of the history
    test_request_ids = [
        event_request_ids[0],        # First event
        event_request_ids[num_events // 4],    # Quarter way
        event_request_ids[num_events // 2],    # Middle
        event_request_ids[3 * num_events // 4],  # Three quarters
        event_request_ids[-1],       # Last event
    ]
    
    print(f"\n🔍 Testing lookup performance for {len(test_request_ids)} events...")
    
    start_time = time.time()
    found_events = []
    
    for request_id in test_request_ids:
        event = await repo.get_event_by_request_id(conv_id, request_id)
        found_events.append(event)
    
    lookup_time = time.time() - start_time
    print(f"✅ Looked up {len(test_request_ids)} events in {lookup_time:.6f}s")
    print(f"📈 Average lookup time: {lookup_time/len(test_request_ids)*1000:.3f}ms per lookup")
    
    # Verify all lookups were successful
    assert all(event is not None for event in found_events), "Some lookups failed"
    assert all(event.extra.get("request_id") in test_request_ids for event in found_events), "Wrong events returned"
    
    # Test duplicate detection performance
    print(f"\n🔄 Testing duplicate detection performance...")
    start_time = time.time()
    
    # Try to add duplicate events (should all be rejected)
    duplicates_rejected = 0
    for i in range(0, min(100, num_events), 10):  # Test every 10th event
        duplicate_event = ChatEvent(
            conversation_id=conv_id,
            type="system_update",
            content=f"Duplicate attempt {i}",
            extra={"request_id": f"req_{i:06d}"}  # Same request_id as existing event
        )
        success = await repo.add_event(duplicate_event)
        if not success:
            duplicates_rejected += 1
    
    duplicate_time = time.time() - start_time
    print(f"✅ Tested {duplicates_rejected} duplicates in {duplicate_time:.6f}s")
    print(f"📈 Average duplicate check time: {duplicate_time/duplicates_rejected*1000:.3f}ms per check")
    
    print(f"\n🎯 Performance Summary:")
    print(f"   • Event creation: {num_events/creation_time:.0f} events/sec")
    print(f"   • Lookup speed: {len(test_request_ids)/lookup_time:.0f} lookups/sec")
    print(f"   • Duplicate detection: {duplicates_rejected/duplicate_time:.0f} checks/sec")
    print(f"   • Total events in conversation: {len(await repo.get_events(conv_id))}")


async def main():
    await test_lookup_performance()


if __name__ == "__main__":
    asyncio.run(main())
