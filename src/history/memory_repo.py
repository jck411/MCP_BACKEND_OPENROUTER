#!/usr/bin/env python3
"""
In-Memory Chat Repository Implementation

Fast in-memory storage for session-only conversations.

CONFIG: storage.type = "memory"
PURPOSE: Development/testing - all data lost on restart
FEATURES: Fastest performance, no persistence, simple cleanup
"""
from __future__ import annotations

from .models import ChatEvent, Usage
from .repository import ChatRepository, _visible_to_llm


class InMemoryRepo(ChatRepository):
    """Fast in-memory storage - configure with type='memory'. Data lost on restart."""

    def __init__(self):
        self._conversations: dict[str, list[ChatEvent]] = {}
        self._seq_counters: dict[str, int] = {}
        self._request_cache: dict[str, dict[str, ChatEvent]] = {}

    async def add_event(self, event: ChatEvent) -> bool:
        conversation_id = event.conversation_id

        # Check for duplicates
        request_id = event.extra.get("request_id")
        if request_id:
            cache = self._request_cache.get(conversation_id, {})
            if request_id in cache:
                return False

        # Add to conversation
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = []
            self._seq_counters[conversation_id] = 0

        self._seq_counters[conversation_id] += 1
        event.seq = self._seq_counters[conversation_id]
        self._conversations[conversation_id].append(event)

        # Cache by request_id
        if request_id:
            if conversation_id not in self._request_cache:
                self._request_cache[conversation_id] = {}
            self._request_cache[conversation_id][request_id] = event

        return True

    async def get_events(
        self, conversation_id: str, limit: int | None = None
    ) -> list[ChatEvent]:
        events = self._conversations.get(conversation_id, [])
        return events[-limit:] if limit else events

    async def get_conversation_history(
        self, conversation_id: str, limit: int | None = None
    ) -> list[ChatEvent]:
        events = self._conversations.get(conversation_id, [])
        filtered = [ev for ev in events if _visible_to_llm(ev)]
        return filtered[-limit:] if limit else filtered

    async def list_conversations(self) -> list[str]:
        return list(self._conversations.keys())

    async def get_event_by_request_id(
        self, conversation_id: str, request_id: str
    ) -> ChatEvent | None:
        cache = self._request_cache.get(conversation_id, {})
        return cache.get(request_id)

    async def get_last_assistant_reply_id(
        self, conversation_id: str, user_request_id: str
    ) -> str | None:
        events = self._conversations.get(conversation_id, [])
        for event in reversed(events):
            if (event.type == "assistant_message" and
                event.extra.get("user_request_id") == user_request_id):
                return event.id
        return None

    async def compact_deltas(
        self, conversation_id: str, user_request_id: str, final_content: str,
        usage: Usage | None = None, model: str | None = None
    ) -> ChatEvent:
        """Compact delta events into a single assistant_message and remove deltas."""
        events = self._conversations.get(conversation_id, [])

        # Check if assistant message already exists
        assistant_req_id = f"assistant:{user_request_id}"
        for event in events:
            if event.extra.get("request_id") == assistant_req_id:
                return event

        # Remove delta events
        self._conversations[conversation_id] = [
            ev for ev in events
            if not (ev.type == "meta" and
                   ev.extra.get("kind") == "assistant_delta" and
                   ev.extra.get("user_request_id") == user_request_id)
        ]

        # Create final assistant message
        assistant_event = ChatEvent(
            conversation_id=conversation_id,
            type="assistant_message",
            role="assistant",
            content=final_content,
            usage=usage,
            model=model,
            extra={
                "user_request_id": user_request_id,
                "request_id": assistant_req_id
            }
        )

        await self.add_event(assistant_event)
        return assistant_event

    async def handle_clear_session(self) -> bool:
        """Clear all conversation data from memory."""
        self._conversations.clear()
        self._seq_counters.clear()
        self._request_cache.clear()
        return True
