#!/usr/bin/env python3
"""
Hybrid Chat Repository Implementation

Combines in-memory for current session with optional persistence.
"""
from __future__ import annotations

from typing import Any

from .memory_repo import InMemoryRepo
from .models import ChatEvent, Usage
from .repository import ChatRepository
from .sqlite_repo import SQLiteRepo


class HybridRepo(ChatRepository):
    """Combines in-memory for current session with optional persistence."""

    def __init__(self, config: dict[str, Any]):
        self.memory_repo = InMemoryRepo()
        storage_config = config.get("chat", {}).get("storage", {})
        db_path = storage_config.get("persistence", {}).get(
            "db_path", "chat_history.db"
        )
        self.persistent_repo = SQLiteRepo(db_path)
        self.config = config
        self.auto_save = storage_config.get("persistence", {}).get("auto_save", False)

    async def add_event(self, event: ChatEvent) -> bool:
        # Always add to memory for fast access
        success = await self.memory_repo.add_event(event)

        # Optionally save to disk
        if self.auto_save or event.extra.get("force_persist", False):
            await self.persistent_repo.add_event(event)

        return success

    async def get_events(
        self, conversation_id: str, limit: int | None = None
    ) -> list[ChatEvent]:
        return await self.memory_repo.get_events(conversation_id, limit)

    async def get_conversation_history(
        self, conversation_id: str, limit: int | None = None
    ) -> list[ChatEvent]:
        return await self.memory_repo.get_conversation_history(conversation_id, limit)

    async def list_conversations(self) -> list[str]:
        return await self.memory_repo.list_conversations()

    async def get_event_by_request_id(
        self, conversation_id: str, request_id: str
    ) -> ChatEvent | None:
        return await self.memory_repo.get_event_by_request_id(
            conversation_id,
            request_id,
        )

    async def get_last_assistant_reply_id(
        self, conversation_id: str, user_request_id: str
    ) -> str | None:
        return await self.memory_repo.get_last_assistant_reply_id(
            conversation_id, user_request_id
        )

    async def compact_deltas(
        self, conversation_id: str, user_request_id: str, final_content: str,
        usage: Usage | None = None, model: str | None = None
    ) -> ChatEvent:
        result = await self.memory_repo.compact_deltas(
            conversation_id, user_request_id, final_content, usage, model
        )

        # Also save to persistent if auto_save is enabled
        if self.auto_save:
            await self.persistent_repo.compact_deltas(
                conversation_id, user_request_id, final_content, usage, model
            )

        return result

    async def save_conversation(self, conversation_id: str) -> bool:
        """Manually save a conversation to persistent storage."""
        events = await self.memory_repo.get_events(conversation_id)
        for event in events:
            await self.persistent_repo.add_event(event)
        return True

    async def load_conversation(self, conversation_id: str) -> bool:
        """Load a conversation from persistent storage to memory."""
        events = await self.persistent_repo.get_events(conversation_id)
        for event in events:
            await self.memory_repo.add_event(event)
        return len(events) > 0

    async def list_persistent_conversations(self) -> list[str]:
        """List conversations available in persistent storage."""
        return await self.persistent_repo.list_conversations()
