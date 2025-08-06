#!/usr/bin/env python3
"""
Chat History Storage Module

This module provides a simplified chat event storage system for the MCP Platform.
It manages conversation history and persistent storage with SQLite backend.

Key Components:
- ChatEvent: Pydantic model representing individual chat messages, tool calls,
  and system events
- ChatRepository: Protocol defining the interface for chat storage backends
- SQLiteRepo: High-performance SQLite storage for production use

Features:
- Comprehensive event tracking (user messages, assistant responses, tool calls,
  system updates)
- Duplicate detection via request IDs
- Conversation-based organization
- Thread-safe operations
- Async/await support
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Protocol

import aiosqlite
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------- Canonical models (Pydantic v2) ----------

Role = Literal["system", "user", "assistant", "tool"]

class StorageMode(str, Enum):
    """Available storage modes for chat history."""
    SESSION_ONLY = "session_only"
    PERSISTENT = "persistent"
    HYBRID = "hybrid"

class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str

Part = TextPart  # extend later

class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    # sequence must always be filled by the repo; start with None
    seq: int | None = None
    schema_version: int = 1
    type: Literal[
        "user_message",
        "assistant_message",
        "tool_call",
        "tool_result",
        "system_update",
        "meta"
    ]
    role: Role | None = None
    content: str | list[Part] | None = None
    tool_calls: list[ToolCall] = []
    usage: Usage | None = None
    provider: str | None = None
    model: str | None = None
    stop_reason: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    extra: dict[str, Any] = Field(default_factory=dict)
    raw: Any | None = None  # keep small; move big things elsewhere later

# ---------- Repository interface ----------

# Context filter for LLM conversation history
_CONTEXT_TYPES = {"user_message", "assistant_message", "tool_result"}

def _visible_to_llm(ev: ChatEvent) -> bool:
    """
    Determine if a chat event should be included in LLM conversation context.

    Includes:
    - user_message: Direct user inputs
    - assistant_message: LLM responses
    - tool_result: Results from tool executions
    - system_update: Only when explicitly marked as visible_to_model=True
    """
    if ev.type in _CONTEXT_TYPES:
        return True
    return ev.type == "system_update" and ev.extra.get("visible_to_model", False)

class ChatRepository(Protocol):
    async def add_event(self, event: ChatEvent) -> bool: ...
    async def get_events(
        self, conversation_id: str, limit: int | None = None
    ) -> list[ChatEvent]: ...
    async def get_conversation_history(
        self, conversation_id: str, limit: int | None = None
    ) -> list[ChatEvent]: ...
    async def list_conversations(self) -> list[str]: ...
    async def get_event_by_request_id(
        self, conversation_id: str, request_id: str
    ) -> ChatEvent | None: ...
    async def get_last_assistant_reply_id(
        self, conversation_id: str, user_request_id: str
    ) -> str | None: ...
    async def compact_deltas(
        self, conversation_id: str, user_request_id: str, final_content: str,
        usage: Usage | None = None, model: str | None = None
    ) -> ChatEvent: ...

# ---------- SQLite implementation ----------

class SQLiteRepo(ChatRepository):
    """High-performance SQLite storage for chat history."""

    def __init__(self, db_path: str = "chat_history.db"):
        self.db_path = db_path
        self._lock = asyncio.Lock()
        self._initialized = False

    async def _ensure_initialized(self):
        """Initialize database schema if not already done."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            async with aiosqlite.connect(self.db_path) as db:
                # Enable WAL mode for better concurrency
                await db.execute("PRAGMA journal_mode=WAL")
                await db.execute("PRAGMA synchronous=NORMAL")
                await db.execute("PRAGMA cache_size=10000")
                await db.execute("PRAGMA temp_store=memory")

                # Create schema
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS chat_events (
                        id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        seq INTEGER NOT NULL,
                        schema_version INTEGER NOT NULL,
                        type TEXT NOT NULL,
                        role TEXT,
                        content TEXT,
                        tool_calls TEXT,
                        usage TEXT,
                        provider TEXT,
                        model TEXT,
                        stop_reason TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        extra TEXT,
                        raw TEXT,
                        request_id TEXT GENERATED ALWAYS AS (
                            json_extract(extra, '$.request_id')
                        ) VIRTUAL
                    )
                """)

                # Create indexes
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_conversation
                    ON chat_events(conversation_id)
                """)
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_conversation_seq
                    ON chat_events(conversation_id, seq)
                """)
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_created
                    ON chat_events(created_at)
                """)
                await db.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_request_id
                    ON chat_events(conversation_id, request_id)
                    WHERE request_id IS NOT NULL
                """)

                await db.commit()

            self._initialized = True

    def _serialize_event(self, event: ChatEvent) -> dict[str, Any]:
        """Convert ChatEvent to database row format."""
        return {
            "id": event.id,
            "conversation_id": event.conversation_id,
            "seq": event.seq,
            "schema_version": event.schema_version,
            "type": event.type,
            "role": event.role,
            "content": (
                event.content if isinstance(event.content, str)
                else json.dumps(event.content)
            ),
            "tool_calls": (
                json.dumps([tc.model_dump() for tc in event.tool_calls])
                if event.tool_calls else None
            ),
            "usage": event.usage.model_dump_json() if event.usage else None,
            "provider": event.provider,
            "model": event.model,
            "stop_reason": event.stop_reason,
            "created_at": event.created_at.isoformat(),
            "extra": json.dumps(event.extra) if event.extra else None,
            "raw": json.dumps(event.raw) if event.raw else None,
        }

    def _deserialize_event(self, row: dict[str, Any]) -> ChatEvent:
        """Convert database row to ChatEvent."""
        # Parse JSON fields
        content = row["content"]
        if content and content.startswith('['):
            with contextlib.suppress(json.JSONDecodeError):
                content = json.loads(content)

        tool_calls = []
        if row["tool_calls"]:
            tool_calls = [ToolCall(**tc) for tc in json.loads(row["tool_calls"])]

        usage = None
        if row["usage"]:
            usage = Usage.model_validate_json(row["usage"])

        extra = json.loads(row["extra"]) if row["extra"] else {}
        raw = json.loads(row["raw"]) if row["raw"] else None

        return ChatEvent(
            id=row["id"],
            conversation_id=row["conversation_id"],
            seq=row["seq"],
            schema_version=row["schema_version"],
            type=row["type"],
            role=row["role"],
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            provider=row["provider"],
            model=row["model"],
            stop_reason=row["stop_reason"],
            created_at=datetime.fromisoformat(row["created_at"]),
            extra=extra,
            raw=raw,
        )

    async def add_event(self, event: ChatEvent) -> bool:
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            # Check for duplicate by request_id
            request_id = event.extra.get("request_id")
            if request_id:
                async with db.execute(
                    ("SELECT 1 FROM chat_events WHERE conversation_id = ? "
                     "AND request_id = ?"),
                    (event.conversation_id, request_id)
                ) as cursor:
                    if await cursor.fetchone():
                        return False  # Duplicate found

            # Get next sequence number
            async with db.execute(
                ("SELECT COALESCE(MAX(seq), 0) + 1 FROM chat_events "
                 "WHERE conversation_id = ?"),
                (event.conversation_id,)
            ) as cursor:
                row = await cursor.fetchone()
                event.seq = row[0] if row else 1

            # Insert event
            row_data = self._serialize_event(event)
            columns = ", ".join(row_data.keys())
            placeholders = ", ".join("?" * len(row_data))

            await db.execute(
                f"INSERT INTO chat_events ({columns}) VALUES ({placeholders})",
                list(row_data.values())
            )
            await db.commit()
            return True

    async def get_events(
        self, conversation_id: str, limit: int | None = None
    ) -> list[ChatEvent]:
        await self._ensure_initialized()

        query = "SELECT * FROM chat_events WHERE conversation_id = ? ORDER BY seq"
        params: list[Any] = [conversation_id]

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [self._deserialize_event(dict(row)) for row in rows]

    async def get_conversation_history(
        self, conversation_id: str, limit: int | None = None
    ) -> list[ChatEvent]:
        await self._ensure_initialized()

        # Only get events visible to LLM
        query = """
            SELECT * FROM chat_events
            WHERE conversation_id = ?
            AND (
                type IN ('user_message', 'assistant_message', 'tool_result')
                OR (type = 'system_update' AND
                    json_extract(extra, '$.visible_to_model') = true)
            )
            ORDER BY seq
        """
        params: list[Any] = [conversation_id]

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [self._deserialize_event(dict(row)) for row in rows]

    async def list_conversations(self) -> list[str]:
        await self._ensure_initialized()

        async with (
            aiosqlite.connect(self.db_path) as db,
            db.execute("SELECT DISTINCT conversation_id FROM chat_events") as cursor,
        ):
            rows = await cursor.fetchall()
            return [row[0] for row in rows]

    async def get_event_by_request_id(
        self, conversation_id: str, request_id: str
    ) -> ChatEvent | None:
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                ("SELECT * FROM chat_events WHERE conversation_id = ? "
                 "AND request_id = ?"),
                (conversation_id, request_id)
            ) as cursor:
                row = await cursor.fetchone()
                return self._deserialize_event(dict(row)) if row else None

    async def get_last_assistant_reply_id(
        self, conversation_id: str, user_request_id: str
    ) -> str | None:
        await self._ensure_initialized()

        async with (
            aiosqlite.connect(self.db_path) as db,
            db.execute(
                """
                SELECT id FROM chat_events
                WHERE conversation_id = ?
                AND type = 'assistant_message'
                AND json_extract(extra, '$.user_request_id') = ?
                ORDER BY seq DESC
                LIMIT 1
                """,
                (conversation_id, user_request_id)
            ) as cursor,
        ):
            row = await cursor.fetchone()
            return row[0] if row else None

    async def compact_deltas(
        self, conversation_id: str, user_request_id: str, final_content: str,
        usage: Usage | None = None, model: str | None = None
    ) -> ChatEvent:
        """Compact delta events into a single assistant_message and remove deltas."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            # Check if assistant message already exists
            assistant_req_id = f"assistant:{user_request_id}"
            db.row_factory = aiosqlite.Row
            async with db.execute(
                ("SELECT * FROM chat_events WHERE conversation_id = ? "
                 "AND request_id = ?"),
                (conversation_id, assistant_req_id)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._deserialize_event(dict(row))

            # Remove delta events
            await db.execute(
                """
                DELETE FROM chat_events
                WHERE conversation_id = ?
                AND type = 'meta'
                AND json_extract(extra, '$.kind') = 'assistant_delta'
                AND json_extract(extra, '$.user_request_id') = ?
                """,
                (conversation_id, user_request_id)
            )

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

            # Get sequence number and insert
            async with db.execute(
                ("SELECT COALESCE(MAX(seq), 0) + 1 FROM chat_events "
                 "WHERE conversation_id = ?"),
                (conversation_id,)
            ) as cursor:
                row = await cursor.fetchone()
                assistant_event.seq = row[0] if row else 1

            row_data = self._serialize_event(assistant_event)
            columns = ", ".join(row_data.keys())
            placeholders = ", ".join("?" * len(row_data))

            await db.execute(
                f"INSERT INTO chat_events ({columns}) VALUES ({placeholders})",
                list(row_data.values())
            )
            await db.commit()

            return assistant_event

# ---------- In-Memory Repository ----------

class InMemoryRepo(ChatRepository):
    """Fast in-memory storage for session-only conversations."""
    
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

# ---------- Hybrid Repository ----------

class HybridRepo(ChatRepository):
    """Combines in-memory for current session with optional persistence."""
    
    def __init__(self, config: dict[str, Any]):
        self.memory_repo = InMemoryRepo()
        storage_config = config.get("chat", {}).get("storage", {})
        db_path = storage_config.get("persistence", {}).get("db_path", "chat_history.db")
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
        return await self.memory_repo.get_event_by_request_id(conversation_id, request_id)
    
    async def get_last_assistant_reply_id(
        self, conversation_id: str, user_request_id: str
    ) -> str | None:
        return await self.memory_repo.get_last_assistant_reply_id(conversation_id, user_request_id)
    
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

# ---------- Repository Factory ----------

def create_repository(config: dict[str, Any]) -> ChatRepository:
    """Factory function to create appropriate repository based on config."""
    storage_config = config.get("chat", {}).get("storage", {})
    mode = storage_config.get("mode", "session_only")
    
    if mode == "session_only":
        logger.info("Using in-memory storage (session-only)")
        return InMemoryRepo()
    
    if mode == "persistent":
        persistence_config = storage_config.get("persistence", {})
        db_path = persistence_config.get("db_path", "chat_history.db")
        logger.info(f"Using persistent storage: {db_path}")
        return SQLiteRepo(db_path)
    
    if mode == "hybrid":
        logger.info("Using hybrid storage (memory + optional persistence)")
        return HybridRepo(config)
    
    raise ValueError(f"Unknown storage mode: {mode}")

# ---------- Demo & Development Testing ----------

if __name__ == "__main__":
    import asyncio
    import logging
    import os
    logging.basicConfig(level=logging.INFO)
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
            extra={"request_id": "user_001"}
        )
        success = await repo.add_event(user_ev)
        logger.info(f"User message added: {success}")

        # Test assistant message with tool calls
        tool_call = ToolCall(
            id="call_123",
            name="get_weather",
            arguments={"location": "San Francisco"}
        )
        asst_ev = ChatEvent(
            conversation_id=conv_id,
            type="assistant_message",
            role="assistant",
            content="Let me check the weather for you.",
            tool_calls=[tool_call],
            usage=Usage(prompt_tokens=50, completion_tokens=25, total_tokens=75),
            provider="test_provider",
            model="test-model",
            extra={"request_id": "assistant_001", "user_request_id": "user_001"}
        )
        success = await repo.add_event(asst_ev)
        logger.info(f"Assistant message added: {success}")

        # Test tool result
        tool_result_ev = ChatEvent(
            conversation_id=conv_id,
            type="tool_result",
            content="The weather in San Francisco is 72°F and sunny.",
            extra={"tool_call_id": "call_123", "request_id": "tool_001"}
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
            visible = _visible_to_llm(ev)
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

    asyncio.run(main())
