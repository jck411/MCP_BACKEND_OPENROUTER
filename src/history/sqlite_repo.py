#!/usr/bin/env python3
"""
SQLite Chat Repository Implementation

High-performance SQLite storage for chat history with full concurrency support.

CONFIG: Not used directly - base class only
PURPOSE: Foundation for AutoPersistRepo (don't instantiate this directly)
FEATURES: SQLite ops, async support, JSON serialization, indexing
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import datetime
from typing import Any

import aiosqlite

from .models import ChatEvent, ToolCall
from .repository import ChatRepository

logger = logging.getLogger(__name__)


class SQLiteRepo(ChatRepository):
    """Base SQLite storage - don't use directly, use AutoPersistRepo instead."""

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
                event.content
                if isinstance(event.content, str)
                else json.dumps(event.content)
            ),
            "tool_calls": (
                json.dumps([tc.model_dump() for tc in event.tool_calls])
                if event.tool_calls
                else None
            ),
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
        if content and content.startswith("["):
            with contextlib.suppress(json.JSONDecodeError):
                content = json.loads(content)

        tool_calls = []
        if row["tool_calls"]:
            tool_calls = [ToolCall(**tc) for tc in json.loads(row["tool_calls"])]

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
                    (
                        "SELECT 1 FROM chat_events WHERE conversation_id = ? "
                        "AND request_id = ?"
                    ),
                    (event.conversation_id, request_id),
                ) as cursor:
                    if await cursor.fetchone():
                        return False  # Duplicate found

            # Get next sequence number
            async with db.execute(
                (
                    "SELECT COALESCE(MAX(seq), 0) + 1 FROM chat_events "
                    "WHERE conversation_id = ?"
                ),
                (event.conversation_id,),
            ) as cursor:
                row = await cursor.fetchone()
                event.seq = row[0] if row else 1

            # Insert event
            row_data = self._serialize_event(event)
            columns = ", ".join(row_data.keys())
            placeholders = ", ".join("?" * len(row_data))

            await db.execute(
                f"INSERT INTO chat_events ({columns}) VALUES ({placeholders})",
                list(row_data.values()),
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
                (
                    "SELECT * FROM chat_events WHERE conversation_id = ? "
                    "AND request_id = ?"
                ),
                (conversation_id, request_id),
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
                (conversation_id, user_request_id),
            ) as cursor,
        ):
            row = await cursor.fetchone()
            return row[0] if row else None

    async def compact_deltas(
        self,
        conversation_id: str,
        user_request_id: str,
        final_content: str,
        model: str | None = None,
    ) -> ChatEvent:
        """Compact delta events into a single assistant_message and remove deltas."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            # Check if assistant message already exists
            assistant_req_id = f"assistant:{user_request_id}"
            db.row_factory = aiosqlite.Row
            async with db.execute(
                (
                    "SELECT * FROM chat_events WHERE conversation_id = ? "
                    "AND request_id = ?"
                ),
                (conversation_id, assistant_req_id),
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
                (conversation_id, user_request_id),
            )

            # Create final assistant message
            assistant_event = ChatEvent(
                conversation_id=conversation_id,
                type="assistant_message",
                role="assistant",
                content=final_content,
                model=model,
                extra={
                    "user_request_id": user_request_id,
                    "request_id": assistant_req_id,
                },
            )

            # Get sequence number and insert
            async with db.execute(
                (
                    "SELECT COALESCE(MAX(seq), 0) + 1 FROM chat_events "
                    "WHERE conversation_id = ?"
                ),
                (conversation_id,),
            ) as cursor:
                row = await cursor.fetchone()
                assistant_event.seq = row[0] if row else 1

            row_data = self._serialize_event(assistant_event)
            columns = ", ".join(row_data.keys())
            placeholders = ", ".join("?" * len(row_data))

            await db.execute(
                f"INSERT INTO chat_events ({columns}) VALUES ({placeholders})",
                list(row_data.values()),
            )
            await db.commit()

            return assistant_event

    async def handle_clear_session(self) -> bool:
        """Clear all conversation data from SQLite database."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM chat_events")
            await db.commit()
            return True

    async def handle_user_message_persistence(
        self, conversation_id: str, user_msg: str, request_id: str
    ) -> bool:
        """Handle user message persistence with idempotency checks."""
        logger.debug(
            "→ Repository: checking for existing response for request_id=%s", request_id
        )

        # Check for existing response first
        existing_response = await self.get_existing_assistant_response(
            conversation_id, request_id
        )
        if existing_response:
            logger.info(
                "← Repository: cached response found for request_id=%s", request_id
            )
            return False

        # Persist user message
        logger.info(
            "→ Repository: persisting user message for request_id=%s", request_id
        )
        user_ev = ChatEvent(
            conversation_id=conversation_id,
            seq=0,  # Will be assigned by repository
            type="user_message",
            role="user",
            content=user_msg,
            extra={"request_id": request_id},
        )
        was_added = await self.add_event(user_ev)

        if not was_added:
            logger.debug(
                "→ Repository: duplicate message detected, re-checking for response"
            )
            # Check for existing response again after duplicate detection
            existing_response = await self.get_existing_assistant_response(
                conversation_id, request_id
            )
            if existing_response:
                logger.info(
                    "← Repository: existing response found after duplicate detection"
                )
                return False

        logger.info("← Repository: user message persisted successfully")
        return True

    async def get_existing_assistant_response(
        self, conversation_id: str, request_id: str
    ) -> ChatEvent | None:
        """Get existing assistant response for a request_id if it exists."""
        existing_assistant_id = await self.get_last_assistant_reply_id(
            conversation_id, request_id
        )
        if existing_assistant_id:
            events = await self.get_events(conversation_id)
            for event in events:
                if event.id == existing_assistant_id:
                    logger.debug(
                        "Found cached assistant response: event_id=%s", event.id
                    )
                    return event
        return None

    async def build_llm_conversation(
        self, conversation_id: str, user_msg: str, system_prompt: str
    ) -> list[dict[str, str]]:
        """Build conversation history in LLM format."""
        logger.debug(
            "→ Repository: fetching conversation history for conversation_id=%s",
            conversation_id,
        )

        events = await self.get_conversation_history(conversation_id, limit=50)
        logger.debug("← Repository: loaded %d conversation events", len(events))

        # Build conversation with system prompt and recent history
        conv: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        for event in events:
            if event.type == "user_message":
                conv.append({"role": "user", "content": str(event.content or "")})
            elif event.type == "assistant_message":
                conv.append({"role": "assistant", "content": str(event.content or "")})
            elif (
                event.type == "tool_result"
                and event.extra
                and "tool_call_id" in event.extra
            ):
                conv.append(
                    {
                        "role": "tool",
                        "tool_call_id": event.extra["tool_call_id"],
                        "content": str(event.content or ""),
                    }
                )

        # Add current user message
        conv.append({"role": "user", "content": user_msg})

        logger.debug(
            "Built conversation with %d messages (including system prompt)", len(conv)
        )
        return conv

    async def persist_assistant_message(
        self,
        conversation_id: str,
        request_id: str,
        content: str,
        model: str,
        provider: str = "unknown",
    ) -> ChatEvent:
        """Persist final assistant message to repository."""
        logger.info(
            "→ Repository: persisting assistant message for request_id=%s", request_id
        )

        assistant_ev = ChatEvent(
            conversation_id=conversation_id,
            seq=0,  # Will be assigned by repository
            type="assistant_message",
            role="assistant",
            content=content,
            provider=provider,
            model=model,
            extra={"user_request_id": request_id},
        )

        await self.add_event(assistant_ev)
        logger.info("← Repository: assistant message persisted successfully")

        return assistant_ev
