#!/usr/bin/env python3
"""
Auto-Persist Chat Repository Implementation

Enhanced SQLite repo with automatic persistence and retention policies.

CONFIG: storage.type = "auto_persist" (default)
PURPOSE: Smart SQLite with auto-cleanup + manual saves
FEATURES: Background retention, session saving, intelligent cleanup
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import aiosqlite

from .models import ChatEvent
from .sqlite_repo import SQLiteRepo

logger = logging.getLogger(__name__)


class AutoPersistRepo(SQLiteRepo):
    """Enhanced SQLite repo with automatic persistence and retention policies."""

    def __init__(self, config: dict[str, Any]):
        storage_config = config.get("chat", {}).get("storage", {})
        persistence_config = storage_config.get("persistence", {})

        db_path = persistence_config.get("db_path", "chat_history.db")
        super().__init__(db_path)

        self.config = config
        self.retention_config = persistence_config.get("retention", {})
        self.saved_sessions_config = storage_config.get("saved_sessions", {})

        # Retention settings
        self.max_age_hours = self.retention_config.get("max_age_hours")
        self.max_messages = self.retention_config.get("max_messages")
        self.max_sessions = self.retention_config.get("max_sessions")
        self.cleanup_interval_minutes = self.retention_config.get("cleanup_interval_minutes", 5)

        # Clear trigger settings
        self.clear_triggers_before_full_wipe = self.retention_config.get(
            "clear_triggers_before_full_wipe", self.max_sessions
        )

        # Manual save settings
        self.saved_sessions_enabled = self.saved_sessions_config.get("enabled", True)
        self.saved_retention_days = self.saved_sessions_config.get("retention_days")
        self.max_saved = self.saved_sessions_config.get("max_saved", 50)

        # Cleanup task
        self._cleanup_task = None
        self._cleanup_running = False

        # Clear session counter for periodic full wipe
        self._clear_session_counter = 0

    async def _ensure_initialized(self):
        """Initialize database schema including saved sessions table."""
        await super()._ensure_initialized()

        if not self._initialized:
            return

        async with self._lock, aiosqlite.connect(self.db_path) as db:
            # Create saved sessions table
            await db.execute("""
                    CREATE TABLE IF NOT EXISTS saved_sessions (
                        id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        session_start TIMESTAMP,
                        session_end TIMESTAMP,
                        message_count INTEGER DEFAULT 0
                    )
                """)

            # Create indexes for saved sessions
            await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_saved_sessions_created
                    ON saved_sessions(created_at)
                """)

            await db.commit()

        # Start cleanup task if not already running
        if not self._cleanup_running:
            await self._start_cleanup_task()

    async def _start_cleanup_task(self):
        """Start the background cleanup task."""
        if self._cleanup_running:
            return

        self._cleanup_running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info(f"Started retention cleanup task (interval: {self.cleanup_interval_minutes} minutes)")

    async def _cleanup_loop(self):
        """Background task that runs retention cleanup."""
        while self._cleanup_running:
            try:
                await asyncio.sleep(
                    self.cleanup_interval_minutes * 60  # Convert to seconds
                )
                await self._run_retention_cleanup()
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                # Continue running even if cleanup fails

    async def _run_retention_cleanup(self):
        """Run all retention cleanup policies."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            cleanup_count = 0

            # Age-based cleanup (use epoch seconds for robust comparison)
            if self.max_age_hours is not None:
                await db.execute(
                    """
                    DELETE FROM chat_events
                    WHERE created_at_unix < CAST(strftime('%s','now') AS INTEGER) - ?
                    """,
                    (int(self.max_age_hours * 3600),),
                )
                result = await db.execute("SELECT changes()")
                row = await result.fetchone()
                age_deleted = row[0] if row else 0
                cleanup_count += age_deleted
                if age_deleted > 0:
                    logger.info(f"Cleaned up {age_deleted} messages older than {self.max_age_hours} hours")

            # Message count cleanup
            if self.max_messages is not None:
                # Count current messages
                async with db.execute("SELECT COUNT(*) FROM chat_events") as cursor:
                    row = await cursor.fetchone()
                    total_messages = row[0] if row else 0

                if total_messages > self.max_messages:
                    # Delete oldest messages beyond limit
                    excess = total_messages - self.max_messages
                    await db.execute(
                        """
                        DELETE FROM chat_events
                        WHERE id IN (
                            SELECT id FROM chat_events
                            ORDER BY created_at_unix ASC
                            LIMIT ?
                        )
                    """,
                        (excess,),
                    )
                    cleanup_count += excess
                    logger.info(f"Cleaned up {excess} oldest messages (limit: {self.max_messages})")

            # Session count cleanup
            if self.max_sessions is not None:
                # Count current sessions (conversations)
                async with db.execute("SELECT COUNT(DISTINCT conversation_id) FROM chat_events") as cursor:
                    row = await cursor.fetchone()
                    total_sessions = row[0] if row else 0

                if total_sessions > self.max_sessions:
                    # Get oldest conversations to delete
                    excess_sessions = total_sessions - self.max_sessions
                    async with db.execute(
                        """
                        SELECT conversation_id, MIN(created_at_unix) as first_message
                        FROM chat_events
                        GROUP BY conversation_id
                        ORDER BY first_message ASC
                        LIMIT ?
                    """,
                        (excess_sessions,),
                    ) as cursor:
                        old_conversations = await cursor.fetchall()

                    # Delete messages from oldest conversations
                    for conv_id, _ in old_conversations:
                        await db.execute(
                            "DELETE FROM chat_events WHERE conversation_id = ?",
                            (conv_id,),
                        )

                    cleanup_count += excess_sessions
                    logger.info(f"Cleaned up {excess_sessions} oldest sessions (limit: {self.max_sessions})")

            # Cleanup saved sessions if retention policy exists
            if self.saved_retention_days is not None:
                await db.execute(f"""
                    DELETE FROM saved_sessions
                    WHERE datetime(created_at) < datetime(
                        'now', '-{self.saved_retention_days} days'
                    )
                """)
                result = await db.execute("SELECT changes()")
                row = await result.fetchone()
                saved_deleted = row[0] if row else 0
                if saved_deleted > 0:
                    logger.info(
                        f"Cleaned up {saved_deleted} saved sessions older than {self.saved_retention_days} days"
                    )

            await db.commit()

            if cleanup_count > 0:
                logger.info(f"Total cleanup: {cleanup_count} items removed")

    async def save_session(self, conversation_id: str, name: str | None = None) -> str:
        """Manually save a session to preserved storage."""
        if not self.saved_sessions_enabled:
            raise RuntimeError("Saved sessions feature is disabled")

        await self._ensure_initialized()

        # Generate name if not provided
        if name is None:
            timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
            name = f"Session_{timestamp}"

        # Get session info
        async with aiosqlite.connect(self.db_path) as db:
            # Get session boundaries and message count
            async with db.execute(
                """
                SELECT MIN(created_at), MAX(created_at), COUNT(*)
                FROM chat_events
                WHERE conversation_id = ?
            """,
                (conversation_id,),
            ) as cursor:
                result = await cursor.fetchone()
                if not result or result[2] == 0:
                    raise ValueError(f"No messages found for conversation {conversation_id}")

                session_start, session_end, message_count = result

            # Check if we exceed max_saved limit
            if self.max_saved is not None:
                async with db.execute("SELECT COUNT(*) FROM saved_sessions") as cursor:
                    row = await cursor.fetchone()
                    current_saved = row[0] if row else 0

                if current_saved >= self.max_saved:
                    # Remove oldest saved session
                    await db.execute("""
                        DELETE FROM saved_sessions
                        WHERE id = (
                            SELECT id FROM saved_sessions
                            ORDER BY created_at ASC
                            LIMIT 1
                        )
                    """)
                    logger.info("Removed oldest saved session to make room for new one")

            # Save session metadata
            saved_id = str(uuid.uuid4())
            await db.execute(
                """
                INSERT INTO saved_sessions
                (id, conversation_id, name, session_start, session_end, message_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    saved_id,
                    conversation_id,
                    name,
                    session_start,
                    session_end,
                    message_count,
                ),
            )

            await db.commit()

            logger.info(f"Saved session '{name}' with {message_count} messages (ID: {saved_id})")
            return saved_id

    async def list_saved_sessions(self) -> list[dict[str, Any]]:
        """List all saved sessions with metadata."""
        if not self.saved_sessions_enabled:
            return []

        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT id, conversation_id, name, created_at,
                       session_start, session_end, message_count
                FROM saved_sessions
                ORDER BY created_at DESC
            """) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def load_saved_session(self, saved_id: str) -> list[ChatEvent]:
        """Load events from a saved session."""
        if not self.saved_sessions_enabled:
            raise RuntimeError("Saved sessions feature is disabled")

        await self._ensure_initialized()

        async with (
            aiosqlite.connect(self.db_path) as db,
            db.execute("SELECT conversation_id FROM saved_sessions WHERE id = ?", (saved_id,)) as cursor,
        ):
            result = await cursor.fetchone()
            if not result:
                raise ValueError(f"Saved session {saved_id} not found")

            conversation_id = result[0]

        # Load all events from that conversation
        return await self.get_events(conversation_id)

    async def handle_clear_session(self) -> bool:
        """
        Handle a clear session request with periodic full wipe logic.

        Returns:
            bool: True if a full history wipe occurred, False otherwise
        """
        self._clear_session_counter += 1
        logger.info(f"Clear session count: {self._clear_session_counter}")

        # Check if we should do a full wipe based on clear trigger count
        trigger_threshold = self.clear_triggers_before_full_wipe
        if trigger_threshold is not None and self._clear_session_counter >= trigger_threshold:
            await self._wipe_all_history_except_saved()
            self._clear_session_counter = 0  # Reset counter after wipe
            return True

        return False

    async def _wipe_all_history_except_saved(self):
        """Wipe all chat history except manually saved sessions."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            # Count messages before deletion
            async with db.execute("SELECT COUNT(*) FROM chat_events") as cursor:
                row = await cursor.fetchone()
                total_messages = row[0] if row else 0

            # Get list of saved conversation IDs to preserve
            saved_conversation_ids = []
            if self.saved_sessions_enabled:
                async with db.execute("SELECT DISTINCT conversation_id FROM saved_sessions") as cursor:
                    rows = await cursor.fetchall()
                    saved_conversation_ids = [row[0] for row in rows]

            if saved_conversation_ids:
                # Delete all messages except those in saved conversations
                placeholders = ", ".join("?" * len(saved_conversation_ids))
                await db.execute(
                    (f"DELETE FROM chat_events WHERE conversation_id NOT IN ({placeholders})"),
                    saved_conversation_ids,
                )
                logger.info(f"Wiped all history except {len(saved_conversation_ids)} saved conversations")
            else:
                # Delete all messages
                await db.execute("DELETE FROM chat_events")
                logger.info("Wiped all conversation history (no saved sessions to preserve)")

            # Count messages after deletion
            async with db.execute("SELECT COUNT(*) FROM chat_events") as cursor:
                row = await cursor.fetchone()
                remaining_messages = row[0] if row else 0

            await db.commit()

            deleted_count = total_messages - remaining_messages
            logger.info(
                f"Full history wipe completed: deleted {deleted_count} messages, {remaining_messages} preserved"
            )

    async def cleanup(self):
        """Cleanup resources including stopping the cleanup task."""
        self._cleanup_running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
            logger.info("Stopped retention cleanup task")
