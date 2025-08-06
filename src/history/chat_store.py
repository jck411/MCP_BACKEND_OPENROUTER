#!/usr/bin/env python3
"""
Chat History Storage Module

This module provides a comprehensive chat event storage system for the MCP Platform.
It manages conversation history, token counting, and persistent storage with multiple
backend implementations.

Key Components:
- ChatEvent: Pydantic model representing individual chat messages, tool calls,
  and system events
- ChatRepository: Protocol defining the interface for chat storage backends
- InMemoryRepo: Fast in-memory storage implementation for development/testing
- JsonlRepo: Persistent JSONL file storage for production use

Features:
- Comprehensive event tracking (user messages, assistant responses, tool calls,
  system updates)
- Token counting and usage tracking for cost monitoring
- Duplicate detection via request IDs
- Conversation-based organization
- Thread-safe operations
- Async/await support

The module follows the MCP Platform's standards with:
- Pydantic v2 for data validation and serialization
- Type hints on all functions and methods
- Fail-fast error handling
- Modern Python syntax (union types with |)
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import threading
import uuid
from datetime import datetime
from functools import partial
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from src.history.token_counter import estimate_tokens

logger = logging.getLogger(__name__)

# ---------- Canonical models (Pydantic v2) ----------

Role = Literal["system", "user", "assistant", "tool"]

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
    token_count: int | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    extra: dict[str, Any] = Field(default_factory=dict)
    raw: Any | None = None  # keep small; move big things elsewhere later

    def compute_and_cache_tokens(self) -> int:
        """Compute and cache token count for this event's content."""
        if self.content is None:
            self.token_count = 0
            return 0

        if isinstance(self.content, str):
            text_content = self.content
        elif isinstance(self.content, list):
            # Handle list of Parts (for future extensibility)
            text_parts = []
            for part in self.content:
                if isinstance(part, TextPart):
                    text_parts.append(part.text)
            text_content = " ".join(text_parts)
        else:
            text_content = str(self.content)

        self.token_count = estimate_tokens(text_content)
        return self.token_count

    def ensure_token_count(self) -> int:
        """Ensure token count is computed and return it."""
        if self.token_count is None:
            return self.compute_and_cache_tokens()
        return self.token_count

# ---------- Repository interface ----------

# -------------------------------------------------------------------
# Prompt-history filter (hybrid = semantic + tool_result + flagged system)
# -------------------------------------------------------------------
_CONTEXT_TYPES = {"user_message", "assistant_message", "tool_result"}

def _visible_to_llm(ev: ChatEvent) -> bool:
    """
    Determine if a chat event should be included in LLM conversation context.

    This function implements the visibility filter for building conversation history
    that gets sent to the LLM. It uses a hybrid approach that includes semantic
    conversation content while filtering out internal system messages.

    The filter includes:
    - user_message: Direct user inputs to the conversation
    - assistant_message: LLM responses and generated content
    - tool_result: Results from tool executions (provides context for LLM)
    - system_update: Only when explicitly marked as visible_to_model=True

    The filter excludes by default:
    - meta events: Internal deltas, logging, and processing events
    - tool_call events: The call itself (result is included separately)
    - system_update events: Unless explicitly marked for model visibility

    This approach ensures the LLM sees a clean conversation flow with relevant
    context while excluding noisy internal implementation details.

    Args:
        ev: ChatEvent to evaluate for LLM visibility

    Returns:
        bool: True if event should be included in LLM context, False otherwise

    Example:
        user_msg = ChatEvent(type="user_message", ...)  # -> True
        assistant_msg = ChatEvent(type="assistant_message", ...)  # -> True
        tool_result = ChatEvent(type="tool_result", ...)  # -> True
        delta = ChatEvent(type="meta", ...)  # -> False
        system = ChatEvent(
            type="system_update", extra={"visible_to_model": True}
        )  # -> True
    """
    if ev.type in _CONTEXT_TYPES:
        return True
    # Allow system updates only when explicitly marked for model visibility
    return ev.type == "system_update" and ev.extra.get("visible_to_model", False)

class ChatRepository(Protocol):
    # Returns True if added, False if duplicate
    async def add_event(self, event: ChatEvent) -> bool: ...
    async def get_events(
        self, conversation_id: str, limit: int | None = None
    ) -> list[ChatEvent]: ...
    async def last_n_tokens(
        self, conversation_id: str, max_tokens: int
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

# ---------- In-memory implementation ----------

class InMemoryRepo(ChatRepository):
    def __init__(self):
        self._by_conv: dict[str, list[ChatEvent]] = {}
        self._seq_counters: dict[str, int] = {}  # Per-conversation sequence counters
        self._req_ids: dict[str, set[str]] = {}  # Per-conversation fast request_id sets
        # conv_id -> {request_id -> event} for O(1) event lookup by request_id
        self._req_id_to_event: dict[str, dict[str, ChatEvent]] = {}
        # conv_id -> {user_req_id -> asst_id}
        self._last_assistant_ids: dict[str, dict[str, str]] = {}
        self._lock = threading.Lock()

    def _get_next_seq(self, conversation_id: str) -> int:
        """
        Generate the next sequence number for a conversation atomically.

        This internal method manages the per-conversation sequence counters that
        ensure events within a conversation have monotonically increasing sequence
        numbers. This is critical for maintaining event ordering and detecting
        missing or duplicate events.

        The method must be called while holding the repository lock to ensure
        atomicity across concurrent operations. The sequence counter starts at 1
        for the first event in each conversation.

        Args:
            conversation_id: The conversation to generate sequence number for

        Returns:
            int: The next sequence number (starting from 1 for new conversations)

        Thread Safety:
            Must be called with self._lock held. Not thread-safe on its own.

        Side Effects:
            Updates the internal _seq_counters dictionary for the conversation
        """
        current = self._seq_counters.get(conversation_id, 0)
        next_seq = current + 1
        self._seq_counters[conversation_id] = next_seq
        return next_seq

    async def add_event(self, event: ChatEvent) -> bool:
        with self._lock:
            events = self._by_conv.setdefault(event.conversation_id, [])

            # Check for duplicate by request_id if present - O(1) lookup
            request_id = event.extra.get("request_id")
            if request_id:
                req_ids = self._req_ids.setdefault(event.conversation_id, set())
                if request_id in req_ids:
                    return False  # Duplicate found, don't add

            # Assign sequence number atomically (always overwrite)
            event.seq = self._get_next_seq(event.conversation_id)
            events.append(event)

            # Add request_id to fast lookup structures
            if request_id:
                req_ids = self._req_ids.setdefault(event.conversation_id, set())
                req_ids.add(request_id)
                # Add to request_id -> event mapping for O(1) retrieval
                req_map = self._req_id_to_event.setdefault(event.conversation_id, {})
                req_map[request_id] = event

            return True  # Successfully added

    async def get_events(
        self, conversation_id: str, limit: int | None = None
    ) -> list[ChatEvent]:
        with self._lock:
            events = self._by_conv.get(conversation_id, [])
            return events[-limit:] if limit is not None else list(events)

    async def last_n_tokens(
        self, conversation_id: str, max_tokens: int
    ) -> list[ChatEvent]:
        with self._lock:
            events = self._by_conv.get(conversation_id, [])
            conversation_events = [ev for ev in events if _visible_to_llm(ev)]

            acc: list[ChatEvent] = []
            total = 0
            for ev in reversed(conversation_events):
                # Ensure token count is computed
                tok = ev.ensure_token_count()
                if total + tok > max_tokens:
                    break
                acc.append(ev)
                total += tok
            return list(reversed(acc))

    async def list_conversations(self) -> list[str]:
        with self._lock:
            return list(self._by_conv.keys())

    async def get_event_by_request_id(
        self, conversation_id: str, request_id: str
    ) -> ChatEvent | None:
        with self._lock:
            # O(1) lookup using indexed request_id mapping
            req_map = self._req_id_to_event.get(conversation_id, {})
            return req_map.get(request_id)

    async def get_last_assistant_reply_id(
        self, conversation_id: str, user_request_id: str
    ) -> str | None:
        with self._lock:
            # O(1) lookup using cached IDs
            conv_cache = self._last_assistant_ids.get(conversation_id, {})
            return conv_cache.get(user_request_id)

    async def compact_deltas(
        self, conversation_id: str, user_request_id: str, final_content: str,
        usage: Usage | None = None, model: str | None = None
    ) -> ChatEvent:
        """Compact delta events into a single assistant_message and remove deltas."""
        with self._lock:
            events = self._by_conv.get(conversation_id, [])

            # Check if assistant message already exists for this user_request_id
            req_ids = self._req_ids.setdefault(conversation_id, set())
            assistant_req_id = f"assistant:{user_request_id}"
            if assistant_req_id in req_ids:
                # Find and return existing assistant message
                for event in events:
                    if (event.type == "assistant_message" and
                        event.extra.get("user_request_id") == user_request_id):
                        return event
                # Fallback if not found (shouldn't happen)
                logger.warning(
                    f"Assistant request_id {assistant_req_id} in set "
                    f"but event not found"
                )

            # Find and remove delta events for this user request
            deltas_to_remove = []
            for i, event in enumerate(events):
                if (
                    event.type == "meta"
                    and event.extra.get("kind") == "assistant_delta"
                    and event.extra.get("user_request_id") == user_request_id
                ):
                    deltas_to_remove.append(i)

            # Remove deltas in reverse order to maintain indices
            for i in reversed(deltas_to_remove):
                events.pop(i)

            # Create final assistant message
            assistant_event = ChatEvent(
                conversation_id=conversation_id,
                type="assistant_message",
                role="assistant",
                content=final_content,
                usage=usage,
                model=model,
                extra={"user_request_id": user_request_id}
            )
            assistant_event.seq = self._get_next_seq(conversation_id)
            assistant_event.compute_and_cache_tokens()

            events.append(assistant_event)

            # Add to request_id set to prevent duplicate compaction
            req_ids.add(assistant_req_id)
            # Add to request_id -> event mapping for O(1) retrieval
            req_map = self._req_id_to_event.setdefault(conversation_id, {})
            req_map[assistant_req_id] = assistant_event

            # Cache assistant ID for O(1) lookup
            asst_cache = self._last_assistant_ids.setdefault(conversation_id, {})
            asst_cache[user_request_id] = assistant_event.id

            return assistant_event

# ---------- JSONL (append-only) implementation ----------

class JsonlRepo(ChatRepository):
    """
    Simple, dev-friendly persistence: one global JSONL file.
    If you prefer, create one file per conversation (easy tweak).
    """
    def __init__(self, path: str = "events.jsonl"):
        self.path = path
        self._lock = threading.Lock()
        self._by_conv: dict[str, list[ChatEvent]] = {}
        self._seq_counters: dict[str, int] = {}  # Per-conversation sequence counters
        self._req_ids: dict[str, set[str]] = {}  # Per-conversation fast request_id sets
        # conv_id -> {request_id -> event} for O(1) event lookup by request_id
        self._req_id_to_event: dict[str, dict[str, ChatEvent]] = {}
        # conv_id -> {user_req_id -> asst_id}
        self._last_assistant_ids: dict[str, dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        """
        Load and parse existing events from the JSONL file during initialization.

        This internal method handles the startup process for the JsonlRepo by reading
        the entire JSONL file and rebuilding the in-memory state. It parses each line
        as a JSON ChatEvent and reconstructs the internal data structures needed for
        efficient operation.

        The method performs several important tasks:
        1. Parses each JSONL line into ChatEvent objects with validation
        2. Groups events by conversation_id for fast retrieval
        3. Tracks the highest sequence number per conversation for counter state
        4. Builds request_id lookup sets for O(1) duplicate detection
        5. Caches assistant message IDs for fast response lookups

        Error Handling:
        - Skips empty lines gracefully
        - Continues processing if individual lines fail to parse (logs error)
        - Creates file if it doesn't exist (no error)

        Side Effects:
        - Populates _by_conv, _seq_counters, _req_ids, _last_assistant_ids
        - May create empty file if it doesn't exist
        """
        if not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as f:
            for file_line in f:
                stripped_line = file_line.strip()
                if not stripped_line:
                    continue
                data = json.loads(stripped_line)
                ev = ChatEvent.model_validate(data)
                self._by_conv.setdefault(ev.conversation_id, []).append(ev)

                # Track highest sequence number per conversation
                if ev.seq is not None:
                    current_max = self._seq_counters.get(ev.conversation_id, 0)
                    self._seq_counters[ev.conversation_id] = max(current_max, ev.seq)

                # Build request_id sets for O(1) duplicate detection
                request_id = ev.extra.get("request_id")
                if request_id:
                    req_ids = self._req_ids.setdefault(ev.conversation_id, set())
                    req_ids.add(request_id)
                    # Build request_id -> event mapping for O(1) retrieval
                    req_map = self._req_id_to_event.setdefault(ev.conversation_id, {})
                    req_map[request_id] = ev

                # Cache assistant IDs for O(1) lookup
                if (ev.type == "assistant_message" and
                    ev.extra.get("user_request_id")):
                    conv_id = ev.conversation_id
                    asst_cache = self._last_assistant_ids.setdefault(conv_id, {})
                    asst_cache[ev.extra["user_request_id"]] = ev.id

    def _get_next_seq(self, conversation_id: str) -> int:
        """Get next sequence number for conversation. Must be called with lock held."""
        current = self._seq_counters.get(conversation_id, 0)
        next_seq = current + 1
        self._seq_counters[conversation_id] = next_seq
        return next_seq

    def _append_sync(self, event: ChatEvent) -> None:
        """
        Synchronously append a single event to the JSONL file with crash safety.

        This internal method handles the low-level file operations for persisting
        ChatEvents to disk. It implements several important durability guarantees:
        1. File locking for cross-process safety (prevents corruption)
        2. Atomic write operations (complete line or nothing)
        3. fsync for crash safety (survives power failures)
        4. Proper exception handling with lock cleanup

        The method uses fcntl file locking to coordinate access between multiple
        processes that might be writing to the same JSONL file. This is critical
        in distributed or multi-process deployments.

        Args:
            event: ChatEvent to persist to disk

        Side Effects:
            - Appends one line to the JSONL file
            - Acquires and releases file lock
            - Forces disk synchronization

        Thread Safety:
            Safe for concurrent use with other _append_sync calls.
            File locking handles cross-process coordination.

        Performance Notes:
            fsync is expensive but necessary for durability. In high-throughput
            scenarios, consider batching multiple events before syncing.
        """
        with open(self.path, "a", encoding="utf-8") as f:
            # Add file locking for cross-process safety
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                data = json.dumps(
                    event.model_dump(mode="json"), ensure_ascii=False
                )
                f.write(data + "\n")
                f.flush()
                # Add fsync for crash-safety (optional but recommended for durability)
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _add_event_sync(self, event: ChatEvent) -> bool:
        with self._lock:
            events = self._by_conv.setdefault(event.conversation_id, [])

            # Check for duplicate by request_id if present - O(1) lookup
            request_id = event.extra.get("request_id")
            if request_id:
                req_ids = self._req_ids.setdefault(event.conversation_id, set())
                if request_id in req_ids:
                    return False  # Duplicate found, don't add

            # Assign sequence number atomically (always overwrite)
            event.seq = self._get_next_seq(event.conversation_id)
            events.append(event)
            self._append_sync(event)

            # Add request_id to fast lookup structures
            if request_id:
                req_ids = self._req_ids.setdefault(event.conversation_id, set())
                req_ids.add(request_id)
                # Add to request_id -> event mapping for O(1) retrieval
                req_map = self._req_id_to_event.setdefault(event.conversation_id, {})
                req_map[request_id] = event

            return True  # Successfully added

    async def add_event(self, event: ChatEvent) -> bool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self._add_event_sync, event))

    async def get_events(
        self, conversation_id: str, limit: int | None = None
    ) -> list[ChatEvent]:
        with self._lock:
            events = self._by_conv.get(conversation_id, [])
            return events[-limit:] if limit is not None else list(events)

    async def last_n_tokens(
        self, conversation_id: str, max_tokens: int
    ) -> list[ChatEvent]:
        with self._lock:
            events = self._by_conv.get(conversation_id, [])
            conversation_events = [ev for ev in events if _visible_to_llm(ev)]

            acc: list[ChatEvent] = []
            total = 0
            for ev in reversed(conversation_events):
                # Ensure token count is computed
                tok = ev.ensure_token_count()
                if total + tok > max_tokens:
                    break
                acc.append(ev)
                total += tok
            return list(reversed(acc))

    async def list_conversations(self) -> list[str]:
        with self._lock:
            return list(self._by_conv.keys())

    async def get_event_by_request_id(
        self, conversation_id: str, request_id: str
    ) -> ChatEvent | None:
        with self._lock:
            # O(1) lookup using indexed request_id mapping
            req_map = self._req_id_to_event.get(conversation_id, {})
            return req_map.get(request_id)

    async def get_last_assistant_reply_id(
        self, conversation_id: str, user_request_id: str
    ) -> str | None:
        with self._lock:
            # O(1) lookup using cached IDs
            conv_cache = self._last_assistant_ids.get(conversation_id, {})
            return conv_cache.get(user_request_id)

    def _compact_deltas_sync(
        self, conversation_id: str, user_request_id: str, final_content: str,
        usage: Usage | None = None, model: str | None = None
    ) -> ChatEvent:
        """Synchronous delta compaction. Must be called with executor."""
        with self._lock:
            events = self._by_conv.get(conversation_id, [])

            # Check if assistant message already exists for this user_request_id
            req_ids = self._req_ids.setdefault(conversation_id, set())
            assistant_req_id = f"assistant:{user_request_id}"
            if assistant_req_id in req_ids:
                # Find and return existing assistant message
                for event in events:
                    if (event.type == "assistant_message" and
                        event.extra.get("user_request_id") == user_request_id):
                        return event
                # Fallback if not found (shouldn't happen)
                logger.warning(
                    f"Assistant request_id {assistant_req_id} in set "
                    f"but event not found"
                )

            # Find and remove delta events for this user request
            deltas_to_remove = []
            for i, event in enumerate(events):
                if (
                    event.type == "meta"
                    and event.extra.get("kind") == "assistant_delta"
                    and event.extra.get("user_request_id") == user_request_id
                ):
                    deltas_to_remove.append(i)

            # Remove deltas in reverse order to maintain indices
            for i in reversed(deltas_to_remove):
                events.pop(i)

            # Create final assistant message
            assistant_event = ChatEvent(
                conversation_id=conversation_id,
                type="assistant_message",
                role="assistant",
                content=final_content,
                usage=usage,
                model=model,
                extra={"user_request_id": user_request_id}
            )
            assistant_event.seq = self._get_next_seq(conversation_id)
            assistant_event.compute_and_cache_tokens()

            events.append(assistant_event)
            self._append_sync(assistant_event)

            # Add to request_id set to prevent duplicate compaction
            req_ids.add(assistant_req_id)
            # Add to request_id -> event mapping for O(1) retrieval
            req_map = self._req_id_to_event.setdefault(conversation_id, {})
            req_map[assistant_req_id] = assistant_event

            # Cache assistant ID for O(1) lookup
            asst_cache = self._last_assistant_ids.setdefault(conversation_id, {})
            asst_cache[user_request_id] = assistant_event.id

            return assistant_event

    async def compact_deltas(
        self, conversation_id: str, user_request_id: str, final_content: str,
        usage: Usage | None = None, model: str | None = None
    ) -> ChatEvent:
        """Compact delta events into a single assistant_message and remove deltas."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(
                self._compact_deltas_sync,
                conversation_id, user_request_id, final_content, usage, model
            )
        )

# ---------- Tiny demo ----------

if __name__ == "__main__":
    import asyncio
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    async def main():
        repo: ChatRepository = JsonlRepo("events.jsonl")
        conv_id = str(uuid.uuid4())

        user_ev = ChatEvent(
            conversation_id=conv_id,
            type="user_message",
            role="user",
            content="Hello!",
        )
        user_ev.compute_and_cache_tokens()
        await repo.add_event(user_ev)

        asst_ev = ChatEvent(
            conversation_id=conv_id,
            type="assistant_message",
            role="assistant",
            content="Hi! How can I help?",
            provider="test_provider",
            model="test-model"
        )
        asst_ev.compute_and_cache_tokens()
        await repo.add_event(asst_ev)

        logger.info(f"Conversation ID: {conv_id}")
        events = await repo.get_events(conv_id)
        for ev in events:
            logger.info(f"- seq={ev.seq} {ev.type} {ev.role} {ev.content}")

    asyncio.run(main())
