#!/usr/bin/env python3
"""
In-Memory Chat Repository Implementation

Fast in-memory storage for session-only conversations.

CONFIG: storage.type = "memory"
PURPOSE: Development/testing - all data lost on restart
FEATURES: Fastest performance, no persistence, simple cleanup
"""

from __future__ import annotations

import logging
from typing import Any

from .models import ChatEvent
from .repository import ChatRepository

logger = logging.getLogger(__name__)


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
        """Get conversation history visible to LLM."""
        events = await self.get_events(conversation_id, limit=limit)
        # Filter for events visible to LLM
        _CONTEXT_TYPES = {"user_message", "assistant_message", "tool_result"}

        def _visible_to_llm(ev: ChatEvent) -> bool:
            if ev.type in _CONTEXT_TYPES:
                return True
            return ev.type == "system_update" and ev.extra.get(
                "visible_to_model", False
            )

        return [ev for ev in events if _visible_to_llm(ev)]

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
            if (
                event.type == "assistant_message"
                and event.extra.get("user_request_id") == user_request_id
            ):
                return event.id
        return None

    async def compact_deltas(
        self,
        conversation_id: str,
        user_request_id: str,
        final_content: str,
        model: str | None = None,
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
            ev
            for ev in events
            if not (
                ev.type == "meta"
                and ev.extra.get("kind") == "assistant_delta"
                and ev.extra.get("user_request_id") == user_request_id
            )
        ]

        # Create final assistant message
        assistant_event = ChatEvent(
            conversation_id=conversation_id,
            type="assistant_message",
            role="assistant",
            content=final_content,
            model=model,
            extra={"user_request_id": user_request_id, "request_id": assistant_req_id},
        )

        await self.add_event(assistant_event)
        return assistant_event

    async def handle_clear_session(self) -> bool:
        """Clear all conversation data from memory."""
        self._conversations.clear()
        self._seq_counters.clear()
        self._request_cache.clear()
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
