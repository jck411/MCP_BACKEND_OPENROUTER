#!/usr/bin/env python3
"""
Chat Repository Interface and Utilities

This module defines the repository protocol and utility functions for chat history.
"""

from __future__ import annotations

from typing import Protocol

from .models import ChatEvent

# ---------- Utility functions ----------

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


# ---------- Repository interface ----------


class ChatRepository(Protocol):
    """Protocol defining the interface for chat storage backends."""

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
        self,
        conversation_id: str,
        user_request_id: str,
        final_content: str,
        model: str | None = None,
    ) -> ChatEvent: ...

    async def handle_clear_session(self) -> bool: ...

    async def handle_user_message_persistence(
        self, conversation_id: str, user_msg: str, request_id: str
    ) -> bool:
        """
        Handle user message persistence with idempotency checks.

        Returns:
            True if processing should continue (new request), False if response
            already exists (cached response should be returned)
        """
        ...

    async def get_existing_assistant_response(
        self, conversation_id: str, request_id: str
    ) -> ChatEvent | None:
        """Get existing assistant response for a request_id if it exists."""
        ...

    async def build_llm_conversation(
        self, conversation_id: str, user_msg: str, system_prompt: str
    ) -> list[dict[str, str]]:
        """Build conversation history in LLM format."""
        ...

    async def persist_assistant_message(
        self,
        conversation_id: str,
        request_id: str,
        content: str,
        model: str,
        provider: str = "unknown",
    ) -> ChatEvent:
        """Persist final assistant message to repository."""
        ...
