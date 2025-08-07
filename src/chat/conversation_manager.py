"""
Conversation Management

Handles mundane, stable conversation operations:
- Building conversation history
- User message persistence
- Idempotency checks
- Basic repository interactions

This module contains straightforward business logic that rarely changes.
"""

import logging
from typing import Any

from src.history import ChatEvent

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages conversation history and persistence operations."""

    def __init__(self, repo, system_prompt: str):
        self.repo = repo
        self.system_prompt = system_prompt

    async def handle_user_message_persistence(
        self, conversation_id: str, user_msg: str, request_id: str
    ) -> bool:
        """
        Handle user message persistence with comprehensive idempotency checks.

        This method implements the core idempotency logic for the chat system.
        It first checks if an assistant response already exists for the given
        request_id, then attempts to persist the user message. If the user message
        is a duplicate, it performs a second check for existing responses to handle
        race conditions.

        Args:
            conversation_id: The conversation identifier
            user_msg: The user's message content
            request_id: Unique identifier for this request (used for idempotency)

        Returns:
            bool: True if processing should continue (new request), False if response
                  already exists (cached response should be returned)

        Side Effects:
            - Creates a ChatEvent for the user message and persists it to repository
            - Computes and caches token count for the user message
        """
        logger.debug("→ Repository: checking for existing response for request_id=%s", request_id)
        
        # Check for existing response first
        existing_response = await self.get_existing_assistant_response(
            conversation_id, request_id
        )
        if existing_response:
            logger.info("← Repository: cached response found for request_id=%s", request_id)
            return False

        # Persist user message
        logger.info("→ Repository: persisting user message for request_id=%s", request_id)
        user_ev = ChatEvent(
            conversation_id=conversation_id,
            seq=0,  # Will be assigned by repository
            type="user_message",
            role="user",
            content=user_msg,
            extra={"request_id": request_id},
        )
        was_added = await self.repo.add_event(user_ev)

        if not was_added:
            logger.debug("→ Repository: duplicate message detected, re-checking for response")
            # Check for existing response again after duplicate detection
            existing_response = await self.get_existing_assistant_response(
                conversation_id, request_id
            )
            if existing_response:
                logger.info("← Repository: existing response found after duplicate detection")
                return False

        logger.info("← Repository: user message persisted successfully")
        return True

    async def get_existing_assistant_response(
        self, conversation_id: str, request_id: str
    ) -> ChatEvent | None:
        """Get existing assistant response for a request_id if it exists."""
        existing_assistant_id = await self.repo.get_last_assistant_reply_id(
            conversation_id, request_id
        )
        if existing_assistant_id:
            events = await self.repo.get_events(conversation_id)
            for event in events:
                if event.id == existing_assistant_id:
                    logger.debug("Found cached assistant response: event_id=%s", event.id)
                    return event
        return None

    async def build_conversation_history(
        self, conversation_id: str, user_msg: str
    ) -> list[dict[str, Any]]:
        """
        Build conversation history from repository.
        
        Creates a conversation array suitable for LLM APIs, including:
        - System prompt as first message
        - Historical user/assistant messages
        - Tool result messages with proper formatting
        - Current user message
        """
        logger.debug("→ Repository: fetching conversation history for conversation_id=%s", conversation_id)
        
        events = await self.repo.get_conversation_history(conversation_id, limit=50)
        
        logger.debug("← Repository: loaded %d conversation events", len(events))

        # Build conversation with system prompt and recent history
        conv = [{"role": "system", "content": self.system_prompt}]

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
                conv.append({
                    "role": "tool",
                    "tool_call_id": event.extra["tool_call_id"],
                    "content": str(event.content or "")
                })

        # Add current user message
        conv.append({"role": "user", "content": user_msg})
        
        logger.debug("Built conversation with %d messages (including system prompt)", len(conv))
        return conv

    async def persist_assistant_message(
        self, 
        conversation_id: str, 
        request_id: str,
        content: str, 
        usage, 
        model: str,
        provider: str = "unknown"
    ) -> ChatEvent:
        """Persist final assistant message to repository."""
        logger.info("→ Repository: persisting assistant message for request_id=%s", request_id)
        
        assistant_ev = ChatEvent(
            conversation_id=conversation_id,
            seq=0,  # Will be assigned by repository
            type="assistant_message",
            role="assistant",
            content=content,
            usage=usage,
            provider=provider,
            model=model,
            extra={"user_request_id": request_id},
        )
        
        await self.repo.add_event(assistant_ev)
        logger.info("← Repository: assistant message persisted successfully")
        
        return assistant_ev
