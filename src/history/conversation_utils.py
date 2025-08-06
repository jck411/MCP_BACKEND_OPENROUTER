"""
Conversation utilities for token-aware context management.

This module provides utilities for building conversations with proper
token accounting and context window management.
"""

from __future__ import annotations

import logging
from typing import Any

from src.history.chat_store import ChatEvent
from src.history.token_counter import count_conversation_tokens

logger = logging.getLogger(__name__)

# Constants for response size estimation
SHORT_CONVERSATION_LIMIT = 100
MEDIUM_CONVERSATION_LIMIT = 500
LONG_CONVERSATION_LIMIT = 1500

SHORT_RESPONSE_TOKENS = 150
MEDIUM_RESPONSE_TOKENS = 300
LONG_RESPONSE_TOKENS = 500
MAX_RESPONSE_TOKENS = 800


def build_conversation_with_token_limit(
    system_prompt: str,
    events: list[ChatEvent],
    user_message: str,
    max_tokens: int,
    reserve_tokens: int = 500,
) -> tuple[list[dict[str, Any]], int]:
    """
    Build a conversation list with token limit enforcement.

    This function builds an OpenAI-style conversation while respecting
    token limits and reserving space for the response.

    Args:
        system_prompt: The system prompt to include
        events: List of chat events to include
        user_message: The current user message
        max_tokens: Maximum tokens for the conversation
        reserve_tokens: Tokens to reserve for the response

    Returns:
        Tuple of (conversation_list, total_tokens_used)
    """
    # Start with minimal conversation: system + current user message
    # This establishes the baseline token cost that we cannot reduce
    base_conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    # Calculate the immutable token cost (system prompt + user message)
    base_tokens = count_conversation_tokens(base_conversation)

    # Calculate how many tokens we have available for historical events
    # This is our "budget" for including conversation history
    available_tokens = max_tokens - base_tokens - reserve_tokens

    # Early exit if system prompt + user message already exceed limits
    # This prevents infinite loops and provides clear error messaging
    if available_tokens <= 0:
        logger.warning(
            f"System prompt and user message exceed token limit. "
            f"Base: {base_tokens}, Max: {max_tokens}, Reserve: {reserve_tokens}"
        )
        return base_conversation, base_tokens

    # Build conversation incrementally, starting with system prompt
    # We'll add historical events between system and user message
    conversation = [{"role": "system", "content": system_prompt}]

    # Process events in chronological order to maintain conversation flow
    # Each event is tested before inclusion to ensure we don't exceed limits
    for event in events:
        # Only include events that make sense in LLM conversation context
        if event.type in ("user_message", "assistant_message", "system_update"):
            event_msg = {"role": event.role, "content": event.content or ""}

            # Create test conversation with this event included
            # This allows us to calculate exact token impact
            test_conversation = [*conversation, event_msg]

            # Calculate final token count if we include this event + user message
            # The user message must always be last, so we simulate the final state
            final_tokens_with_user = count_conversation_tokens(
                [*test_conversation, {"role": "user", "content": user_message}]
            )

            # Check if including this event would breach our token budget
            # We need to account for both the final conversation AND reserved tokens
            if final_tokens_with_user + reserve_tokens > max_tokens:
                logger.debug(
                    f"Stopping conversation build at {len(conversation)} messages. "
                    f"Would use {final_tokens_with_user} + {reserve_tokens} tokens."
                )
                break  # Stop adding events - we've hit our limit

            # Safe to include this event - add it to the conversation
            conversation.append(event_msg)

    # Always append the current user message last (required for OpenAI format)
    conversation.append({"role": "user", "content": user_message})

    # Calculate and return final token count for monitoring/logging
    final_tokens = count_conversation_tokens(conversation)

    return conversation, final_tokens


def estimate_response_tokens(conversation: list[dict[str, Any]]) -> int:
    """
    Estimate how many tokens a response might need based on conversation.

    This is a heuristic based on typical response patterns.
    """
    conversation_tokens = count_conversation_tokens(conversation)

    # Heuristics for response size
    if conversation_tokens < SHORT_CONVERSATION_LIMIT:
        return SHORT_RESPONSE_TOKENS  # Short responses for short conversations
    if conversation_tokens < MEDIUM_CONVERSATION_LIMIT:
        return MEDIUM_RESPONSE_TOKENS  # Medium responses
    if conversation_tokens < LONG_CONVERSATION_LIMIT:
        return LONG_RESPONSE_TOKENS  # Longer responses for complex conversations
    return MAX_RESPONSE_TOKENS  # Max response for very long conversations


def get_conversation_token_distribution(events: list[ChatEvent]) -> dict[str, int]:
    """
    Analyze token distribution across different event types.

    Returns a dictionary with token counts by event type.
    """
    distribution: dict[str, int] = {}

    for event in events:
        tokens = event.ensure_token_count()
        event_type = event.type
        distribution[event_type] = distribution.get(event_type, 0) + tokens

    return distribution


def optimize_conversation_for_tokens(
    events: list[ChatEvent],
    target_tokens: int,
    preserve_recent: int = 5,
) -> list[ChatEvent]:
    """
    Optimize a conversation to fit within a token budget.

    This function intelligently removes older messages while preserving
    important context and recent messages.

    Args:
        events: List of chat events
        target_tokens: Target token count
        preserve_recent: Number of recent messages to always preserve

    Returns:
        Optimized list of chat events
    """
    if not events:
        return events

    # Always preserve recent messages
    recent_events = events[-preserve_recent:] if preserve_recent > 0 else []
    older_events = events[:-preserve_recent] if preserve_recent > 0 else events

    # Calculate tokens for recent events
    recent_tokens = sum(event.ensure_token_count() for event in recent_events)

    if recent_tokens >= target_tokens:
        logger.warning(
            f"Recent {preserve_recent} messages already use {recent_tokens} tokens, "
            f"exceeding target of {target_tokens}"
        )
        return recent_events

    # Add older events until we hit the token limit
    remaining_tokens = target_tokens - recent_tokens
    selected_older = []

    for event in reversed(older_events):
        event_tokens = event.ensure_token_count()
        if remaining_tokens >= event_tokens:
            selected_older.insert(0, event)  # Insert at beginning to maintain order
            remaining_tokens -= event_tokens
        else:
            break

    return selected_older + recent_events
