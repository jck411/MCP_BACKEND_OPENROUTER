"""
Chat Service Data Models

Stable data structures for chat functionality.
These models rarely change and form the core data layer.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.history import Usage


class ChatMessage(BaseModel):
    """
    Represents a chat message with metadata.
    Pydantic model for proper validation and serialization.
    """
    type: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallContext(BaseModel):
    """Parameters for tool call iteration handling."""
    conv: list[dict[str, Any]]
    tools_payload: list[dict[str, Any]]
    conversation_id: str
    request_id: str
    assistant_msg: dict[str, Any]
    full_content: str


def convert_usage(api_usage: dict[str, Any] | None) -> Usage:
    """
    Convert LLM API usage statistics to internal Usage model.

    This method normalizes usage data from different LLM API providers
    into the platform's standardized Usage Pydantic model. It handles cases
    where usage information may be missing or incomplete.

    The method provides safe defaults (0 tokens) for missing fields to ensure
    consistent usage tracking across all LLM interactions. This is important
    for cost monitoring and rate limiting.

    Args:
        api_usage: Raw usage dictionary from LLM API response, may be None
                  Expected fields: prompt_tokens, completion_tokens, total_tokens

    Returns:
        Usage: Pydantic model with normalized token counts, using 0 for
               missing fields

    Example:
        api_usage = {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15
        }
        usage = convert_usage(api_usage)
        # Returns: Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    """
    if not api_usage:
        return Usage()

    return Usage(
        prompt_tokens=api_usage.get("prompt_tokens", 0),
        completion_tokens=api_usage.get("completion_tokens", 0),
        total_tokens=api_usage.get("total_tokens", 0),
    )
