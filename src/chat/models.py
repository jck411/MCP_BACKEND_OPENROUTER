"""
Chat Service Data Models

Stable data structures for chat functionality.
These models rarely change and form the core data layer.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


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
