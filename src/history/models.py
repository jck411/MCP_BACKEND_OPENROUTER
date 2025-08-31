#!/usr/bin/env python3
"""
Chat History Data Models

This module contains all Pydantic models for the chat history system.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------- Type definitions ----------

Role = Literal["system", "user", "assistant", "tool"]


# ---------- Content models ----------


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


Part = TextPart  # extend later


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


# ---------- Main event model ----------


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
        "meta",
    ]
    role: Role | None = None
    content: str | list[Part] | None = None
    tool_calls: list[ToolCall] = []
    provider: str | None = None
    model: str | None = None
    stop_reason: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    extra: dict[str, Any] = Field(default_factory=dict)
    raw: Any | None = None  # keep small; move big things elsewhere later
