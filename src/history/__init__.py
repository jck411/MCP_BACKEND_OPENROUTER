#!/usr/bin/env python3
"""
Chat History Module

Modular chat history system with multiple storage backends.
"""

from __future__ import annotations

from .auto_persist_repo import AutoPersistRepo
from .factory import create_repository
from .models import ChatEvent, ToolCall
from .repository import ChatRepository

__all__ = [
    "AutoPersistRepo",
    "ChatEvent",
    "ChatRepository",
    "ToolCall",
    "create_repository",
]
