#!/usr/bin/env python3
"""
Repository Factory

Factory function to create appropriate repository based on configuration.
"""
from __future__ import annotations

import logging
from typing import Any

from .auto_persist_repo import AutoPersistRepo
from .memory_repo import InMemoryRepo
from .repository import ChatRepository

logger = logging.getLogger(__name__)


def create_repository(config: dict[str, Any]) -> ChatRepository:
    """Factory function to create appropriate repository based on config."""
    storage_config = config.get("chat", {}).get("storage", {})
    repo_type = storage_config.get("type", "auto_persist")

    if repo_type == "memory":
        logger.info("Using in-memory storage (ephemeral)")
        return InMemoryRepo()
    if repo_type == "auto_persist":
        logger.info("Using auto-persist storage with retention policies")
        return AutoPersistRepo(config)

    logger.warning(f"Unknown repository type '{repo_type}', defaulting to auto_persist")
    return AutoPersistRepo(config)
