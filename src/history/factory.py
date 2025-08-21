#!/usr/bin/env python3
"""
Repository Factory

Factory function to create appropriate repository based on configuration.
"""

from __future__ import annotations

import logging
from typing import Any

from .auto_persist_repo import AutoPersistRepo
from .repository import ChatRepository

logger = logging.getLogger(__name__)


def create_repository(config: dict[str, Any]) -> ChatRepository:
    """Create the chat repository.

    The system now uses a single, optimized storage backend: AutoPersistRepo.
    Any legacy `storage.type` configuration is ignored.
    """
    logger.info("Using auto-persist storage with retention policies")
    return AutoPersistRepo(config)
