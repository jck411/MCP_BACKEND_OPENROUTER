"""
Main application entry point.
"""

from __future__ import annotations

import asyncio

from src.application import main

if __name__ == "__main__":
    asyncio.run(main())
