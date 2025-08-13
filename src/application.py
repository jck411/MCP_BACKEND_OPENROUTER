"""
Main application entry point - WebSocket interface with graceful shutdown handling.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys

import src.chat.chat_orchestrator
from src.chat import ChatOrchestrator
from src.clients import LLMClient, MCPClient
from src.config import Configuration
from src.history import create_repository
from src.websocket_server import run_websocket_server

# Configure logging for the application
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


async def main() -> None:
    """Main entry point - WebSocket interface with graceful shutdown handling."""
    config = Configuration()

    config_path = os.path.join(os.path.dirname(__file__), "servers_config.json")
    servers_config = config.load_config(config_path)

    # Get MCP connection configuration
    mcp_connection_config = config.get_mcp_connection_config()

    clients = []
    for name, server_config in servers_config["mcpServers"].items():
        # Only create clients for enabled servers
        if server_config.get("enabled", False):
            clients.append(MCPClient(name, server_config, mcp_connection_config))
        else:
            logging.info(f"Skipping disabled server: {name}")

    # Create repository for chat history using configured storage mode
    repo = create_repository(config.get_config_dict())

    # Now that MCPClient and LLMClient are defined, make them available
    # in the chat orchestrator module's namespace and rebuild config
    src.chat.chat_orchestrator.MCPClient = MCPClient  # type: ignore[attr-defined]
    src.chat.chat_orchestrator.LLMClient = LLMClient  # type: ignore[attr-defined]
    ChatOrchestrator.ChatOrchestratorConfig.model_rebuild()

    # Setup graceful shutdown handler
    shutdown_event = asyncio.Event()

    def signal_handler() -> None:
        """Handle shutdown signals gracefully."""
        logging.info("Received shutdown signal, initiating graceful shutdown...")
        shutdown_event.set()

    # Register signal handlers for graceful shutdown
    if sys.platform != "win32":
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)

    async with LLMClient(config) as llm_client:
        try:
            # Start configuration file watching for event-driven updates
            await config.start_watching()

            # Run server with shutdown handling
            server_task = asyncio.create_task(
                run_websocket_server(
                    clients, llm_client, config.get_config_dict(), repo, config
                )
            )

            # Wait for either server completion or shutdown signal
            done, pending = await asyncio.wait(
                [server_task, asyncio.create_task(shutdown_event.wait())],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel pending tasks if shutdown was requested
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            # Check if server task completed with an exception
            for task in done:
                if task == server_task:
                    exception = task.exception()
                    if exception is not None:
                        raise exception

        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received, shutting down...")
        except Exception as e:
            logging.error(f"Application error: {e}")
            raise
        finally:
            # Stop configuration watching
            await config.stop_watching()
            logging.info("Application shutdown complete")
