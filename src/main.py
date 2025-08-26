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
from typing import Dict, Any

import src.chat.chat_orchestrator
from src.chat import ChatOrchestrator
from src.clients import LLMClient, MCPClient
from src.config import Configuration
from src.history import create_repository
from src.websocket_server import run_websocket_server


def _configure_advanced_logging(logging_config: Dict[str, Any]) -> None:
    """
    Advanced logging configuration with hierarchical loggers and feature control.

    This approach is more efficient than setting individual loggers because:
    1. Uses parent logger inheritance for automatic propagation
    2. Supports different log levels per module
    3. Enables/disables features at the configuration level
    4. Pre-configures known logger hierarchies
    """
    # Level mapping for efficient lookup
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }

    # Set global level
    global_level = logging_config.get("level", "WARNING")
    logging.getLogger().setLevel(level_map.get(global_level, logging.WARNING))

    # Module-to-logger mapping for efficient configuration
    module_logger_map = {
        "chat": {
            "loggers": ["src.chat"],
            "default_level": "INFO",
            "features": ["llm_replies", "system_prompt", "tool_execution", "tool_results"]
        },
        "connection_pool": {
            "loggers": ["src.clients"],
            "default_level": "INFO",
            "features": ["connection_events", "pool_stats", "http_requests"]
        },
        "mcp": {
            "loggers": ["mcp", "src.clients.mcp_client"],
            "default_level": "INFO",
            "features": ["connection_attempts", "health_checks", "tool_calls", "tool_arguments", "tool_results"]
        }
    }

    modules_config = logging_config.get("modules", {})

    # Configure each module's loggers
    for module_name, module_config in modules_config.items():
        if not isinstance(module_config, dict):
            continue

        # Get the module's specific level, or use global default
        module_level = module_config.get("level", module_logger_map.get(module_name, {}).get("default_level", global_level))
        level_value = level_map.get(module_level, logging.WARNING)

        # Set level on parent loggers (efficient - children inherit)
        for logger_name in module_logger_map.get(module_name, {}).get("loggers", []):
            logging.getLogger(logger_name).setLevel(level_value)

        # Store feature flags for runtime checking (more efficient than repeated config lookups)
        if not hasattr(logging, '_module_features'):
            logging._module_features = {}
        logging._module_features[module_name] = module_config.get("enable_features", {})

    # Advanced configuration (for future enhancement)
    advanced_config = logging_config.get("advanced", {})
    if advanced_config.get("structured_logging"):
        # Could implement JSON structured logging here
        pass

    if advanced_config.get("async_logging"):
        # Could implement async log writing here
        pass


def _on_logging_config_change(new_config: Dict[str, Any]) -> None:
    """
    Handle real-time logging configuration changes.

    This function is called whenever the configuration file is modified,
    allowing for dynamic logging reconfiguration without restart.
    """
    try:
        logging_config = new_config.get("logging", {})
        if logging_config:
            # Reconfigure logging with new settings
            _configure_advanced_logging(logging_config)
            logging.info("🔄 Logging configuration updated in real-time")
    except Exception as e:
        # Log errors but don't crash the application
        logging.error(f"❌ Failed to update logging configuration: {e}")


# Configure logging for the application
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


async def main() -> None:
    """Main entry point - WebSocket interface with graceful shutdown handling."""
    config = Configuration()

    # Apply consolidated logging configuration from YAML
    logging_config = config.get_logging_config()

    if "format" in logging_config:
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setFormatter(logging.Formatter(logging_config["format"]))

    # Initialize advanced logging configuration
    _configure_advanced_logging(logging_config)

    # Subscribe to configuration changes for real-time logging updates
    config.subscribe_to_changes(_on_logging_config_change)

    config_path = os.path.join(os.path.dirname(__file__), "servers_config.json")
    servers_config = config.load_config(config_path)

    # Get MCP connection configuration
    mcp_connection_config = config.get_mcp_connection_config()

    # Get MCP logging configuration
    mcp_logging_config = config.get_mcp_logging_config()

    # Combine connection and logging configuration for MCP clients
    mcp_client_config = {**mcp_connection_config, **mcp_logging_config}

    clients: list[MCPClient] = []
    for name, server_config in servers_config["mcpServers"].items():
        # Only create clients for enabled servers
        if server_config.get("enabled", False):
            clients.append(MCPClient(name, server_config, mcp_client_config))
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


if __name__ == "__main__":
    asyncio.run(main())


def cli_main() -> None:
    """Synchronous CLI entrypoint that runs the async main."""
    asyncio.run(main())
