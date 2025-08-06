"""
Main module for MCP client implementation.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import signal
import sys
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from typing import Any

import httpx
from mcp import ClientSession, McpError, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from pydantic import AnyUrl

import src.chat_service
from src.chat_service import ChatService
from src.config import Configuration
from src.history.chat_store import JsonlRepo
from src.websocket_server import run_websocket_server

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class MCPClient:
    """
    Official MCP Client implementation following SDK patterns.

    Supports configurable connection timeout and retry behavior through YAML
    configuration. Connection parameters include:
    - max_reconnect_attempts: Maximum number of reconnection attempts
    - initial_reconnect_delay: Initial delay between reconnection attempts
    - max_reconnect_delay: Maximum delay (with exponential backoff)
    - connection_timeout: Timeout for server initialization
    - ping_timeout: Timeout for connection health checks
    """

    def __init__(
        self,
        name: str,
        config: dict[str, Any],
        connection_config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize MCP client with configurable connection parameters.

        Args:
            name: Client name for logging and identification
            config: Server-specific configuration (command, args, etc.)
            connection_config: Connection and retry configuration parameters
        """
        self.name: str = name
        self.config: dict[str, Any] = config
        self.session: ClientSession | None = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self._reconnect_attempts: int = 0
        self._is_connected: bool = False
        self.client_version = "0.1.0"

        # Configure connection parameters from config or use defaults
        conn_config = connection_config or {}
        self._max_reconnect_attempts: int = conn_config.get(
            "max_reconnect_attempts", 5
        )
        self._initial_reconnect_delay: float = conn_config.get(
            "initial_reconnect_delay", 1.0
        )
        self._max_reconnect_delay: float = conn_config.get(
            "max_reconnect_delay", 30.0
        )
        self._connection_timeout: float = conn_config.get("connection_timeout", 30.0)
        self._ping_timeout: float = conn_config.get("ping_timeout", 10.0)

        # Initialize reconnect delay to the configured initial value
        self._reconnect_delay: float = self._initial_reconnect_delay

        # Log configuration for debugging
        logging.info(
            f"MCP client '{name}' configured with: "
            f"max_attempts={self._max_reconnect_attempts}, "
            f"initial_delay={self._initial_reconnect_delay}s, "
            f"max_delay={self._max_reconnect_delay}s, "
            f"connection_timeout={self._connection_timeout}s, "
            f"ping_timeout={self._ping_timeout}s"
        )

    def _resolve_command(self) -> str | None:
        """
        Resolve command configuration to absolute executable path with validation.

        This internal method handles the complex task of finding executable commands
        across different operating systems and environments. It supports both absolute
        paths and commands that need to be resolved through the system PATH.

        The method implements several resolution strategies:
        1. Direct return for absolute paths (if they exist)
        2. PATH resolution using shutil.which for relative commands
        3. Windows-specific workaround for npx/node compatibility issues

        This is critical for MCP server startup since many servers are distributed
        as npm packages or require specific runtime environments.

        Returns:
            str | None: Absolute path to executable if found, None if resolution fails

        Implementation Notes:
            - Windows compatibility: Falls back to 'node' if 'npx' is not found
            - Only returns paths to files that actually exist on the filesystem
            - Does not validate that the file is executable (left to OS)

        Example:
            config = {"command": "python"}  # -> "/usr/bin/python"
            config = {"command": "/opt/custom/tool"}  # -> "/opt/custom/tool"
            config = {"command": "nonexistent"}  # -> None
        """
        command = self.config.get("command")
        if not command:
            return None

        # Handle absolute paths first - validate they exist
        if os.path.isabs(command):
            return command if os.path.exists(command) else None

        # Resolve through system PATH
        resolved = shutil.which(command)

        # Windows-specific npx compatibility workaround
        if not resolved and command == "npx" and sys.platform == "win32":
            node_path = shutil.which("node")
            if node_path:
                logging.warning("Using node instead of npx on Windows")
                return node_path

        return resolved

    async def connect(self) -> None:
        """
        Connect to MCP server using official transport patterns with configurable
        retry logic.

        Implements exponential backoff retry strategy using configuration parameters:
        - Retries up to max_reconnect_attempts times
        - Uses initial_reconnect_delay as starting delay, doubling each attempt
        - Caps delay at max_reconnect_delay to prevent excessive wait times
- Resets delay and attempt count on successful connection

        Raises:
            Exception: If all connection attempts fail
        """
        while self._reconnect_attempts < self._max_reconnect_attempts:
            try:
                await self._attempt_connection()
                self._is_connected = True
                self._reconnect_attempts = 0
                # Reset delay for future reconnection attempts
                self._reconnect_delay = self._initial_reconnect_delay
                return
            except Exception as e:
                self._reconnect_attempts += 1
                self._is_connected = False

                if self._reconnect_attempts >= self._max_reconnect_attempts:
                    # Reset delay for future connection attempts
                    self._reconnect_delay = self._initial_reconnect_delay
                    logging.error(
                        f"Failed to connect to {self.name} after "
                        f"{self._max_reconnect_attempts} attempts: {e}"
                    )
                    raise

                logging.warning(
                    f"Connection attempt {self._reconnect_attempts} failed for "
                    f"{self.name}: {e}. Retrying in {self._reconnect_delay}s..."
                )
                await asyncio.sleep(self._reconnect_delay)

                # Apply exponential backoff with maximum delay limit
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, self._max_reconnect_delay
                )

    async def _attempt_connection(self) -> None:
        """
        Attempt a single connection to the MCP server with comprehensive error handling.

        This internal method performs the low-level work of establishing an MCP client
        connection using the official SDK patterns. It handles command resolution,
        server parameter setup, transport initialization, and session establishment.

        The method implements the standard MCP connection flow:
        1. Resolve command to executable path (may fail if command not found)
        2. Configure server parameters with command, args, and environment
        3. Initialize stdio transport for communication with the server process
        4. Create and initialize the ClientSession with proper client info
        5. Wait for server initialization with timeout protection

        Failure at any step will raise an exception that is caught by the retry
        logic in the connect() method.

        Raises:
            ValueError: If the configured command cannot be resolved to an executable
            asyncio.TimeoutError: If server initialization takes longer than 30 seconds
            Exception: Any other error during transport or session initialization

        Side Effects:
            - Creates server process through stdio transport
            - Modifies self.session to store the active ClientSession
            - Registers transport and session with exit_stack for cleanup
            - Logs successful connection
        """
        command = self._resolve_command()

        if not command:
            raise ValueError(
                f"Command '{self.config.get('command')}' not found in PATH"
            )

        # Configure server parameters with environment inheritance
        server_params = StdioServerParameters(
            command=command,
            args=self.config.get("args", []),
            env={**os.environ, **self.config.get("env", {})}
            if self.config.get("env")
            else None,
        )

        # Initialize stdio transport for server communication
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read_stream, write_stream = stdio_transport

        # Create client info for MCP handshake
        client_info = types.Implementation(name=self.name, version=self.client_version)

        # Initialize client session with transport streams
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream, client_info=client_info)
        )

        # Complete MCP initialization handshake with configurable timeout
        await asyncio.wait_for(
            self.session.initialize(), timeout=self._connection_timeout
        )

        logging.info(f"MCP client '{self.name}' connected successfully")

    async def ping(self) -> bool:
        """Send ping to verify connection is alive."""
        if not self.session or not self._is_connected:
            return False

        try:
            # Use list_tools() as a lightweight ping with configurable timeout
            await asyncio.wait_for(
                self.session.list_tools(), timeout=self._ping_timeout
            )
            return True
        except Exception as e:
            logging.warning(f"Ping failed for {self.name}: {e}")
            self._is_connected = False
            return False

    async def list_tools(self) -> list[types.Tool]:
        """List available tools using official SDK patterns."""
        if not self.session:
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Client {self.name} not connected",
                )
            )

        try:
            result = await self.session.list_tools()
            return result.tools
        except McpError as e:
            logging.error(
                f"MCP error listing tools from {self.name}: {e.error.message}"
            )
            raise
        except Exception as e:
            logging.error(f"Error listing tools from {self.name}: {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Failed to list tools: {e!s}",
                )
            ) from e

    async def list_prompts(self) -> list[types.Prompt]:
        """List available prompts using official SDK patterns."""
        if not self.session:
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Client {self.name} not connected",
                )
            )

        try:
            result = await self.session.list_prompts()
            return result.prompts
        except McpError as e:
            logging.error(
                f"MCP error listing prompts from {self.name}: {e.error.message}"
            )
            raise
        except Exception as e:
            logging.error(f"Error listing prompts from {self.name}: {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Failed to list prompts: {e!s}",
                )
            ) from e

    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> types.GetPromptResult:
        """Get a prompt by name using official SDK patterns."""
        if not self.session:
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Client {self.name} not connected",
                )
            )

        try:
            return await self.session.get_prompt(name, arguments)
        except McpError as e:
            logging.error(
                f"MCP error getting prompt '{name}' from {self.name}: {e.error.message}"
            )
            raise
        except Exception as e:
            logging.error(f"Error getting prompt '{name}' from {self.name}: {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Failed to get prompt '{name}': {e!s}",
                )
            ) from e

    async def list_resources(self) -> list[types.Resource]:
        """List available resources using official SDK patterns."""
        if not self.session:
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Client {self.name} not connected",
                )
            )

        try:
            result = await self.session.list_resources()
            return result.resources
        except McpError as e:
            logging.error(
                f"MCP error listing resources from {self.name}: {e.error.message}"
            )
            raise
        except Exception as e:
            logging.error(f"Error listing resources from {self.name}: {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Failed to list resources: {e!s}",
                )
            ) from e

    async def read_resource(self, uri: str) -> types.ReadResourceResult:
        """Read a resource by URI using official SDK patterns."""
        if not self.session:
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Client {self.name} not connected",
                )
            )

        try:
            resource_uri = AnyUrl(uri)
            return await self.session.read_resource(resource_uri)
        except McpError as e:
            logging.error(
                f"MCP error reading resource '{uri}' from {self.name}: "
                f"{e.error.message}"
            )
            raise
        except Exception as e:
            logging.error(f"Error reading resource '{uri}' from {self.name}: {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Failed to read resource '{uri}': {e!s}",
                )
            ) from e

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> types.CallToolResult:
        """Call a tool using official SDK patterns."""
        if not self.session:
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Client {self.name} not connected",
                )
            )

        try:
            logging.info(f"Calling tool '{name}' on client '{self.name}'")

            result = await self.session.call_tool(name, arguments)
            logging.info(f"Tool '{name}' executed successfully")
            return result
        except McpError as e:
            logging.error(f"MCP error calling tool '{name}': {e.error.message}")
            raise
        except Exception as e:
            logging.error(f"Error calling tool '{name}': {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Tool call failed: {e!s}",
                )
            ) from e

    async def get_tool_schemas(self) -> list[str]:
        """Get tool schemas as JSON strings using SDK's native methods."""
        if not self.session:
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Client {self.name} not connected",
                )
            )

        try:
            tools = await self.list_tools()
            return [tool.model_dump_json() for tool in tools]
        except Exception as e:
            logging.error(f"Error getting tool schemas from {self.name}: {e}")
            raise

    async def close(self) -> None:
        """Close the client connection and clean up resources."""
        async with self._cleanup_lock:
            try:
                self._is_connected = False
                await self.exit_stack.aclose()
                self.session = None
                logging.info(f"MCP client '{self.name}' disconnected")
            except Exception as e:
                logging.error(f"Error during cleanup of client {self.name}: {e}")

    @property
    def is_connected(self) -> bool:
        """Check if client is currently connected."""
        return self._is_connected


class LLMClient:
    """HTTP client for LLM API requests with structured tool call support."""

    def __init__(self, config: dict[str, Any], api_key: str) -> None:
        self.config: dict[str, Any] = config
        self.api_key: str = api_key
        self.client: httpx.AsyncClient = httpx.AsyncClient(
            base_url=config["base_url"],
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )

    async def get_response_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Get response from LLM API with structured tool calls support."""
        try:
            payload = {
                "model": self.config["model"],
                "messages": messages,
                "temperature": self.config.get("temperature", 0.7),
                "max_tokens": self.config.get("max_tokens", 4096),
                "top_p": self.config.get("top_p", 1.0),
            }

            if tools:
                payload["tools"] = tools

            response = await self.client.post("/chat/completions", json=payload)
            response.raise_for_status()
            result = response.json()

            choice = result["choices"][0]
            return {
                "message": choice["message"],
                "finish_reason": choice.get("finish_reason"),
                "index": choice.get("index", 0),
                "usage": result.get("usage"),
                "model": result.get("model", self.config["model"]),
            }
        except httpx.HTTPError as e:
            logging.error(f"HTTP error: {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR, message=f"HTTP error: {e!s}"
                )
            ) from e
        except KeyError as e:
            logging.error(f"Unexpected response format: {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.PARSE_ERROR,
                    message=f"Unexpected response format: {e!s}",
                )
            ) from e
        except Exception as e:
            logging.error(f"LLM API error: {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"LLM API error: {e!s}",
                )
            ) from e

    async def get_streaming_response_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncGenerator[dict[str, Any]]:
        """Get streaming response from LLM API with structured tool calls support."""
        try:
            payload = {
                "model": self.config["model"],
                "messages": messages,
                "temperature": self.config.get("temperature", 0.7),
                "max_tokens": self.config.get("max_tokens", 4096),
                "top_p": self.config.get("top_p", 1.0),
                "stream": True,
            }

            if tools:
                payload["tools"] = tools

            async with self.client.stream(
                "POST", "/chat/completions", json=payload
            ) as response:
                # FAIL FAST: Ensure streaming response is valid
                HTTP_OK = 200
                if response.status_code != HTTP_OK:
                    error_text = await response.aread()
                    raise McpError(
                        error=types.ErrorData(
                            code=types.INTERNAL_ERROR,
                            message=(
                                f"Streaming API error {response.status_code}: "
                                f"{error_text}"
                            ),
                        )
                    )

                content_type = response.headers.get("content-type", "")
                expected_types = ["text/event-stream", "stream"]
                if not any(t in content_type for t in expected_types):
                    raise McpError(
                        error=types.ErrorData(
                            code=types.INTERNAL_ERROR,
                            message=(
                                f"Expected streaming response, got "
                                f"content-type: {content_type}"
                            ),
                        )
                    )

                chunk_count = 0
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix

                        if data.strip() == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            chunk_count += 1
                            yield chunk
                        except json.JSONDecodeError as e:
                            # FAIL FAST: Invalid JSON in stream
                            raise McpError(
                                error=types.ErrorData(
                                    code=types.PARSE_ERROR,
                                    message=f"Invalid JSON in stream chunk: {e}",
                                )
                            ) from e

                # FAIL FAST: Ensure we got at least some data
                if chunk_count == 0:
                    raise McpError(
                        error=types.ErrorData(
                            code=types.INTERNAL_ERROR,
                            message="No streaming chunks received from API",
                        )
                    )

        except httpx.HTTPError as e:
            logging.error(f"HTTP error during streaming: {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR, message=f"HTTP error: {e!s}"
                )
            ) from e
        except Exception as e:
            logging.error(f"LLM streaming API error: {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"LLM streaming API error: {e!s}",
                )
            ) from e

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


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

    llm_config = config.get_llm_config()
    api_key = config.llm_api_key

    # Create repository for chat history
    repo = JsonlRepo("events.jsonl")

    # Now that MCPClient and LLMClient are defined, make them available
    # in the chat_service module's namespace and rebuild ChatServiceConfig
    src.chat_service.MCPClient = MCPClient  # type: ignore[attr-defined]
    src.chat_service.LLMClient = LLMClient  # type: ignore[attr-defined]
    ChatService.ChatServiceConfig.model_rebuild()

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

    async with LLMClient(llm_config, api_key) as llm_client:
        try:
            # Run server with shutdown handling
            server_task = asyncio.create_task(
                run_websocket_server(
                    clients, llm_client, config.get_config_dict(), repo, config
                )
            )

            # Wait for either server completion or shutdown signal
            done, pending = await asyncio.wait(
                [server_task, asyncio.create_task(shutdown_event.wait())],
                return_when=asyncio.FIRST_COMPLETED
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
            logging.info("Application shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
