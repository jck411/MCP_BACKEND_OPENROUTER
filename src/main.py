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

from src.chat import ChatOrchestrator
from src.config import Configuration
from src.history import create_repository
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
    """
    Event-driven LLM HTTP client that automatically updates when configuration changes.
    
    This client uses the observer pattern to efficiently update only when the runtime
    configuration actually changes, eliminating unnecessary polling on every API call.
    """

    def __init__(self, configuration: Configuration) -> None:
        self.configuration: Configuration = configuration
        self._current_config: dict[str, Any] = {}
        self._current_api_key: str = ""
        self._current_provider: str = ""
        self.client: httpx.AsyncClient | None = None
        
        # Initialize with current configuration
        self._update_client_config()
        
        # Subscribe to configuration changes for event-driven updates
        self.configuration.subscribe_to_changes(self._on_config_change)
        
    def _on_config_change(self, new_config: dict[str, Any]) -> None:
        """Event handler for configuration changes."""
        try:
            # Check if LLM configuration actually changed
            new_llm_config = new_config.get("llm", {})
            active_provider = new_llm_config.get("active", "openai")
            provider_config = new_llm_config.get("providers", {}).get(active_provider, {})
            new_api_key = self.configuration.llm_api_key
            
            if provider_config != self._current_config or new_api_key != self._current_api_key:
                logging.info("LLM configuration changed - updating client")
                logging.info(f"New active provider: {active_provider}")
                logging.info(f"New model: {provider_config.get('model', 'unknown')}")
                
                # Close existing client
                if self.client:
                    asyncio.create_task(self.client.aclose())
                
                # Update configuration
                self._current_config = provider_config
                self._current_api_key = new_api_key
                self._current_provider = self._detect_provider(provider_config.get("base_url", ""))
                
                # Create new HTTP client
                self.client = httpx.AsyncClient(
                    base_url=provider_config["base_url"],
                    headers={"Authorization": f"Bearer {new_api_key}"},
                    timeout=60.0
                )
                
        except Exception as e:
            logging.error(f"Error handling configuration change in LLM client: {e}")
        
    def _update_client_config(self) -> None:
        """Initialize client configuration (called once during __init__)."""
        try:
            llm_config = self.configuration.get_llm_config()
            api_key = self.configuration.llm_api_key
            
            self._current_config = llm_config
            self._current_api_key = api_key
            self._current_provider = self._detect_provider(llm_config.get("base_url", ""))
            
            # Create HTTP client
            self.client = httpx.AsyncClient(
                base_url=llm_config["base_url"],
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=60.0
            )
            
            logging.info(f"LLM client initialized with provider: {self._current_provider}")
            logging.info(f"Model: {llm_config.get('model', 'unknown')}")
                
        except Exception as e:
            logging.error(f"Error initializing LLM client configuration: {e}")
            raise

    @property
    def config(self) -> dict[str, Any]:
        """Get current LLM configuration (cached, no I/O)."""
        return self._current_config
        
    @property
    def api_key(self) -> str:
        """Get current API key (cached, no I/O)."""
        return self._current_api_key
        
    @property
    def provider(self) -> str:
        """Get current provider (cached, no I/O)."""
        return self._current_provider

    def _detect_provider(self, base_url: str) -> str:
        """Detect provider from base URL for potential provider-specific handling."""
        if "openai.com" in base_url:
            return "openai"
        elif "groq.com" in base_url:
            return "groq"
        elif "openrouter.ai" in base_url:
            return "openrouter"
        elif "anthropic.com" in base_url:
            return "anthropic"
        else:
            return "unknown"

    def _build_payload(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, stream: bool = False) -> dict[str, Any]:
        """
        Build API payload by passing through all config parameters.
        This makes the client ready for any new parameters without code changes.
        """
        # Start with required parameters
        payload = {
            "model": self.config["model"],
            "messages": messages,
        }
        
        # Add streaming if requested
        if stream:
            payload["stream"] = True
        
        # Pass through ALL other parameters from config (except infrastructure ones)
        excluded_keys = {"base_url", "model"}  # These are handled separately
        for key, value in self.config.items():
            if key not in excluded_keys:
                payload[key] = value
        
        # Add tools if provided
        if tools:
            payload["tools"] = tools
            
        return payload

    def _extract_reasoning_content(self, response_data: dict[str, Any]) -> str | None:
        """
        Extract thinking/reasoning content from response if present.
        Checks multiple common locations where providers might put reasoning content.
        """
        reasoning_content = None
        
        # Check for reasoning content in various locations
        # Different providers and models may use different field names
        possible_reasoning_fields = [
            "thinking",           # Common field name
            "reasoning",          # Alternative field name  
            "thought_process",    # Another alternative
            "internal_thoughts",  # Anthropic style
            "chain_of_thought",   # CoT models
            "rationale"           # Academic models
        ]
        
        # Check top-level response
        for field in possible_reasoning_fields:
            if field in response_data:
                reasoning_content = response_data[field]
                break
        
        # Check within choices[0] if not found at top level
        if not reasoning_content and "choices" in response_data and response_data["choices"]:
            choice = response_data["choices"][0]
            
            # Check in message
            message = choice.get("message", {})
            for field in possible_reasoning_fields:
                if field in message:
                    reasoning_content = message[field]
                    break
            
            # Check in choice directly
            if not reasoning_content:
                for field in possible_reasoning_fields:
                    if field in choice:
                        reasoning_content = choice[field]
                        break
        
        return reasoning_content

    async def get_response_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """
        Get response from LLM API with automatic parameter pass-through and reasoning extraction.
        Ready for any model type including reasoning models like o1, o3, Claude with thinking, etc.
        """
        if not self.client:
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message="LLM client not initialized"
                )
            )
            
        try:
            payload = self._build_payload(messages, tools, stream=False)

            response = await self.client.post("/chat/completions", json=payload)
            response.raise_for_status()
            result = response.json()

            # Handle empty choices gracefully
            if not result.get("choices"):
                raise McpError(
                    error=types.ErrorData(
                        code=types.PARSE_ERROR,
                        message="No choices in API response",
                    )
                )

            choice = result["choices"][0]
            
            # Extract reasoning content if present
            thinking_content = self._extract_reasoning_content(result)
            
            response_dict = {
                "message": choice["message"],
                "finish_reason": choice.get("finish_reason"),
                "index": choice.get("index", 0),
                "model": result.get("model", self.config["model"]),
            }
            
            # Include thinking/reasoning content if found
            if thinking_content:
                response_dict["thinking"] = thinking_content
            
            return response_dict
            
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
        """
        Get streaming response with automatic parameter pass-through and reasoning handling.
        Supports reasoning models by buffering thinking content and yielding it separately.
        """
        if not self.client:
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message="LLM client not initialized"
                )
            )
            
        try:
            payload = self._build_payload(messages, tools, stream=True)

            # Buffers for reasoning content
            thinking_buffer = ""
            thinking_complete = False
            
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
                expected_types = ["text/event-stream", "text/plain", "application/json", "stream"]
                if not any(t in content_type for t in expected_types):
                    logging.warning(f"Unexpected content-type: {content_type}, proceeding anyway")

                chunk_count = 0
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix

                        if data.strip() == "[DONE]":
                            # If we have accumulated thinking content, yield it as final chunk
                            if thinking_buffer and not thinking_complete:
                                yield {
                                    "type": "thinking",
                                    "content": thinking_buffer,
                                    "complete": True
                                }
                            break

                        try:
                            chunk = json.loads(data)
                            chunk_count += 1
                            
                            # Check for reasoning/thinking content in this chunk
                            thinking_delta = self._extract_thinking_from_chunk(chunk)
                            if thinking_delta is not None:
                                thinking_buffer += thinking_delta
                                # Yield thinking chunk separately
                                yield {
                                    "type": "thinking",
                                    "content": thinking_delta,
                                    "complete": False
                                }
                                continue
                            
                            # Check if thinking section just completed
                            if self._is_thinking_complete(chunk):
                                thinking_complete = True
                                if thinking_buffer:
                                    yield {
                                        "type": "thinking", 
                                        "content": "",  # Empty content signals end
                                        "complete": True,
                                        "full_thinking": thinking_buffer
                                    }
                            
                            # Yield normal content chunks
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

    def _extract_thinking_from_chunk(self, chunk: dict[str, Any]) -> str | None:
        """
        Extract thinking/reasoning content from a streaming chunk.
        Returns the thinking content if found, None otherwise.
        """
        # Check for thinking in various locations within streaming chunk
        possible_thinking_paths = [
            ["thinking"],
            ["reasoning"],
            ["choices", 0, "delta", "thinking"],
            ["choices", 0, "delta", "reasoning"],
            ["choices", 0, "message", "thinking"],
            ["choices", 0, "message", "reasoning"],
            ["delta", "thinking"],
            ["delta", "reasoning"]
        ]
        
        for path in possible_thinking_paths:
            current = chunk
            try:
                for key in path:
                    current = current[key]
                if isinstance(current, str):
                    return current
            except (KeyError, TypeError, IndexError):
                continue
                
        return None

    def _is_thinking_complete(self, chunk: dict[str, Any]) -> bool:
        """
        Check if the thinking section has completed in this chunk.
        Different providers may signal this differently.
        """
        # Check for explicit thinking completion signals
        if chunk.get("thinking_complete"):
            return True
            
        # Check if we're starting to get regular content (thinking usually comes first)
        if "choices" in chunk and chunk["choices"]:
            choice = chunk["choices"][0]
            if "delta" in choice and "content" in choice["delta"]:
                return True
                
        return False

    async def close(self) -> None:
        """Close the HTTP client and unsubscribe from config changes."""
        # Unsubscribe from configuration changes
        self.configuration.unsubscribe_from_changes(self._on_config_change)
        
        # Close HTTP client
        if self.client:
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

    # Create repository for chat history using configured storage mode
    repo = create_repository(config.get_config_dict())

    # Now that MCPClient and LLMClient are defined, make them available
    # in the chat orchestrator module's namespace and rebuild config
    import src.chat.chat_orchestrator
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
            # Stop configuration watching
            await config.stop_watching()
            logging.info("Application shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
