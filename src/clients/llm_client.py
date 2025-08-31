"""
Event-driven LLM HTTP client that automatically updates when configuration changes.

This client uses the observer pattern to efficiently update only when the runtime
configuration actually changes, eliminating unnecessary polling on every API call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from collections.abc import AsyncGenerator
from typing import Any, cast

import httpx
from mcp import McpError, types

from src.chat.models import (
    AssistantMessage,
    ChatCompletionMessage,
    LLMResponseData,
    ToolDefinition,
)
from src.config import Configuration


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
        self._active_streams: int = 0  # Track active streaming requests
        self._pending_config_change: dict[str, Any] | None = None  # Queue changes
        self._config_lock: asyncio.Lock = asyncio.Lock()  # Protect config changes
        self._background_tasks: set[asyncio.Task[Any]] = set()  # Track background tasks

        # Cache connection pool configuration for performance
        self._connection_pool_config = self.configuration.get_connection_pool_config()
        self._connection_logging_config = self.configuration.get_connection_logging_config()

        # Initialize with current configuration
        self._update_client_config()

        # Subscribe to configuration changes for event-driven updates
        self.configuration.subscribe_to_changes(self._on_config_change)

        # Set up connection logging if enabled
        self._setup_connection_logging()

    def _on_config_change(self, new_config: dict[str, Any]) -> None:
        """Event handler for configuration changes."""
        task = asyncio.create_task(self._handle_config_change_async(new_config))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _handle_config_change_async(self, new_config: dict[str, Any]) -> None:
        """Handle configuration changes asynchronously with stream protection."""
        async with self._config_lock:
            try:
                # Check if LLM configuration actually changed
                new_llm_config = new_config.get("llm", {})
                active_provider = new_llm_config.get("active", "openai")
                provider_config = new_llm_config.get("providers", {}).get(active_provider, {})
                new_api_key = self.configuration.llm_api_key

                if provider_config != self._current_config or new_api_key != self._current_api_key:
                    # Enhanced logging for configuration changes
                    logging.info("🔄 LLM configuration change detected")
                    logging.info(f"Current client state: {self.client is not None}")
                    logging.info(f"Active streams: {self._active_streams}")
                    logging.info(f"New active provider: {active_provider}")
                    logging.info(f"New model: {provider_config.get('model', 'unknown')}")

                    # Check what type of changes we have
                    client_breaking_changes = self._requires_new_client(provider_config, new_api_key)

                    # Log specific changes with type indicators
                    if provider_config != self._current_config:
                        logging.info("📝 Provider configuration changed")
                        if self._current_config:
                            # Log what specifically changed
                            for key, new_val in provider_config.items():
                                old_val = self._current_config.get(key)
                                if old_val != new_val:
                                    change_type = "🔧" if key in client_breaking_changes else "⚡"
                                    logging.info(f"   {change_type} {key}: {old_val} → {new_val}")

                    if new_api_key != self._current_api_key:
                        logging.info("🔑 API key changed (requires new client)")

                    # Check if we need to create a new HTTP client
                    if client_breaking_changes:
                        # These changes require a new HTTP client
                        if self._active_streams > 0:
                            logging.warning(
                                f"⏸️  Deferring client replacement due to {self._active_streams} active stream(s)"
                            )
                            logging.info(f"🔧 Client-breaking changes: {client_breaking_changes}")
                            self._pending_config_change = {
                                "provider_config": provider_config,
                                "api_key": new_api_key,
                                "active_provider": active_provider,
                            }
                            return
                        # Safe to replace client immediately
                        await self._apply_config_change(provider_config, new_api_key)
                    else:
                        # These are just API parameter changes - update immediately
                        logging.info("⚡ Applying non-breaking config changes immediately")
                        self._current_config = provider_config
                        logging.info("✅ Configuration updated without client replacement")

            except Exception as e:
                logging.error(f"❌ Error handling configuration change in LLM client: {e}")
                logging.error(f"Exception type: {type(e).__name__}")
                logging.error(f"Exception args: {e.args}")
                logging.error(f"Traceback: {traceback.format_exc()}")

    def _requires_new_client(self, new_config: dict[str, Any], new_api_key: str) -> list[str]:
        """
        Determine which config changes require creating a new HTTP client.

        Connection-level changes (require new client):
        - base_url: switching API providers (OpenAI -> OpenRouter -> Groq)
        - api_key: authentication changes
        - timeout: connection timeout settings

        Hyperparameter changes (just payload data, no client change needed):
        - temperature, top_p, presence_penalty, frequency_penalty
        - max_tokens, model (same provider), stop sequences
        - Any other model parameters

        Returns:
            List of config keys that require client replacement
        """
        client_breaking_keys: list[str] = []

        # API key change always requires new client (authentication)
        if new_api_key != self._current_api_key:
            client_breaking_keys.append("api_key")

        # Check for HTTP client-level changes (connection settings)
        for key in ["base_url", "timeout"]:
            if key in new_config and new_config[key] != self._current_config.get(key):
                client_breaking_keys.append(key)

        # Provider change implies base_url change (connection routing)
        new_provider = self._detect_provider(new_config.get("base_url", ""))
        if new_provider != self._current_provider:
            client_breaking_keys.append("provider")

        return client_breaking_keys

    async def _apply_config_change(self, provider_config: dict[str, Any], new_api_key: str) -> None:
        """Apply configuration change by replacing HTTP client."""
        # Close existing client
        if self.client:
            self._log_connection_event(
                "client_replacement",
                {"provider": self._current_provider, "reason": "configuration_change"},
            )
            logging.info("🔄 Replacing HTTP client with new configuration")
            await self.client.aclose()

        # Update configuration
        self._current_config = provider_config
        self._current_api_key = new_api_key
        self._current_provider = self._detect_provider(provider_config.get("base_url", ""))

        # Create new HTTP client with configurable connection pooling
        self.client = httpx.AsyncClient(
            base_url=provider_config["base_url"],
            headers={
                "Authorization": f"Bearer {new_api_key}",
                "Content-Type": "application/json",
            },
            timeout=self._connection_pool_config["request_timeout_seconds"],
            http2=True,
            # Configurable connection limits for performance tuning
            limits=httpx.Limits(
                max_connections=self._connection_pool_config["max_connections"],
                max_keepalive_connections=self._connection_pool_config["max_keepalive_connections"],
                keepalive_expiry=self._connection_pool_config["keepalive_expiry_seconds"],
            ),
            trust_env=False,
        )

        self._log_connection_event(
            "client_created",
            {
                "provider": self._current_provider,
                "max_connections": self._connection_pool_config["max_connections"],
                "max_keepalive": self._connection_pool_config["max_keepalive_connections"],
                "timeout": self._connection_pool_config["request_timeout_seconds"],
            },
        )
        logging.info("✅ New HTTP client created successfully")

    async def _check_pending_config_change(self) -> None:
        """Check and apply pending configuration changes when no streams are active."""
        async with self._config_lock:
            if self._pending_config_change and self._active_streams == 0:
                logging.info("▶️  Applying deferred configuration change")
                change = self._pending_config_change
                self._pending_config_change = None

                await self._apply_config_change(
                    change["provider_config"],
                    change["api_key"],
                )

    def _update_client_config(self) -> None:
        """Initialize client configuration (called once during __init__)."""
        try:
            llm_config = self.configuration.get_llm_config()
            api_key = self.configuration.llm_api_key

            self._current_config = llm_config
            self._current_api_key = api_key
            self._current_provider = self._detect_provider(llm_config.get("base_url", ""))

            # Create HTTP client with configurable connection pooling
            self.client = httpx.AsyncClient(
                base_url=llm_config["base_url"],
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self._connection_pool_config["request_timeout_seconds"],
                http2=True,
                # Configurable connection limits for performance tuning
                limits=httpx.Limits(
                    max_connections=self._connection_pool_config["max_connections"],
                    max_keepalive_connections=self._connection_pool_config["max_keepalive_connections"],
                    keepalive_expiry=self._connection_pool_config["keepalive_expiry_seconds"],
                ),
                trust_env=False,
            )

            self._log_connection_event(
                "client_initialized",
                {
                    "provider": self._current_provider,
                    "model": llm_config.get("model", "unknown"),
                    "max_connections": self._connection_pool_config["max_connections"],
                    "timeout": self._connection_pool_config["request_timeout_seconds"],
                },
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
        if "groq.com" in base_url:
            return "groq"
        if "openrouter.ai" in base_url:
            return "openrouter"
        if "anthropic.com" in base_url:
            return "anthropic"
        return "unknown"

    def _setup_connection_logging(self) -> None:
        """Set up connection logging based on configuration."""
        if not self._connection_logging_config["enabled"]:
            return

        if self._connection_logging_config["pool_stats"]:
            # Start periodic pool statistics logging
            task = asyncio.create_task(self._log_pool_stats_periodically())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        if self._connection_logging_config["connection_events"]:
            logging.info("🔌 Connection event logging enabled for LLM client")

    async def _log_pool_stats_periodically(self) -> None:
        """Periodically log connection pool statistics."""
        interval = self._connection_logging_config["pool_stats_interval_seconds"]

        while True:
            try:
                await asyncio.sleep(interval)
                if self.client and hasattr(self.client, "_pool"):
                    pool = self.client._pool
                    if hasattr(pool, "_pool"):
                        # httpx connection pool statistics
                        total_connections = len(pool._pool)
                        available_connections = sum(1 for conn in pool._pool if conn.is_available())
                        active_connections = total_connections - available_connections

                        logging.info(
                            f"🔌 Connection pool stats - Active: {active_connections}, "
                            f"Available: {available_connections}, Total: {total_connections}"
                        )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.debug(f"Error logging pool stats: {e}")
                break

    def _log_connection_event(self, event_type: str, details: dict[str, Any]) -> None:
        """Log a connection event if logging is enabled."""
        if not (self._connection_logging_config["enabled"] and self._connection_logging_config["connection_events"]):
            return

        log_message = f"🔌 Connection {event_type}: {details}"
        logging.info(log_message)

    def _log_http_request(
        self,
        method: str,
        url: str,
        status_code: int | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Log HTTP request details if logging is enabled."""
        if not (self._connection_logging_config["enabled"] and self._connection_logging_config["http_requests"]):
            return

        message_parts = [f"🔌 HTTP {method} {url}"]
        if status_code is not None:
            message_parts.append(f"Status: {status_code}")
        if duration_ms is not None:
            message_parts.append(f"Duration: {duration_ms:.2f}ms")

        logging.info(" | ".join(message_parts))

    def _log_connection_reuse(self, reused: bool, connection_id: str | None = None) -> None:
        """Log connection reuse information if logging is enabled."""
        if not (self._connection_logging_config["enabled"] and self._connection_logging_config["connection_reuse"]):
            return

        status = "reused" if reused else "new"
        details = f"Connection {status}"
        if connection_id:
            details += f" (ID: {connection_id})"

        logging.debug(f"🔌 {details}")

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """
        Build API payload by passing through all config parameters.
        This makes the client ready for any new parameters without code changes.
        """
        # Start with required parameters
        payload: dict[str, Any] = {
            "model": self.config["model"],
            "messages": messages,
        }

        # Add streaming if requested
        if stream:
            payload["stream"] = True

        # Pass through ALL other parameters from config (except infrastructure ones)
        excluded_keys = {
            "base_url",
            "model",
            # Provider-incompatible or confusing keys to strip proactively
            # Some providers use different names or don't support these
            "stop_tokens",
            "end_of_text",
            "stop_token_ids",
            "end_token_id",
        }
        for key, value in self.config.items():
            if key not in excluded_keys and value is not None:
                payload[key] = value

        # Add tools if provided
        if tools:
            payload["tools"] = tools

        # Remove any None values that might have slipped in
        return {k: v for k, v in payload.items() if v is not None}

    def _extract_reasoning_content(self, response_data: dict[str, Any]) -> str | None:
        """
        Extract thinking/reasoning content from response if present.
        Checks multiple common locations where providers might put reasoning content.
        """
        reasoning_content = None

        # Check for reasoning content in various locations
        # Different providers and models may use different field names
        possible_reasoning_fields = [
            "thinking",  # Common field name
            "reasoning",  # Alternative field name
            "thought_process",  # Another alternative
            "internal_thoughts",  # Anthropic style
            "chain_of_thought",  # CoT models
            "rationale",  # Academic models
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
        self,
        messages: list[ChatCompletionMessage],
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponseData:
        """
        Get response from LLM API with full typing support and automatic parameter
        pass-through. Supports reasoning models like o1, o3, Claude with thinking, etc.
        """
        if not self.client:
            raise McpError(error=types.ErrorData(code=types.INTERNAL_ERROR, message="LLM client not initialized"))

        try:
            # Convert typed inputs to dict format for API, omitting None fields
            dict_messages = [msg.model_dump(exclude_none=True) for msg in messages]
            dict_tools = [tool.model_dump(exclude_none=True) for tool in tools] if tools else None

            payload = self._build_payload(dict_messages, dict_tools, stream=False)

            start_time = time.monotonic()
            response = await self.client.post("/chat/completions", json=payload)
            response.raise_for_status()
            result = response.json()
            duration_ms = (time.monotonic() - start_time) * 1000

            self._log_http_request("POST", "/chat/completions", response.status_code, duration_ms)

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

            # Convert to typed response
            assistant_msg = AssistantMessage.from_dict(choice["message"])

            return LLMResponseData(
                message=assistant_msg,
                finish_reason=choice.get("finish_reason"),
                index=choice.get("index", 0),
                model=result.get("model", self.config["model"]),
                thinking=thinking_content,
            )

        except httpx.HTTPError as e:
            logging.error(f"HTTP error: {e}")
            raise McpError(error=types.ErrorData(code=types.INTERNAL_ERROR, message=f"HTTP error: {e!s}")) from e
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
        self,
        messages: list[ChatCompletionMessage],
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncGenerator[dict[str, Any]]:
        """
        Get streaming response with full typing support and automatic parameter
        pass-through. Supports reasoning models by buffering thinking content.
        """
        if not self.client:
            raise McpError(error=types.ErrorData(code=types.INTERNAL_ERROR, message="LLM client not initialized"))

        # Increment active stream counter to prevent config changes
        self._active_streams += 1
        self._log_connection_event(
            "stream_started",
            {
                "active_streams": self._active_streams,
                "provider": self._current_provider,
            },
        )
        logging.debug(f"📈 Started stream, active streams: {self._active_streams}")

        # Fast JSON loader (use stdlib for portability and clear typing)
        def fast_json_loads(s: str) -> dict[str, Any]:
            return cast(dict[str, Any], json.loads(s))

        try:
            # Convert typed inputs to dict format for API, omitting None fields
            dict_messages = [msg.model_dump(exclude_none=True) for msg in messages]
            dict_tools = [tool.model_dump(exclude_none=True) for tool in tools] if tools else None

            payload = self._build_payload(dict_messages, dict_tools, stream=True)

            async with self.client.stream(
                "POST",
                "/chat/completions",
                json=payload,
                headers={"Accept": "text/event-stream", "Accept-Encoding": "identity"},
            ) as response:
                # FAIL FAST: Ensure streaming response is valid
                HTTP_OK = 200
                if response.status_code != HTTP_OK:
                    error_text = await response.aread()
                    raise McpError(
                        error=types.ErrorData(
                            code=types.INTERNAL_ERROR,
                            message=(f"Streaming API error {response.status_code}: {error_text}"),
                        )
                    )

                content_type = response.headers.get("content-type", "")
                expected_types = [
                    "text/event-stream",
                    "text/plain",
                    "application/json",
                    "stream",
                ]
                if not any(t in content_type for t in expected_types):
                    logging.warning(f"Unexpected content-type: {content_type}, proceeding anyway")

                chunk_count = 0
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix

                        if data.strip() == "[DONE]":
                            break

                        try:
                            chunk: dict[str, Any] = fast_json_loads(data)
                            chunk_count += 1

                            # Yield raw chunk dict to avoid per-chunk Pydantic cost
                            if "choices" in chunk:
                                yield chunk  # handled by StreamingHandler

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
            # Enhanced error logging for better debugging
            logging.error(f"HTTP error during streaming: {e}")
            logging.error(f"HTTP error type: {type(e).__name__}")
            logging.error(f"HTTP error args: {e.args}")

            # Log response details if available (only for HTTPStatusError)
            if isinstance(e, httpx.HTTPStatusError):
                response = e.response
                logging.error(f"Response status: {response.status_code}")
                logging.error(f"Response headers: {dict(response.headers)}")
                try:
                    # Try to read response body for more context
                    if hasattr(response, "text"):
                        error_body = response.text
                        logging.error(f"Response body: {error_body[:1000]}...")  # Truncate long responses
                    elif hasattr(response, "content"):
                        error_body = str(response.content[:1000])
                        logging.error(f"Response content: {error_body}...")
                except Exception as read_err:
                    logging.error(f"Could not read response body: {read_err}")
            else:
                logging.error("No response object available in HTTP error")

            raise McpError(error=types.ErrorData(code=types.INTERNAL_ERROR, message=f"HTTP error: {e!s}")) from e
        except Exception as e:
            logging.error(f"LLM streaming API error: {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"LLM streaming API error: {e!s}",
                )
            ) from e
        finally:
            # Decrement active stream counter and check for pending config changes
            self._active_streams -= 1
            self._log_connection_event(
                "stream_ended",
                {
                    "active_streams": self._active_streams,
                    "provider": self._current_provider,
                },
            )
            logging.debug(f"📉 Ended stream, active streams: {self._active_streams}")

            # Check if we can apply any pending configuration changes
            if self._active_streams == 0:
                task = asyncio.create_task(self._check_pending_config_change())
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

    async def close(self) -> None:
        """Close the HTTP client and unsubscribe from config changes."""
        # Unsubscribe from configuration changes
        self.configuration.unsubscribe_from_changes(self._on_config_change)

        # Close HTTP client
        if self.client:
            self._log_connection_event(
                "client_closing",
                {
                    "provider": self._current_provider,
                    "active_streams": self._active_streams,
                },
            )
            await self.client.aclose()
            self._log_connection_event("client_closed", {"provider": self._current_provider})

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit."""
        await self.close()
