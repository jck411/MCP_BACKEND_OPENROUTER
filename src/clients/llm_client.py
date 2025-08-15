"""
Event-driven LLM HTTP client that automatically updates when configuration changes.

This client uses the observer pattern to efficiently update only when the runtime
configuration actually changes, eliminating unnecessary polling on every API call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from mcp import McpError, types

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

        # Initialize with current configuration
        self._update_client_config()

        # Subscribe to configuration changes for event-driven updates
        self.configuration.subscribe_to_changes(self._on_config_change)

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
                provider_config = new_llm_config.get("providers", {}).get(
                    active_provider, {}
                )
                new_api_key = self.configuration.llm_api_key

                if (
                    provider_config != self._current_config
                    or new_api_key != self._current_api_key
                ):
                    # Enhanced logging for configuration changes
                    logging.info("🔄 LLM configuration change detected")
                    logging.info(f"Current client state: {self.client is not None}")
                    logging.info(f"Active streams: {self._active_streams}")
                    logging.info(f"New active provider: {active_provider}")
                    logging.info(
                        f"New model: {provider_config.get('model', 'unknown')}"
                    )

                    # Check what type of changes we have
                    client_breaking_changes = self._requires_new_client(
                        provider_config, new_api_key
                    )

                    # Log specific changes with type indicators
                    if provider_config != self._current_config:
                        logging.info("📝 Provider configuration changed")
                        if self._current_config:
                            # Log what specifically changed
                            for key, new_val in provider_config.items():
                                old_val = self._current_config.get(key)
                                if old_val != new_val:
                                    change_type = (
                                        "🔧" if key in client_breaking_changes else "⚡"
                                    )
                                    logging.info(
                                        f"   {change_type} {key}: {old_val} → {new_val}"
                                    )

                    if new_api_key != self._current_api_key:
                        logging.info("🔑 API key changed (requires new client)")

                    # Check if we need to create a new HTTP client
                    if client_breaking_changes:
                        # These changes require a new HTTP client
                        if self._active_streams > 0:
                            logging.warning(
                                f"⏸️  Deferring client replacement due to "
                                f"{self._active_streams} active stream(s)"
                            )
                            logging.info(
                                f"🔧 Client-breaking changes: {client_breaking_changes}"
                            )
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
                        logging.info(
                            "⚡ Applying non-breaking config changes immediately"
                        )
                        self._current_config = provider_config
                        logging.info(
                            "✅ Configuration updated without client replacement"
                        )

            except Exception as e:
                logging.error(
                    f"❌ Error handling configuration change in LLM client: {e}"
                )
                logging.error(f"Exception type: {type(e).__name__}")
                logging.error(f"Exception args: {e.args}")
                logging.error(f"Traceback: {traceback.format_exc()}")

    def _requires_new_client(
        self, new_config: dict[str, Any], new_api_key: str
    ) -> list[str]:
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
        client_breaking_keys = []

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

    async def _apply_config_change(
        self, provider_config: dict[str, Any], new_api_key: str
    ) -> None:
        """Apply configuration change by replacing HTTP client."""
        # Close existing client
        if self.client:
            logging.info("🔄 Replacing HTTP client with new configuration")
            await self.client.aclose()

        # Update configuration
        self._current_config = provider_config
        self._current_api_key = new_api_key
        self._current_provider = self._detect_provider(
            provider_config.get("base_url", "")
        )

        # Create new HTTP client
        self.client = httpx.AsyncClient(
            base_url=provider_config["base_url"],
            headers={"Authorization": f"Bearer {new_api_key}"},
            timeout=60.0,
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
            self._current_provider = self._detect_provider(
                llm_config.get("base_url", "")
            )

            # Create HTTP client
            self.client = httpx.AsyncClient(
                base_url=llm_config["base_url"],
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=60.0,
            )

            logging.info(
                f"LLM client initialized with provider: {self._current_provider}"
            )
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
        if (
            not reasoning_content
            and "choices" in response_data
            and response_data["choices"]
        ):
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
        Get response from LLM API with automatic parameter pass-through and
        reasoning extraction. Ready for any model type including reasoning models
        like o1, o3, Claude with thinking, etc.
        """
        if not self.client:
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR, message="LLM client not initialized"
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

    async def get_streaming_response_with_tools(  # noqa: PLR0912, PLR0915
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncGenerator[dict[str, Any]]:
        """
        Get streaming response with automatic parameter pass-through and
        reasoning handling. Supports reasoning models by buffering thinking
        content and yielding it separately.
        """
        if not self.client:
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR, message="LLM client not initialized"
                )
            )

        # Increment active stream counter to prevent config changes
        self._active_streams += 1
        logging.debug(f"📈 Started stream, active streams: {self._active_streams}")

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
                expected_types = [
                    "text/event-stream",
                    "text/plain",
                    "application/json",
                    "stream",
                ]
                if not any(t in content_type for t in expected_types):
                    logging.warning(
                        f"Unexpected content-type: {content_type}, proceeding anyway"
                    )

                chunk_count = 0
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix

                        if data.strip() == "[DONE]":
                            # If we have accumulated thinking content, yield it
                            # as final chunk
                            if thinking_buffer and not thinking_complete:
                                yield {
                                    "type": "thinking",
                                    "content": thinking_buffer,
                                    "complete": True,
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
                                    "complete": False,
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
                                        "full_thinking": thinking_buffer,
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
            # Enhanced error logging for better debugging
            logging.error(f"HTTP error during streaming: {e}")
            logging.error(f"HTTP error type: {type(e).__name__}")
            logging.error(f"HTTP error args: {e.args}")

            # Log response details if available (only for HTTPStatusError)
            if isinstance(e, httpx.HTTPStatusError) and e.response is not None:
                logging.error(f"Response status: {e.response.status_code}")
                logging.error(f"Response headers: {dict(e.response.headers)}")
                try:
                    # Try to read response body for more context
                    if hasattr(e.response, "text"):
                        error_body = e.response.text
                        logging.error(
                            f"Response body: {error_body[:1000]}..."
                        )  # Truncate long responses
                    elif hasattr(e.response, "content"):
                        error_body = str(e.response.content[:1000])
                        logging.error(f"Response content: {error_body}...")
                except Exception as read_err:
                    logging.error(f"Could not read response body: {read_err}")
            else:
                logging.error("No response object available in HTTP error")

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
        finally:
            # Decrement active stream counter and check for pending config changes
            self._active_streams -= 1
            logging.debug(f"📉 Ended stream, active streams: {self._active_streams}")

            # Check if we can apply any pending configuration changes
            if self._active_streams == 0:
                task = asyncio.create_task(self._check_pending_config_change())
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

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
            ["delta", "reasoning"],
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
        if chunk.get("choices"):
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
