"""
Chat Service for MCP Platform

This module handles the business logic for chat sessions, including:
- Conversation management with simple default prompts
- Tool orchestration
- MCP client coordination
- LLM interactions
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from mcp import types
from pydantic import BaseModel, ConfigDict, Field

from src.config import Configuration
from src.history import ChatEvent, Usage

if TYPE_CHECKING:
    from src.tool_schema_manager import ToolSchemaManager
else:
    from src.tool_schema_manager import ToolSchemaManager

logger = logging.getLogger(__name__)


class ToolCallContext(BaseModel):
    """Parameters for tool call iteration handling."""
    conv: list[dict[str, Any]]
    tools_payload: list[dict[str, Any]]
    conversation_id: str
    request_id: str
    assistant_msg: dict[str, Any]
    full_content: str


class ChatMessage(BaseModel):
    """
    Represents a chat message with metadata.
    Pydantic model for proper validation and serialization.
    """
    type: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatService:
    """
    Conversation orchestrator - recommended pattern
    1. Takes your message
    2. Figures out what tools might be needed
    3. Asks the AI to respond (and use tools if needed)
    4. Sends you back the response
    """

    class ChatServiceConfig(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        clients: list[Any]  # MCPClient
        llm_client: Any  # LLMClient
        config: dict[str, Any]
        repo: Any  # ChatRepository
        configuration: Configuration
        ctx_window: int = 4000

    def __init__(
        self,
        service_config: ChatServiceConfig,
    ):
        self.clients = service_config.clients
        self.llm_client = service_config.llm_client
        self.config = service_config.config
        self.repo = service_config.repo
        self.configuration = service_config.configuration
        self.ctx_window = service_config.ctx_window
        self.chat_conf = self.config.get("chat", {}).get("service", {})
        self.tool_mgr: ToolSchemaManager | None = None
        self._init_lock = asyncio.Lock()
        self._ready = asyncio.Event()
        self._resource_catalog: list[str] = []  # Initialize resource catalog

    async def initialize(self) -> None:
        """Initialize the chat service and all MCP clients."""
        async with self._init_lock:
            if self._ready.is_set():
                return

            # Connect to all clients and collect results
            connection_results = await asyncio.gather(
                *(c.connect() for c in self.clients), return_exceptions=True
            )

            # Filter out only successfully connected clients
            connected_clients = []
            for i, result in enumerate(connection_results):
                if isinstance(result, Exception):
                    logger.warning(
                        f"Client '{self.clients[i].name}' failed to connect: {result}"
                    )
                else:
                    connected_clients.append(self.clients[i])

            if not connected_clients:
                logger.warning(
                    "No MCP clients connected - running with basic functionality"
                )
            else:
                logger.info(
                    f"Successfully connected to {len(connected_clients)} out of "
                    f"{len(self.clients)} MCP clients"
                )

            # Use connected clients for tool management (empty list is acceptable)
            self.tool_mgr = ToolSchemaManager(connected_clients)
            await self.tool_mgr.initialize()

            # Update resource catalog to only include available resources
            await self._update_resource_catalog_on_availability()
            self._system_prompt = await self._make_system_prompt()

            logger.info(
                "ChatService ready: %d tools, %d resources, %d prompts",
                len(self.tool_mgr.get_openai_tools()),
                len(self._resource_catalog),
                len(self.tool_mgr.list_available_prompts()),
            )

            logger.info("Resource catalog: %s", self._resource_catalog)

            # Configurable system prompt logging
            if self.chat_conf.get("logging", {}).get("system_prompt", True):
                logger.info("System prompt being used:\n%s", self._system_prompt)
            else:
                logger.debug("System prompt logging disabled in configuration")

            self._ready.set()

    async def process_message(
        self,
        conversation_id: str,
        user_msg: str,
        request_id: str,
    ) -> AsyncGenerator[ChatMessage]:
        """
        Process a user message with consistent history management and delta persistence.

        Flow:
        1. Persist user message first (with idempotency check)
        2. Build history from repository
        3. Stream response while persisting deltas
        4. Compact deltas into final assistant message
        """
        await self._ready.wait()
        self._validate_streaming_support()

        # Handle idempotency and user message persistence
        should_continue = await self._handle_user_message_persistence(
            conversation_id, user_msg, request_id
        )
        if not should_continue:
            async for msg in self._yield_existing_response(conversation_id, request_id):
                yield msg
            return

        # Build conversation and generate response
        conv = await self._build_conversation_history(conversation_id, user_msg)

        # Type assertion: tool_mgr is guaranteed non-None after validation
        assert self.tool_mgr is not None
        tools_payload = self.tool_mgr.get_openai_tools()

        # Stream and handle tool calls
        async for msg in self._stream_and_handle_tools(
            conv, tools_payload, conversation_id, request_id
        ):
            yield msg

    def _validate_streaming_support(self) -> None:
        """
        Validate that streaming is supported by both tool manager and LLM client.

        This internal method performs fail-fast validation to ensure that all required
        components for streaming are available before attempting to process a message.
        Called early in process_message() to prevent partial execution.

        Raises:
            RuntimeError: If tool manager is not initialized or if LLM client
                         doesn't support streaming functionality
        """
        if not self.tool_mgr:
            raise RuntimeError("Tool manager not initialized")

        if not hasattr(self.llm_client, 'get_streaming_response_with_tools'):
            raise RuntimeError(
                "LLM client does not support streaming. "
                "Use chat_once() for non-streaming responses."
            )

    async def _handle_user_message_persistence(
        self, conversation_id: str, user_msg: str, request_id: str
    ) -> bool:
        """
        Handle user message persistence with comprehensive idempotency checks.

        This internal method implements the core idempotency logic for the chat system.
        It first checks if an assistant response already exists for the given
        request_id, then attempts to persist the user message. If the user message
        is a duplicate, it performs a second check for existing responses to handle
        race conditions.

        The method follows a defensive programming pattern:
        1. Check for existing response (prevents double-billing)
        2. Attempt to persist user message
        3. If duplicate detected, re-check for response (handles concurrent requests)

        Args:
            conversation_id: The conversation identifier
            user_msg: The user's message content
            request_id: Unique identifier for this request (used for idempotency)

        Returns:
            bool: True if processing should continue (new request), False if response
                  already exists (cached response should be returned)

        Side Effects:
            - Creates a ChatEvent for the user message and persists it to repository
            - Computes and caches token count for the user message
        """
        # Check for existing response first
        existing_response = await self._get_existing_assistant_response(
            conversation_id, request_id
        )
        if existing_response:
            logger.info(f"Returning cached response for request_id: {request_id}")
            return False

        # Persist user message
        user_ev = ChatEvent(
            conversation_id=conversation_id,
            seq=0,  # Will be assigned by repository
            type="user_message",
            role="user",
            content=user_msg,
            extra={"request_id": request_id},
        )
        was_added = await self.repo.add_event(user_ev)

        if not was_added:
            # Check for existing response again after duplicate detection
            existing_response = await self._get_existing_assistant_response(
                conversation_id, request_id
            )
            if existing_response:
                logger.info(
                    "Returning existing response for duplicate "
                    f"request_id: {request_id}"
                )
                return False

        return True

    async def _yield_existing_response(
        self, conversation_id: str, request_id: str
    ) -> AsyncGenerator[ChatMessage]:
        """
        Yield existing response content as ChatMessage for cached responses.

        This internal method retrieves and streams a previously computed assistant
        response when handling duplicate requests. It maintains consistency with
        the streaming interface by yielding ChatMessage objects even for cached
        content.

        Used in the idempotency flow when _handle_user_message_persistence returns
        False, indicating that a response already exists for the given request_id.

        Args:
            conversation_id: The conversation identifier
            request_id: The request identifier to find cached response for

        Yields:
            ChatMessage: Single message containing the cached response content
            with metadata indicating it's from cache

        Note:
            If no existing response is found (edge case), this generator yields
            nothing, which will result in an empty response stream.
        """
        existing_response = await self._get_existing_assistant_response(
            conversation_id, request_id
        )
        if existing_response and existing_response.content:
            content_str = (
                existing_response.content
                if isinstance(existing_response.content, str)
                else str(existing_response.content)
            )
            yield ChatMessage(
                type="text",
                content=content_str,
                metadata={"cached": True}
            )

    async def _build_conversation_history(
        self, conversation_id: str, user_msg: str
    ) -> list[dict[str, Any]]:
        """Build conversation history from repository."""
        events = await self.repo.get_conversation_history(conversation_id, limit=50)

        # Build conversation with system prompt and recent history
        conv = [{"role": "system", "content": self._system_prompt}]

        for event in events:
            if event.type == "user_message":
                conv.append({"role": "user", "content": str(event.content or "")})
            elif event.type == "assistant_message":
                conv.append({"role": "assistant", "content": str(event.content or "")})
            elif (
                event.type == "tool_result"
                and event.extra
                and "tool_call_id" in event.extra
            ):
                conv.append({
                    "role": "tool",
                    "tool_call_id": event.extra["tool_call_id"],
                    "content": str(event.content or "")
                })

        # Add current user message
        conv.append({"role": "user", "content": user_msg})
        return conv

    async def _stream_and_handle_tools(
        self,
        conv: list[dict[str, Any]],
        tools_payload: list[dict[str, Any]],
        conversation_id: str,
        request_id: str,
    ) -> AsyncGenerator[ChatMessage]:
        """Stream response and handle tool calls iteratively."""
        full_content = ""

        # Initial response streaming
        assistant_msg: dict[str, Any] = {}
        async for chunk in self._stream_llm_response_with_deltas(
            conv, tools_payload, conversation_id, request_id
        ):
            if isinstance(chunk, ChatMessage):
                if chunk.type == "text":
                    full_content += chunk.content
                yield chunk
            else:
                assistant_msg = chunk
                if assistant_msg.get("content"):
                    full_content += assistant_msg["content"]

        self._log_initial_response(assistant_msg)

        # Handle tool call iterations
        context = ToolCallContext(
            conv=conv,
            tools_payload=tools_payload,
            conversation_id=conversation_id,
            request_id=request_id,
            assistant_msg=assistant_msg,
            full_content=full_content
        )
        async for msg in self._handle_tool_call_iterations(context):
            if isinstance(msg, str):
                full_content = msg  # Updated full content
            else:
                yield msg

        # Final compaction
        await self.repo.compact_deltas(
            conversation_id,
            request_id,
            full_content,
            usage=self._convert_usage(None),
            model=self.llm_client.config.get("model", "")
        )

    def _log_initial_response(self, assistant_msg: dict[str, Any]) -> None:
        """
        Log initial LLM response if configured in chat service settings.

        This internal method provides structured logging of the first LLM response
        in a conversation turn. It creates a standardized log entry that includes
        the assistant message, usage information, and model details for debugging
        and monitoring purposes.

        The method respects the chat service logging configuration and only logs
        if 'llm_replies' is enabled. It uses the _log_llm_reply helper to ensure
        consistent log formatting across all LLM interactions.

        Args:
            assistant_msg: The raw assistant message dictionary from the LLM API,
                          containing content, tool_calls, and other response data

        Side Effects:
            - May write to application logs if logging is enabled
            - Does not modify the assistant_msg or any other state
        """
        reply_data = {
            "message": assistant_msg,
            "usage": None,
            "model": self.llm_client.config.get("model", "")
        }
        self._log_llm_reply(reply_data, "Streaming initial response")

    async def _handle_tool_call_iterations(
        self, context: ToolCallContext
    ) -> AsyncGenerator[ChatMessage | str]:
        """Handle iterative tool calls with hop limit."""
        hops = 0

        while context.assistant_msg.get("tool_calls"):
            max_tool_hops = self.configuration.get_max_tool_hops()
            if hops >= max_tool_hops:
                warning_msg = (
                    f"⚠️ Reached maximum tool call limit ({max_tool_hops}). "
                    "Stopping to prevent infinite recursion."
                )
                logger.warning(f"Maximum tool hops ({max_tool_hops}) reached, stopping")
                context.full_content += "\n\n" + warning_msg
                yield ChatMessage(
                    type="text",
                    content=warning_msg,
                    metadata={"finish_reason": "tool_limit_reached"}
                )
                break

            # Execute tool calls
            context.conv.append({
                "role": "assistant",
                "content": context.assistant_msg.get("content") or "",
                "tool_calls": context.assistant_msg["tool_calls"],
            })
            await self._execute_tool_calls(
                context.conv, context.assistant_msg["tool_calls"]
            )

            # Get follow-up response
            context.assistant_msg = {}
            async for chunk in self._stream_llm_response_with_deltas(
                context.conv, context.tools_payload, context.conversation_id,
                context.request_id, hop_number=hops + 1
            ):
                if isinstance(chunk, ChatMessage):
                    if chunk.type == "text":
                        context.full_content += chunk.content
                    yield chunk
                else:
                    context.assistant_msg = chunk
                    if context.assistant_msg.get("content"):
                        context.full_content += context.assistant_msg["content"]

            hops += 1

        yield context.full_content  # Return updated full_content

    async def _stream_llm_response_with_deltas(
        self,
        conv: list[dict[str, Any]],
        tools_payload: list[dict[str, Any]],
        conversation_id: str,
        user_request_id: str,
        hop_number: int = 0
    ) -> AsyncGenerator[ChatMessage | dict[str, Any]]:
        """
        Stream response from LLM, persist deltas, and yield chunks to user.

        Key behavior: Message content streams immediately to user while tool calls
        are accumulated in the background for efficient execution.
        """
        message_buffer = ""
        current_tool_calls: list[dict[str, Any]] = []
        finish_reason: str | None = None
        delta_index = 0  # Track delta order for proper reconstruction

        # Stream from LLM API - this is where the real-time magic happens
        async for chunk in self.llm_client.get_streaming_response_with_tools(
            conv, tools_payload
        ):
            # Skip malformed chunks (defensive programming)
            if "choices" not in chunk or not chunk["choices"]:
                continue

            choice = chunk["choices"][0]
            delta = choice.get("delta", {})

            # PRIORITY 1: Stream content immediately to provide responsive UX
            # Content deltas are the most time-sensitive part of the response
            if content := delta.get("content"):
                message_buffer += content  # Build complete message for final storage

                # Persist this content delta for history reconstruction
                # Each delta gets a unique index for proper ordering
                delta_event = ChatEvent(
                    conversation_id=conversation_id,
                    seq=0,  # Repository will assign sequence number
                    type="meta",  # Internal event type for system operations
                    content=content,
                    extra={
                        "kind": "assistant_delta",  # Specific delta type identifier
                        "user_request_id": user_request_id,  # Link to request
                        "hop_number": hop_number,  # Track tool call iteration depth
                        "delta_index": delta_index,  # Preserve streaming order
                        "request_id": user_request_id  # Redundant for easier queries
                    }
                )
                await self.repo.add_event(delta_event)
                delta_index += 1  # Increment for next delta

                # IMMEDIATE USER FEEDBACK: Stream to user without waiting
                # This is what makes the UI feel responsive during long responses
                yield ChatMessage(
                    type="text",
                    content=content,
                    metadata={"type": "delta", "hop": hop_number}
                )

            # PRIORITY 2: Accumulate tool calls for batch execution
            # Tool calls need to be complete before execution, so we buffer them
            if tool_calls := delta.get("tool_calls"):
                # Use robust accumulation that handles out-of-order and partial deltas
                self._accumulate_tool_calls(current_tool_calls, tool_calls)

            # PRIORITY 3: Track completion status for proper flow control
            if "finish_reason" in choice:
                finish_reason = choice["finish_reason"]

        # Provide user feedback when transitioning to tool execution phase
        # This helps users understand what's happening during longer operations
        if current_tool_calls and any(
            call["function"]["name"] for call in current_tool_calls
        ):
            yield ChatMessage(
                type="tool_execution",
                content=f"Executing {len(current_tool_calls)} tool(s)...",
                metadata={"tool_count": len(current_tool_calls), "hop": hop_number}
            )

        # Return complete assistant message for tool call processing
        # This allows the caller to determine if more LLM interactions are needed
        yield {
            "content": message_buffer or None,
            "tool_calls": current_tool_calls if current_tool_calls and any(
                call["function"]["name"] for call in current_tool_calls
            ) else None,
            "finish_reason": finish_reason
        }

    def _accumulate_tool_calls(
        self, current_tool_calls: list[dict[str, Any]], tool_calls: list[dict[str, Any]]
    ) -> None:
        """
        Accumulate streaming tool call deltas into complete tool call structures.

        This is a critical internal method that handles the complex task of assembling
        tool calls from streaming deltas. LLM APIs stream tool calls in fragments:
        - Tool call IDs may arrive first
        - Function names may arrive in multiple chunks
        - Function arguments arrive as partial JSON strings
        - Deltas may arrive out of order or with missing indices

        The method uses a defensive accumulation strategy:
        1. Ensures the current_tool_calls list has enough slots for all indices
        2. Uses list-based accumulation for both names and arguments
        3. Rebuilds complete strings from accumulated parts after each delta

        This approach handles edge cases like:
        - Out-of-order deltas (common with parallel processing)
        - Missing or duplicate deltas
        - Partial JSON in arguments that needs to be concatenated

        Args:
            current_tool_calls: The mutable list being built up (modified in-place)
            tool_calls: List of delta fragments from the current streaming chunk

        Side Effects:
            - Modifies current_tool_calls in-place by extending or updating elements
            - Creates intermediate "_parts" lists to handle out-of-order accumulation
            - Rebuilds name and arguments strings from parts after each update

        Implementation Notes:
            The method maintains both the final strings (name, arguments) and the
            intermediate parts lists (name_parts, args_parts) to handle streaming
            edge cases robustly. This prevents corruption when deltas arrive
            out of order.
        """
        for tool_call in tool_calls:
            # Ensure we have enough space in the buffer for this tool call index
            while len(current_tool_calls) <= tool_call.get("index", 0):
                current_tool_calls.append({
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""}
                })

            idx = tool_call.get("index", 0)

            # Accumulate tool call ID (usually arrives first and complete)
            if "id" in tool_call:
                current_tool_calls[idx]["id"] = tool_call["id"]

            # Accumulate function details using part-based strategy
            if "function" in tool_call:
                func = tool_call["function"]

                # Handle function name accumulation (may arrive in chunks)
                if "name" in func:
                    # Initialize parts list if not exists
                    if "name_parts" not in current_tool_calls[idx]["function"]:
                        current_tool_calls[idx]["function"]["name_parts"] = []
                    # Accumulate this name fragment
                    current_tool_calls[idx]["function"]["name_parts"].append(func["name"])
                    # Rebuild complete name from all parts
                    current_tool_calls[idx]["function"]["name"] = "".join(
                        current_tool_calls[idx]["function"]["name_parts"]
                    )

                # Handle function arguments accumulation (partial JSON strings)
                if "arguments" in func:
                    # Initialize parts list if not exists
                    if "args_parts" not in current_tool_calls[idx]["function"]:
                        current_tool_calls[idx]["function"]["args_parts"] = []
                    # Accumulate this argument fragment
                    current_tool_calls[idx]["function"]["args_parts"].append(func["arguments"])
                    # Rebuild complete arguments JSON from all parts
                    current_tool_calls[idx]["function"]["arguments"] = "".join(
                        current_tool_calls[idx]["function"]["args_parts"]
                    )

    async def _execute_tool_calls(
        self, conv: list[dict[str, Any]], calls: list[dict[str, Any]]
    ) -> None:
        """
        Execute tool calls and append results to conversation history.

        This internal method handles the execution phase of tool calls after they
        have been accumulated from streaming deltas. It processes each tool call
        sequentially, validates the JSON arguments, executes the tool through the
        tool manager, and formats the results for inclusion in the conversation.

        The method implements several important behaviors:
        1. Parses JSON arguments with fallback to empty dict for malformed JSON
        2. Executes tools through the tool manager (which handles validation)
        3. Extracts content from MCP CallToolResult using _pluck_content helper
        4. Appends tool results to conversation in OpenAI chat format

        Args:
            conv: The conversation history list (modified in-place)
            calls: List of complete tool call dictionaries with id, name, and arguments

        Side Effects:
            - Modifies conv by appending tool result messages
            - Executes external tool calls through MCP clients
            - May raise exceptions if tool execution fails (handled by caller)

        Note:
            Tool calls are executed sequentially, not in parallel. This ensures
            deterministic execution order and prevents resource conflicts between tools.
        """
        assert self.tool_mgr is not None

        for call in calls:
            tool_name = call["function"]["name"]
            # Parse JSON arguments with defensive handling for malformed JSON
            args = json.loads(call["function"]["arguments"] or "{}")

            # Execute tool through tool manager (handles validation and routing)
            result = await self.tool_mgr.call_tool(tool_name, args)

            # Extract readable content from MCP result structure
            content = self._pluck_content(result)

            # Append tool result to conversation in OpenAI format
            conv.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": content,
                }
            )

    async def _get_existing_assistant_response(
        self, conversation_id: str, request_id: str
    ) -> ChatEvent | None:
        """Get existing assistant response for a request_id if it exists."""
        existing_assistant_id = await self.repo.get_last_assistant_reply_id(
            conversation_id, request_id
        )
        if existing_assistant_id:
            events = await self.repo.get_events(conversation_id)
            for event in events:
                if event.id == existing_assistant_id:
                    return event
        return None

    async def chat_once(
        self,
        conversation_id: str,
        user_msg: str,
        request_id: str,
    ) -> ChatEvent:
        """
        Non-streaming chat with consistent history management.
        
        Flow:
        1. Persist user message first (with idempotency check)
        2. Build history from repository
        3. Generate response with tools
        4. Persist final assistant message
        """  # noqa: W293
        await self._ready.wait()

        if not self.tool_mgr:
            raise RuntimeError("Tool manager not initialized")

        # Check for existing response to prevent double-billing
        existing_response = await self._get_existing_assistant_response(
            conversation_id, request_id
        )
        if existing_response:
            logger.info(
                f"Returning cached response for request_id: {request_id}"
            )
            return existing_response

        # 1) persist user message FIRST with idempotency check
        user_ev = ChatEvent(
            conversation_id=conversation_id,
            seq=0,  # Will be assigned by repository
            type="user_message",
            role="user",
            content=user_msg,
            extra={"request_id": request_id},
        )
        was_added = await self.repo.add_event(user_ev)

        if not was_added:
            # User message already exists, check for assistant response again
            existing_response = await self._get_existing_assistant_response(
                conversation_id, request_id
            )
            if existing_response:
                logger.info(
                    "Returning existing response for duplicate "
                    f"request_id: {request_id}"
                )
                return existing_response
            # No assistant response yet, continue with LLM call

        # 2) build canonical history from repo
        conv = await self._build_conversation_history(conversation_id, user_msg)

        # 4) Generate assistant response with usage tracking
        (
            assistant_full_text,
            total_usage,
            model
        ) = await self._generate_assistant_response(conv)

        # 5) persist assistant message with usage and reference to user request
        assistant_ev = ChatEvent(
            conversation_id=conversation_id,
            seq=0,  # Will be assigned by repository
            type="assistant_message",
            role="assistant",
            content=assistant_full_text,
            usage=total_usage,
            provider=self.llm_client.config.get("provider", "unknown"),
            model=model,
            extra={"user_request_id": request_id},
        )
        await self.repo.add_event(assistant_ev)

        return assistant_ev

    def _convert_usage(self, api_usage: dict[str, Any] | None) -> Usage:
        """
        Convert LLM API usage statistics to internal Usage model.

        This internal method normalizes usage data from different LLM API providers
        into the platform's standardized Usage Pydantic model. It handles cases
        where usage information may be missing or incomplete.

        The method provides safe defaults (0 tokens) for missing fields to ensure
        consistent usage tracking across all LLM interactions. This is important
        for cost monitoring and rate limiting.

        Args:
            api_usage: Raw usage dictionary from LLM API response, may be None
                      Expected fields: prompt_tokens, completion_tokens, total_tokens

        Returns:
            Usage: Pydantic model with normalized token counts, using 0 for
                   missing fields

        Example:
            api_usage = {
                "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15
            }
            usage = self._convert_usage(api_usage)
            # Returns: Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        """
        if not api_usage:
            return Usage()

        return Usage(
            prompt_tokens=api_usage.get("prompt_tokens", 0),
            completion_tokens=api_usage.get("completion_tokens", 0),
            total_tokens=api_usage.get("total_tokens", 0),
        )

    async def _generate_assistant_response(
        self, conv: list[dict[str, Any]]
    ) -> tuple[str, Usage, str]:
        """Generate assistant response using tools if needed."""
        if not self.tool_mgr:
            raise RuntimeError("Tool manager not initialized")

        tools_payload = self.tool_mgr.get_openai_tools()

        assistant_full_text = ""
        total_usage = Usage()
        model = ""

        reply = await self.llm_client.get_response_with_tools(conv, tools_payload)
        assistant_msg = reply["message"]

        # Log LLM reply if configured
        self._log_llm_reply(reply, "Initial LLM response")

        # Track usage from this API call
        if reply.get("usage"):
            call_usage = self._convert_usage(reply["usage"])
            total_usage.prompt_tokens += call_usage.prompt_tokens
            total_usage.completion_tokens += call_usage.completion_tokens
            total_usage.total_tokens += call_usage.total_tokens

        # Store model from first API call
        model = reply.get("model", "")

        if txt := assistant_msg.get("content"):
            assistant_full_text += txt

        hops = 0
        while calls := assistant_msg.get("tool_calls"):
            max_tool_hops = self.configuration.get_max_tool_hops()
            if hops >= max_tool_hops:
                logger.warning(
                    f"Maximum tool hops ({max_tool_hops}) reached, stopping recursion"
                )
                assistant_full_text += (
                    f"\n\n⚠️ Reached maximum tool call limit ({max_tool_hops}). "
                    "Stopping to prevent infinite recursion."
                )
                break

            conv.append(
                {
                    "role": "assistant",
                    "content": assistant_msg.get("content") or "",
                    "tool_calls": calls,
                }
            )

            for call in calls:
                tool_name = call["function"]["name"]
                args = json.loads(call["function"]["arguments"] or "{}")

                result = await self.tool_mgr.call_tool(tool_name, args)
                content = self._pluck_content(result)

                conv.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": content,
                    }
                )

            reply = await self.llm_client.get_response_with_tools(conv, tools_payload)
            assistant_msg = reply["message"]

            # Log LLM reply if configured
            self._log_llm_reply(reply, f"Tool call follow-up response (hop {hops + 1})")

            # Track usage from subsequent API calls
            if reply.get("usage"):
                call_usage = self._convert_usage(reply["usage"])
                total_usage.prompt_tokens += call_usage.prompt_tokens
                total_usage.completion_tokens += call_usage.completion_tokens
                total_usage.total_tokens += call_usage.total_tokens

            if txt := assistant_msg.get("content"):
                assistant_full_text += txt

            hops += 1

        return assistant_full_text, total_usage, model

    def _log_llm_reply(self, reply: dict[str, Any], context: str) -> None:
        """Log LLM reply if configured."""
        logging_config = self.chat_conf.get("logging", {})
        if not logging_config.get("llm_replies", False):
            return

        message = reply.get("message", {})
        content = message.get("content", "")
        tool_calls = message.get("tool_calls", [])

        # Truncate content if configured
        truncate_length = logging_config.get("llm_reply_truncate_length", 500)
        if content and len(content) > truncate_length:
            content = content[:truncate_length] + "..."

        log_parts = [f"LLM Reply ({context}):"]

        if content:
            log_parts.append(f"Content: {content}")

        if tool_calls:
            log_parts.append(f"Tool calls: {len(tool_calls)}")
            for i, call in enumerate(tool_calls):
                func_name = call.get("function", {}).get("name", "unknown")
                log_parts.append(f"  - Tool {i+1}: {func_name}")

        usage = reply.get("usage", {})
        if usage:
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', 0)
            log_parts.append(
                f"Usage: {prompt_tokens}p + {completion_tokens}c = {total_tokens}t"
            )

        model = reply.get("model", "unknown")
        log_parts.append(f"Model: {model}")

        logger.info(" | ".join(log_parts))

    def _pluck_content(self, res: types.CallToolResult) -> str:
        """
        Extract readable content from MCP CallToolResult for conversation context.

        This internal method handles the complex task of converting MCP tool results
        into plain text suitable for inclusion in LLM conversation context. MCP tool
        results can contain various content types (text, images, binary data, embedded
        resources) that need different handling strategies.

        The method implements a fallback chain:
        1. Handle structured content (if present) by JSON serialization
        2. Process each content item based on its type:
           - TextContent: Extract text directly
           - ImageContent: Create descriptive placeholder
           - BlobResourceContents: Create size-based placeholder
           - EmbeddedResource: Recursively extract from nested resource
           - Unknown types: Create type-based placeholder
        3. Return "✓ done" for empty results to indicate successful completion

        This approach ensures that all tool results can be meaningfully represented
        in text form for the LLM, while preserving important metadata about non-text
        content types.

        Args:
            res: MCP CallToolResult containing the tool execution result

        Returns:
            str: Human-readable text representation of the tool result content

        Side Effects:
            - May log warnings if structured content serialization fails
            - Does not modify the input result object
        """
        if not res.content:
            return "✓ done"

        # Handle structured content with graceful fallback
        if hasattr(res, "structuredContent") and res.structuredContent:
            try:
                return json.dumps(res.structuredContent, indent=2)
            except Exception as e:
                logger.warning(f"Failed to serialize structured content: {e}")

        # Extract text from each piece of content
        out: list[str] = []
        for item in res.content:
            if isinstance(item, types.TextContent):
                out.append(item.text)
            elif isinstance(item, types.ImageContent):
                out.append(f"[Image: {item.mimeType}, {len(item.data)} bytes]")
            elif isinstance(item, types.BlobResourceContents):
                out.append(f"[Binary content: {len(item.blob)} bytes]")
            elif isinstance(item, types.EmbeddedResource):
                if isinstance(item.resource, types.TextResourceContents):
                    out.append(f"[Embedded resource: {item.resource.text}]")
                else:
                    out.append(f"[Embedded resource: {type(item.resource).__name__}]")
            else:
                out.append(f"[{type(item).__name__}]")

        return "\n".join(out)

    async def _make_system_prompt(self) -> str:
        """Build the system prompt with actual resource contents and prompts."""
        base = self.chat_conf["system_prompt"].rstrip()

        assert self.tool_mgr is not None

        # Only include resources that are actually available
        available_resources = await self._get_available_resources()
        if available_resources:
            base += "\n\n**Available Resources:**"
            for uri, content_info in available_resources.items():
                resource_info = self.tool_mgr.get_resource_info(uri)
                name = resource_info.resource.name if resource_info else uri

                base += f"\n\n**{name}** ({uri}):"

                for content in content_info:
                    if isinstance(content, types.TextResourceContents):
                        lines = content.text.strip().split('\n')
                        for line in lines:
                            base += f"\n{line}"
                    elif isinstance(content, types.BlobResourceContents):
                        base += f"\n[Binary content: {len(content.blob)} bytes]"
                    else:
                        base += f"\n[{type(content).__name__} available]"

        prompt_names = self.tool_mgr.list_available_prompts()
        if prompt_names:
            prompt_list = []
            for name in prompt_names:
                pinfo = self.tool_mgr.get_prompt_info(name)
                if pinfo:
                    desc = pinfo.prompt.description or "No description available"
                    prompt_list.append(f"• **{name}**: {desc}")

            prompts_text = "\n".join(prompt_list)
            base += (
                f"\n\n**Available Prompts** (use apply_prompt method):\n"
                f"{prompts_text}"
            )

        return base

    async def _get_available_resources(
        self,
    ) -> dict[str, list[types.TextResourceContents | types.BlobResourceContents]]:
        """
        Check resource availability and return only resources that can be read
        successfully.

        This implements graceful degradation by only including working resources
        in the system prompt, following best practices for resource management.
        """
        available_resources: dict[
            str, list[types.TextResourceContents | types.BlobResourceContents]
        ] = {}

        if not self._resource_catalog or not self.tool_mgr:
            return available_resources

        for uri in self._resource_catalog:
            try:
                resource_result = await self.tool_mgr.read_resource(uri)
                if resource_result.contents:
                    # Only include resources that have actual content
                    available_resources[uri] = resource_result.contents
                    logger.debug(f"Resource {uri} is available and loaded")
                else:
                    logger.debug(f"Resource {uri} has no content, skipping")
            except Exception as e:
                # Log the error but don't include in system prompt
                # This prevents the LLM from being told about broken resources
                logger.warning(
                    f"Resource {uri} is unavailable and will be excluded "
                    f"from system prompt: {e}"
                )
                continue

        if available_resources:
            logger.info(
                f"Including {len(available_resources)} available resources "
                f"in system prompt"
            )
        else:
            logger.info(
                "No resources are currently available - system prompt will not "
                "include resource section"
            )

        return available_resources

    async def _update_resource_catalog_on_availability(self) -> None:
        """
        Update the resource catalog to reflect current availability.

        This implements a circuit-breaker-like pattern where we periodically
        check if previously failed resources have become available again.
        """
        if not self.tool_mgr:
            return

        # Get all registered resources from the tool manager
        all_resource_uris = self.tool_mgr.list_available_resources()

        # Filter to only include resources that are actually available
        available_uris = []
        for uri in all_resource_uris:
            try:
                resource_result = await self.tool_mgr.read_resource(uri)
                if resource_result.contents:
                    available_uris.append(uri)
            except Exception:
                # Skip unavailable resources silently
                continue

        # Update the catalog to only include working resources
        self._resource_catalog = available_uris
        logger.debug(
            f"Updated resource catalog: {len(available_uris)} of "
            f"{len(all_resource_uris)} resources are available"
        )

    async def cleanup(self) -> None:
        """Clean up resources by closing all connected MCP clients and LLM client."""
        if not self.tool_mgr:
            raise RuntimeError("Tool manager not initialized")

        # Get connected clients from tool manager
        for client in self.tool_mgr.clients:
            try:
                await client.close()
            except Exception as e:
                logger.warning(f"Error closing client {client.name}: {e}")

        # Close LLM HTTP client to prevent dangling connections
        try:
            await self.llm_client.close()
            logger.info("LLM client closed successfully")
        except Exception as e:
            logger.warning(f"Error closing LLM client: {e}")

    async def apply_prompt(self, name: str, args: dict[str, str]) -> list[dict]:
        """Apply a parameterized prompt and return conversation messages."""
        if not self.tool_mgr:
            raise RuntimeError("Tool manager not initialized")

        res = await self.tool_mgr.get_prompt(name, args)

        return [
            {"role": m.role, "content": m.content.text}
            for m in res.messages
            if isinstance(m.content, types.TextContent)
        ]
