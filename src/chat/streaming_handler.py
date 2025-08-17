"""
Streaming Response Handler

Handles complex/fragile streaming operations:
- LLM response streaming
- Delta accumulation and persistence
- Streaming tool call accumulation
- Hybrid message yielding (ChatMessage + dict)
- Frontend message streaming
- Tool call iteration handling

This is the most complex and fragile part of the chat system. Streaming bugs
are hard to debug, so this isolation makes it easier to add detailed logging
for what's sent to frontend.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from src.chat.logging_utils import log_llm_reply
from src.chat.models import (
    AssistantMessage,
    ChatMessage,
    ConversationHistory,
    StreamingChunk,
    ThinkingChunk,
    ToolCallContext,
    ToolCallDelta,
    ToolDefinition,
    ToolMessage,
)
from src.history import ChatEvent

if TYPE_CHECKING:
    from src.chat.resource_loader import ResourceLoader
    from src.chat.tool_executor import ToolExecutor
    from src.clients import LLMClient
    from src.history.repository import ChatRepository

logger = logging.getLogger(__name__)


class StreamingHandler:
    """Handles streaming responses and tool call iterations."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_executor: ToolExecutor,
        repo: ChatRepository,
        resource_loader: ResourceLoader,
        chat_conf: dict[str, Any],
    ):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.repo = repo
        self.resource_loader = resource_loader
        self.chat_conf = chat_conf

    def validate_streaming_support(self) -> None:
        """
        Validate that streaming is supported by LLM client.

        This method performs fail-fast validation to ensure that all required
        components for streaming are available before attempting to process a message.
        Called early in process_message() to prevent partial execution.

        Raises:
            RuntimeError: If LLM client doesn't support streaming functionality
        """
        if not hasattr(self.llm_client, "get_streaming_response_with_tools"):
            raise RuntimeError(
                "LLM client does not support streaming. "
                "Use chat_once() for non-streaming responses."
            )

    async def stream_and_handle_tools(
        self,
        conv: ConversationHistory,
        tools_payload: list[ToolDefinition],
        conversation_id: str,
        request_id: str,
    ) -> AsyncGenerator[ChatMessage]:
        """Stream response and handle tool calls iteratively."""
        logger.info(
            "→ Frontend: starting streaming response for request_id=%s", request_id
        )

        full_content = ""

        # Initial response streaming - collect complete response
        raw_assistant_msg = None
        async for chunk in self.stream_llm_response_with_deltas(
            conv, tools_payload, conversation_id, request_id
        ):
            if isinstance(chunk, ChatMessage):
                if chunk.type == "text":
                    full_content += chunk.content
                logger.debug("→ Frontend: text delta, length=%d", len(chunk.content))
                yield chunk
            else:
                # This is our completed dict response
                raw_assistant_msg = chunk
                if raw_assistant_msg.get("content"):
                    full_content += raw_assistant_msg["content"]

        if not raw_assistant_msg:
            logger.warning("No assistant message received from streaming")
            return

        # Convert dict to typed AssistantMessage
        assistant_msg = AssistantMessage.from_dict(raw_assistant_msg)

        self.log_initial_response(raw_assistant_msg)

        # Handle tool call iterations
        context = ToolCallContext(
            conv=conv,
            tools_payload=tools_payload,
            conversation_id=conversation_id,
            request_id=request_id,
            assistant_msg=assistant_msg,
            full_content=full_content,
        )
        async for msg in self.handle_tool_call_iterations(context):
            if isinstance(msg, str):
                full_content = msg  # Updated full content
            else:
                logger.debug("→ Frontend: tool iteration message, type=%s", msg.type)
                yield msg

        # Final compaction
        logger.info("→ Repository: compacting deltas for request_id=%s", request_id)
        await self.repo.compact_deltas(
            conversation_id,
            request_id,
            full_content,
            model=self.llm_client.config.get("model", ""),
        )
        logger.info("← Repository: delta compaction completed")
        logger.info(
            "← Frontend: streaming response completed for request_id=%s", request_id
        )

    async def handle_tool_call_iterations(
        self, context: ToolCallContext
    ) -> AsyncGenerator[ChatMessage | str]:
        """Handle iterative tool calls with hop limit."""
        hops = 0

        while context.assistant_msg.tool_calls:
            should_stop, warning_msg = self.tool_executor.check_tool_hop_limit(hops)
            if should_stop and warning_msg:
                context.full_content += "\n\n" + warning_msg
                logger.info("→ Frontend: tool hop limit warning")
                yield ChatMessage(
                    type="text",
                    content=warning_msg,
                    metadata={"finish_reason": "tool_limit_reached"},
                )
                break

            logger.info("Starting tool call iteration %d", hops + 1)

            # Execute tool calls - add assistant message to conversation
            assistant_message = AssistantMessage(
                content=context.assistant_msg.content or "",
                tool_calls=context.assistant_msg.tool_calls,
            )
            context.conv.add_message(assistant_message)

            # Execute the tool calls and get updated conversation
            conv_dict = context.conv.get_dict_format()
            await self.tool_executor.execute_tool_calls(
                conv_dict,
                [tc.model_dump() for tc in context.assistant_msg.tool_calls],
            )

            # Update the conversation object with tool responses
            # The tool executor adds tool response messages to the dict conversation
            current_msg_count = len(context.conv.get_dict_format())
            # Get only the new tool messages
            new_messages = conv_dict[current_msg_count:]

            # Add tool response messages to the conversation object
            for msg_dict in new_messages:
                tool_msg = ToolMessage(
                    content=msg_dict["content"], tool_call_id=msg_dict["tool_call_id"]
                )
                context.conv.add_message(tool_msg)  # Get follow-up response
            logger.info("→ LLM: requesting follow-up response for hop %d", hops + 1)
            raw_assistant_msg = None
            async for chunk in self.stream_llm_response_with_deltas(
                context.conv,
                context.tools_payload,
                context.conversation_id,
                context.request_id,
                hop_number=hops + 1,
            ):
                if isinstance(chunk, ChatMessage):
                    if chunk.type == "text":
                        context.full_content += chunk.content
                    yield chunk
                else:
                    raw_assistant_msg = chunk
                    if raw_assistant_msg.get("content"):
                        context.full_content += raw_assistant_msg["content"]

            # Convert new response to typed model
            if raw_assistant_msg:
                context.assistant_msg = AssistantMessage.from_dict(raw_assistant_msg)

            hops += 1
            logger.info("Completed tool call iteration %d", hops)

        yield context.full_content  # Return updated full_content

    async def stream_llm_response_with_deltas(
        self,
        conv: ConversationHistory,
        tools_payload: list[ToolDefinition],
        conversation_id: str,
        user_request_id: str,
        hop_number: int = 0,
    ) -> AsyncGenerator[ChatMessage | dict[str, Any]]:
        """
        Stream response from LLM, persist deltas, and yield chunks to user.

        Key behavior: Message content streams immediately to user while tool calls
        are accumulated in the background for efficient execution.
        """
        logger.info("→ LLM: starting streaming request (hop %d)", hop_number)

        message_buffer = ""
        current_tool_calls: list[dict[str, Any]] = []
        finish_reason: str | None = None
        delta_index = 0  # Track delta order for proper reconstruction

        # Stream from LLM API using typed models
        async for chunk in self.llm_client.get_streaming_response_with_tools(
            conv.get_api_format(), tools_payload
        ):
            # Handle StreamingChunk (normal content)
            if isinstance(chunk, StreamingChunk) and chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta

                # Stream content immediately to user
                if delta.content:
                    content = delta.content
                    message_buffer += content

                    # Persist this content delta for history reconstruction
                    delta_event = ChatEvent(
                        conversation_id=conversation_id,
                        seq=0,  # Repository will assign sequence number
                        type="meta",  # Internal event type for system operations
                        content=content,
                        extra={
                            "kind": "assistant_delta",
                            "user_request_id": user_request_id,
                            "hop_number": hop_number,
                            "delta_index": delta_index,
                            "request_id": user_request_id,
                        },
                    )
                    await self.repo.add_event(delta_event)
                    delta_index += 1

                    # Stream to user immediately
                    logger.debug(
                        "→ Frontend: streaming content delta, length=%d", len(content)
                    )
                    yield ChatMessage(
                        type="text",
                        content=content,
                        metadata={"type": "delta", "hop": hop_number},
                    )

                # Handle tool calls from delta
                if delta.tool_calls:
                    # Accumulate tool call deltas properly
                    for tool_call_delta in delta.tool_calls:
                        self._accumulate_tool_call_delta(
                            current_tool_calls, tool_call_delta
                        )

                # Handle finish reason
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

            # Handle ThinkingChunk (reasoning models)
            elif isinstance(chunk, ThinkingChunk):
                # For now, we can log thinking content but don't stream it to user
                logger.debug("→ Thinking: %s", chunk.content[:100])

        logger.info(
            "← LLM: streaming completed (hop %d), finish_reason=%s",
            hop_number,
            finish_reason,
        )

        # Filter out incomplete tool calls before returning
        complete_tool_calls: list[dict[str, Any]] = []
        for call in current_tool_calls:
            if (
                call.get("id")
                and call.get("function", {}).get("name")
                and call.get("function", {}).get("arguments") is not None
            ):
                complete_tool_calls.append(call)

        # Provide user feedback when transitioning to tool execution phase
        if complete_tool_calls:
            tool_count = len(complete_tool_calls)
            logger.info(
                "→ Frontend: tool execution notification (%d tools)", tool_count
            )
            yield ChatMessage(
                type="tool_execution",
                content=f"Executing {tool_count} tool(s)...",
                metadata={"tool_count": tool_count, "hop": hop_number},
            )

        # Return complete assistant message for tool call processing
        yield {
            "content": message_buffer or None,
            "tool_calls": complete_tool_calls if complete_tool_calls else None,
            "finish_reason": finish_reason,
        }

    async def yield_existing_response(
        self, conversation_id: str, request_id: str
    ) -> AsyncGenerator[ChatMessage]:
        """
        Yield existing response content as ChatMessage for cached responses.

        This method retrieves and streams a previously computed assistant
        response when handling duplicate requests. It maintains consistency with
        the streaming interface by yielding ChatMessage objects even for cached
        content.

        Used in the idempotency flow when user message persistence returns
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
        logger.info(
            "→ Repository: retrieving cached response for request_id=%s", request_id
        )

        existing_response = await self.repo.get_existing_assistant_response(
            conversation_id, request_id
        )
        if existing_response and existing_response.content:
            content_str = (
                existing_response.content
                if isinstance(existing_response.content, str)
                else str(existing_response.content)
            )
            logger.info(
                "→ Frontend: streaming cached response, length=%d", len(content_str)
            )
            yield ChatMessage(
                type="text", content=content_str, metadata={"cached": True}
            )
        else:
            logger.warning("No cached response found for request_id=%s", request_id)

    def log_initial_response(self, assistant_msg: dict[str, Any]) -> None:
        """
        Log initial LLM response if configured in chat service settings.

        This method provides structured logging of the first LLM response
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
        reply_data: dict[str, Any] = {
            "message": assistant_msg,
            "model": self.llm_client.config.get("model", ""),
        }
        log_llm_reply(reply_data, "Streaming initial response", self.chat_conf)

    def _accumulate_tool_call_delta(
        self, current_tool_calls: list[dict[str, Any]], delta: ToolCallDelta
    ) -> None:
        """
        Accumulate tool call delta into the current tool calls list.

        This handles the incremental nature of streaming tool calls where
        each delta may contain partial information (id, function name, arguments)
        that needs to be accumulated into complete tool call objects.
        """
        # Get the delta as a dict for easier manipulation
        delta_dict = delta.model_dump() if hasattr(delta, "model_dump") else dict(delta)

        # Extract the index if available, otherwise assume it's a new tool call
        index = delta_dict.get("index", len(current_tool_calls))

        # Ensure we have enough slots in the list
        while len(current_tool_calls) <= index:
            current_tool_calls.append(
                {
                    "id": None,
                    "type": "function",
                    "function": {"name": None, "arguments": ""},
                }
            )

        # Get the current tool call at this index
        current_call: dict[str, Any] = current_tool_calls[index]  # type: ignore

        # Update fields from delta
        if delta_dict.get("id"):
            current_call["id"] = delta_dict["id"]

        if delta_dict.get("type"):
            current_call["type"] = delta_dict["type"]

        # Handle function delta
        if delta_dict.get("function"):
            function_delta = delta_dict["function"]

            if function_delta.get("name"):
                current_call["function"]["name"] = function_delta["name"]

            if function_delta.get("arguments"):
                # Accumulate arguments as they come in chunks
                current_call["function"]["arguments"] += function_delta["arguments"]
