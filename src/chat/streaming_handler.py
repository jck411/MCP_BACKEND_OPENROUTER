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

import logging
from collections.abc import AsyncGenerator
from typing import Any

from src.chat.models import ChatMessage, ToolCallContext, convert_usage
from src.history import ChatEvent

logger = logging.getLogger(__name__)


class StreamingHandler:
    """Handles streaming responses and tool call iterations."""

    def __init__(
        self, 
        llm_client, 
        tool_executor, 
        conversation_manager,
        repo,
        chat_conf: dict[str, Any]
    ):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.conversation_manager = conversation_manager
        self.repo = repo
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
        if not hasattr(self.llm_client, 'get_streaming_response_with_tools'):
            raise RuntimeError(
                "LLM client does not support streaming. "
                "Use chat_once() for non-streaming responses."
            )

    async def stream_and_handle_tools(
        self,
        conv: list[dict[str, Any]],
        tools_payload: list[dict[str, Any]],
        conversation_id: str,
        request_id: str,
    ) -> AsyncGenerator[ChatMessage]:
        """Stream response and handle tool calls iteratively."""
        logger.info("→ Frontend: starting streaming response for request_id=%s", request_id)
        
        full_content = ""

        # Initial response streaming
        assistant_msg: dict[str, Any] = {}
        async for chunk in self.stream_llm_response_with_deltas(
            conv, tools_payload, conversation_id, request_id
        ):
            if isinstance(chunk, ChatMessage):
                if chunk.type == "text":
                    full_content += chunk.content
                logger.debug("→ Frontend: text delta, length=%d", len(chunk.content))
                yield chunk
            else:
                assistant_msg = chunk
                if assistant_msg.get("content"):
                    full_content += assistant_msg["content"]

        self.log_initial_response(assistant_msg)

        # Handle tool call iterations
        context = ToolCallContext(
            conv=conv,
            tools_payload=tools_payload,
            conversation_id=conversation_id,
            request_id=request_id,
            assistant_msg=assistant_msg,
            full_content=full_content
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
            usage=convert_usage(None),
            model=self.llm_client.config.get("model", "")
        )
        logger.info("← Repository: delta compaction completed")
        logger.info("← Frontend: streaming response completed for request_id=%s", request_id)

    async def handle_tool_call_iterations(
        self, context: ToolCallContext
    ) -> AsyncGenerator[ChatMessage | str]:
        """Handle iterative tool calls with hop limit."""
        hops = 0

        while context.assistant_msg.get("tool_calls"):
            should_stop, warning_msg = self.tool_executor.check_tool_hop_limit(hops)
            if should_stop:
                context.full_content += "\n\n" + warning_msg
                logger.info("→ Frontend: tool hop limit warning")
                yield ChatMessage(
                    type="text",
                    content=warning_msg,
                    metadata={"finish_reason": "tool_limit_reached"}
                )
                break

            logger.info("Starting tool call iteration %d", hops + 1)
            
            # Execute tool calls
            context.conv.append({
                "role": "assistant",
                "content": context.assistant_msg.get("content") or "",
                "tool_calls": context.assistant_msg["tool_calls"],
            })
            await self.tool_executor.execute_tool_calls(
                context.conv, context.assistant_msg["tool_calls"]
            )

            # Get follow-up response
            logger.info("→ LLM: requesting follow-up response for hop %d", hops + 1)
            context.assistant_msg = {}
            async for chunk in self.stream_llm_response_with_deltas(
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
            logger.info("Completed tool call iteration %d", hops)

        yield context.full_content  # Return updated full_content

    async def stream_llm_response_with_deltas(
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
        logger.info("→ LLM: starting streaming request (hop %d)", hop_number)
        
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
                logger.debug("→ Frontend: streaming content delta, length=%d", len(content))
                yield ChatMessage(
                    type="text",
                    content=content,
                    metadata={"type": "delta", "hop": hop_number}
                )

            # PRIORITY 2: Accumulate tool calls for batch execution
            # Tool calls need to be complete before execution, so we buffer them
            if tool_calls := delta.get("tool_calls"):
                # Use robust accumulation that handles out-of-order and partial deltas
                self.tool_executor.accumulate_tool_calls(current_tool_calls, tool_calls)

            # PRIORITY 3: Track completion status for proper flow control
            if "finish_reason" in choice:
                finish_reason = choice["finish_reason"]

        logger.info("← LLM: streaming completed (hop %d), finish_reason=%s", 
                   hop_number, finish_reason)

        # Provide user feedback when transitioning to tool execution phase
        # This helps users understand what's happening during longer operations
        if current_tool_calls and any(
            call["function"]["name"] for call in current_tool_calls
        ):
            tool_count = len(current_tool_calls)
            logger.info("→ Frontend: tool execution notification (%d tools)", tool_count)
            yield ChatMessage(
                type="tool_execution",
                content=f"Executing {tool_count} tool(s)...",
                metadata={"tool_count": tool_count, "hop": hop_number}
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
        logger.info("→ Repository: retrieving cached response for request_id=%s", request_id)
        
        existing_response = await self.conversation_manager.get_existing_assistant_response(
            conversation_id, request_id
        )
        if existing_response and existing_response.content:
            content_str = (
                existing_response.content
                if isinstance(existing_response.content, str)
                else str(existing_response.content)
            )
            logger.info("→ Frontend: streaming cached response, length=%d", len(content_str))
            yield ChatMessage(
                type="text",
                content=content_str,
                metadata={"cached": True}
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
        reply_data = {
            "message": assistant_msg,
            "usage": None,
            "model": self.llm_client.config.get("model", "")
        }
        self.log_llm_reply(reply_data, "Streaming initial response")

    def log_llm_reply(self, reply: dict[str, Any], context: str) -> None:
        """Log LLM reply if configured, including thinking content for reasoning models."""
        logging_config = self.chat_conf.get("logging", {})
        if not logging_config.get("llm_replies", False):
            return

        message = reply.get("message", {})
        content = message.get("content", "")
        tool_calls = message.get("tool_calls", [])
        thinking = reply.get("thinking", "")

        # Truncate content if configured
        truncate_length = logging_config.get("llm_reply_truncate_length", 500)
        if content and len(content) > truncate_length:
            content = content[:truncate_length] + "..."

        # Truncate thinking content if present
        if thinking and len(thinking) > truncate_length:
            thinking = thinking[:truncate_length] + "..."

        log_parts = [f"LLM Reply ({context}):"]

        # Log thinking content first for reasoning models
        if thinking:
            log_parts.append(f"Thinking: {thinking}")

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
