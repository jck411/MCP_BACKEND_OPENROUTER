"""
Simple Chat Handler

Mundane/stable non-streaming chat operations:
- Basic request/response flow
- No delta complexity
- Simple tool call execution

This module keeps the simple case simple. Code duplication with streaming
is acceptable since they have different complexity levels.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from mcp import types

from src.chat.logging_utils import (
    log_llm_reply,
    log_tool_args_error,
    log_tool_execution_start,
    log_tool_execution_success,
)
from src.history import ChatEvent

from .models import (
    AssistantMessage,
    ConversationHistory,
    SystemMessage,
    ToolDefinition,
    ToolMessage,
    UserMessage,
)

if TYPE_CHECKING:
    from src.chat.resource_loader import ResourceLoader
    from src.chat.tool_executor import ToolExecutor
    from src.clients import LLMClient
    from src.history.repository import ChatRepository

logger = logging.getLogger(__name__)


class SimpleChatHandler:
    """Handles non-streaming chat operations."""

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

    async def chat_once(
        self,
        conversation_id: str,
        user_msg: str,
        request_id: str,
        tools_payload: list[ToolDefinition],
    ) -> ChatEvent:
        """
        Non-streaming chat with consistent history management.

        Flow:
        1. Persist user message first (with idempotency check)
        2. Build history from repository
        3. Generate response with tools
        4. Persist final assistant message
        """
        logger.info("Starting non-streaming chat for request_id=%s", request_id)

        # Check for existing response to prevent double-billing
        existing_response = await self.repo.get_existing_assistant_response(
            conversation_id, request_id
        )
        if existing_response:
            logger.debug("Found cached response, returning without invoking LLM")
            return existing_response

        # Handle user message persistence (handles idempotency internally)
        should_continue = await self.repo.handle_user_message_persistence(
            conversation_id, user_msg, request_id
        )
        if not should_continue:
            # Race condition: duplicate request resolved after we checked
            existing_response = await self.repo.get_existing_assistant_response(
                conversation_id, request_id
            )
            if existing_response:
                logger.debug("Race condition resolved, returning cached response")
                return existing_response

        # Build conversation history
        logger.debug("Building conversation history for request_id=%s", request_id)
        system_prompt = await self.resource_loader.make_system_prompt()
        conv_dict = await self.repo.build_llm_conversation(
            conversation_id, user_msg, system_prompt
        )

        # Convert to typed ConversationHistory
        conv = self._convert_to_conversation_history(conv_dict)

        # Generate response
        content, model = await self.generate_assistant_response(
            conv, tools_payload, conversation_id=conversation_id, request_id=request_id
        )

        # Persist assistant message
        return await self.repo.persist_assistant_message(
            conversation_id,
            request_id,
            content,
            model,
            self.llm_client.provider,
        )

    async def generate_assistant_response(
        self,
        conv: ConversationHistory,
        tools_payload: list[ToolDefinition],
        conversation_id: str | None = None,
        request_id: str | None = None,
    ) -> tuple[str, str]:
        """Generate assistant response using tools if needed."""
        logger.info("→ LLM: requesting non-streaming response")

        assistant_full_text = ""
        model = ""

        reply = await self.llm_client.get_response_with_tools(
            conv.get_api_format(), tools_payload
        )
        assistant_msg = reply.message

        # Log LLM reply if configured
        log_llm_reply(reply.model_dump(), "Initial LLM response", self.chat_conf)

        # Store model from first API call
        model = reply.model

        if txt := assistant_msg.content:
            assistant_full_text += txt

        # Handle tool call iterations
        hops = 0
        while calls := assistant_msg.tool_calls:
            should_stop, warning_msg = self.tool_executor.check_tool_hop_limit(hops)
            if should_stop and warning_msg:
                logger.info("Tool hop limit reached, appending warning")
                assistant_full_text += "\n\n" + warning_msg
                break

            logger.info("Starting tool execution iteration %d", hops + 1)

            conv.add_message(
                AssistantMessage(
                    role="assistant",
                    content=assistant_msg.content or "",
                    tool_calls=calls,
                )
            )

            # Execute tools sequentially and persist tool results
            for i, call in enumerate(calls):
                tool_name = call.function.name
                try:
                    args: dict[str, Any] = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError as e:
                    log_tool_args_error(tool_name, e)
                    args = {}

                log_tool_execution_start(tool_name)
                result: types.CallToolResult = (
                    await self.tool_executor.tool_mgr.call_tool(tool_name, args)
                )
                content: str = self.tool_executor.pluck_content(result)
                log_tool_execution_success(tool_name, len(content))

                conv.add_message(
                    ToolMessage(
                        role="tool",
                        tool_call_id=call.id,
                        content=content,
                    )
                )

                # Persist tool_result so it becomes part of future context
                if conversation_id:
                    req_id = f"tool:{request_id or call.id}:{hops}:{i}"
                    await self.repo.add_event(
                        ChatEvent(
                            conversation_id=conversation_id,
                            seq=0,
                            type="tool_result",
                            role="tool",
                            content=content,
                            extra={
                                "tool_call_id": call.id,
                                "request_id": req_id,
                            },
                        )
                    )

            # Get follow-up response
            logger.info("→ LLM: requesting follow-up response (hop %d)", hops + 1)
            reply = await self.llm_client.get_response_with_tools(
                conv.get_api_format(), tools_payload
            )
            assistant_msg = reply.message

            # Log LLM reply if configured
            log_llm_reply(
                reply.model_dump(),
                f"Tool call follow-up response (hop {hops + 1})",
                self.chat_conf,
            )

            if txt := assistant_msg.content:
                assistant_full_text += txt

            hops += 1
            logger.info("Completed tool execution iteration %d", hops)

        logger.info("← LLM: response generation completed")
        return assistant_full_text, model

    def _convert_to_conversation_history(
        self, conv_dict: list[dict[str, Any]]
    ) -> ConversationHistory:
        """Convert dictionary conversation to typed ConversationHistory."""
        history = ConversationHistory()

        for msg_dict in conv_dict:
            role = msg_dict.get("role")
            if role == "system":
                history.system_prompt = SystemMessage(
                    role="system", content=msg_dict.get("content", "")
                )
            elif role == "user":
                history.add_message(
                    UserMessage(role="user", content=msg_dict.get("content", ""))
                )
            elif role == "assistant":
                history.add_message(
                    AssistantMessage(
                        role="assistant",
                        content=msg_dict.get("content", ""),
                        tool_calls=msg_dict.get("tool_calls", []),
                    )
                )
            elif role == "tool":
                history.add_message(
                    ToolMessage(
                        role="tool",
                        tool_call_id=msg_dict.get("tool_call_id", ""),
                        content=msg_dict.get("content", ""),
                    )
                )

        return history
