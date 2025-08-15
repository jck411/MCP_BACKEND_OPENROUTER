"""
Simple Chat Handler

Handles mundane/stable non-streaming chat operations:
- Basic request/response flow
- No delta complexity
- Simple tool call execution

This module keeps the simple case simple. Code duplication with streaming
is acceptable since they have different complexity levels.
"""

import json
import logging
from typing import Any

from src.history import ChatEvent

logger = logging.getLogger(__name__)


class SimpleChatHandler:
    """Handles non-streaming chat operations."""

    def __init__(
        self,
        llm_client,
        tool_executor,
        repo,
        resource_loader,
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
        tools_payload: list[dict[str, Any]],
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
        conv = await self.repo.build_llm_conversation(
            conversation_id, user_msg, system_prompt
        )

        # Generate response
        response = await self._generate_response_with_tools(conv, tools_payload)

        # Persist assistant message
        assistant_ev = await self.repo.persist_assistant_message(
            conversation_id,
            request_id,
            response["content"],
            response["model"],
            response.get("provider", "unknown"),
        )

    async def generate_assistant_response(
        self, conv: list[dict[str, Any]], tools_payload: list[dict[str, Any]]
    ) -> tuple[str, str]:
        """Generate assistant response using tools if needed."""
        logger.info("→ LLM: requesting non-streaming response")

        assistant_full_text = ""
        model = ""

        reply = await self.llm_client.get_response_with_tools(conv, tools_payload)
        assistant_msg = reply["message"]

        # Log LLM reply if configured
        self.log_llm_reply(reply, "Initial LLM response")

        # Store model from first API call
        model = reply.get("model", "")

        if txt := assistant_msg.get("content"):
            assistant_full_text += txt

        # Handle tool call iterations
        hops = 0
        while calls := assistant_msg.get("tool_calls"):
            should_stop, warning_msg = self.tool_executor.check_tool_hop_limit(hops)
            if should_stop:
                logger.info("Tool hop limit reached, appending warning")
                assistant_full_text += "\n\n" + warning_msg
                break

            logger.info("Starting tool execution iteration %d", hops + 1)

            conv.append(
                {
                    "role": "assistant",
                    "content": assistant_msg.get("content") or "",
                    "tool_calls": calls,
                }
            )

            # Execute tools sequentially
            for call in calls:
                tool_name = call["function"]["name"]
                try:
                    args = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError as e:
                    logger.warning("Malformed JSON arguments for %s: %s", tool_name, e)
                    args = {}

                logger.info("→ MCP[%s]: executing tool", tool_name)
                result = await self.tool_executor.tool_mgr.call_tool(tool_name, args)
                content = self.tool_executor.pluck_content(result)
                logger.info(
                    "← MCP[%s]: success, content length: %d", tool_name, len(content)
                )

                conv.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": content,
                    }
                )

            # Get follow-up response
            logger.info("→ LLM: requesting follow-up response (hop %d)", hops + 1)
            reply = await self.llm_client.get_response_with_tools(conv, tools_payload)
            assistant_msg = reply["message"]

            # Log LLM reply if configured
            self.log_llm_reply(reply, f"Tool call follow-up response (hop {hops + 1})")

            if txt := assistant_msg.get("content"):
                assistant_full_text += txt

            hops += 1
            logger.info("Completed tool execution iteration %d", hops)

        logger.info("← LLM: response generation completed")
        return assistant_full_text, model

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
                log_parts.append(f"  - Tool {i + 1}: {func_name}")

        model = reply.get("model", "unknown")
        log_parts.append(f"Model: {model}")

        logger.info(" | ".join(log_parts))
