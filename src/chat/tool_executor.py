"""
Tool Execution Handler

Handles complex/fragile tool-related operations:
- Tool call execution
- Tool result formatting
- MCP client interactions
- Tool call retry logic with hop limits

This module is prone to breaking when MCP servers change, so it's isolated
for better error handling and logging of MCP server interactions.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from mcp import types

from src.chat.logging_utils import (
    log_tool_args_error,
    log_tool_arguments,
    log_tool_execution_error,
    log_tool_execution_start,
    log_tool_execution_success,
)

if TYPE_CHECKING:
    from src.config import Configuration
    from src.tool_schema_manager import ToolSchemaManager

logger = logging.getLogger("src.clients.mcp_client")


class ToolExecutor:
    """Handles tool execution and MCP client interactions."""

    def __init__(
        self,
        tool_mgr: ToolSchemaManager,
        configuration: Configuration,
    ):
        self.tool_mgr = tool_mgr
        self.configuration = configuration

    async def execute_tool_calls(self, conv: list[dict[str, Any]], calls: list[dict[str, Any]]) -> None:
        """
        Execute tool calls and append results to conversation history.

        This method handles the execution phase of tool calls after they
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
        logger.info("→ MCP: executing %d tool calls", len(calls))

        for i, call in enumerate(calls):
            tool_name = call["function"]["name"]
            call_id = call["id"]

            # Parse JSON arguments with defensive handling for malformed JSON
            try:
                args: dict[str, Any] = json.loads(call["function"]["arguments"] or "{}")
                logger.debug("→ MCP[%s]: calling with args: %s", tool_name, args)
            except json.JSONDecodeError as e:
                log_tool_args_error(tool_name, e)
                args = {}

            try:
                # Log arguments being sent to MCP server
                mcp_config = self.configuration.get_mcp_logging_config()
                log_tool_arguments(
                    tool_name,
                    args,
                    f"call {i + 1}/{len(calls)}",
                    mcp_config.get("tool_arguments_truncate", 500),
                )

                # Execute tool through tool manager (handles validation and routing)
                log_tool_execution_start(tool_name, i, len(calls))
                result = await self.tool_mgr.call_tool(tool_name, args)

                # Extract readable content from MCP result structure
                content = self.pluck_content(result)
                log_tool_execution_success(tool_name, len(content))

                # Append tool result to conversation in OpenAI format
                conv.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": content,
                    }
                )

            except Exception as e:
                error_msg = f"Tool execution failed: {e!s}"
                log_tool_execution_error(tool_name, error_msg)

                # Still append an error result to maintain conversation flow
                conv.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": error_msg,
                    }
                )

        logger.info("← MCP: completed all tool executions")

    def check_tool_hop_limit(self, hops: int) -> tuple[bool, str | None]:
        """
        Check if tool call hop limit has been reached.

        Returns:
            tuple: (should_stop, warning_message)
        """
        max_tool_hops = self.configuration.get_max_tool_hops()
        if hops >= max_tool_hops:
            warning_msg = (
                f"⚠️ Reached maximum tool call limit ({max_tool_hops}). Stopping to prevent infinite recursion."
            )
            logger.warning("Maximum tool hops (%d) reached, stopping recursion", max_tool_hops)
            return True, warning_msg
        return False, None

    def pluck_content(self, res: types.CallToolResult) -> str:
        """
        Extract readable content from MCP CallToolResult for conversation context.

        This method handles the complex task of converting MCP tool results
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
                logger.warning("Failed to serialize structured content: %s", e)

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

        content = "\n".join(out)
        logger.debug("Extracted tool result content: %d characters", len(content))
        return content
