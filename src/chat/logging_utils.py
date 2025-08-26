"""
Enhanced Chat Logging Utilities

Shared logging functionality with feature control and performance optimizations.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)


def should_log_feature(module: str, feature: str) -> bool:
    """
    Check if a specific logging feature should be enabled.

    Uses cached feature flags for better performance during runtime.
    """
    if hasattr(logging, '_module_features'):
        module_features = getattr(logging, '_module_features', {}).get(module, {})
        return module_features.get(feature, False)
    return False


def log_llm_reply(
    reply: dict[str, Any], context: str, chat_conf: dict[str, Any]
) -> None:
    """
    Enhanced LLM reply logging with feature control and configuration-based truncation.

    Shared implementation to avoid duplication between streaming and
    non-streaming handlers.

    Args:
        reply: LLM API response containing message, model, and optional thinking
        context: Descriptive context for the log entry
        chat_conf: Chat configuration containing logging settings
    """
    # Use efficient feature checking instead of repeated config lookups
    if not should_log_feature("chat", "llm_replies"):
        return

    message = reply.get("message", {})
    content = message.get("content", "")
    tool_calls = message.get("tool_calls", [])
    thinking = reply.get("thinking", "")

    # Truncate content if configured
    truncate_length = chat_conf.get("logging", {}).get("llm_reply", 500)
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
            name = call.get("function", {}).get("name", "unknown")
            log_parts.append(f"  [{i}] {name}")

    model = reply.get("model", "unknown")
    log_parts.append(f"Model: {model}")

    logger.info(" | ".join(log_parts))


def log_tool_execution_start(
    tool_name: str, call_index: int = 0, total_calls: int = 1
) -> None:
    """
    Log the start of tool execution with consistent formatting.

    Args:
        tool_name: Name of the tool being executed
        call_index: Index of current call (0-based, used for batch execution)
        total_calls: Total number of calls in batch (for batch execution context)
    """
    if total_calls > 1:
        logger.info(
            "→ MCP[%s]: executing tool call %d/%d",
            tool_name,
            call_index + 1,
            total_calls,
        )
    else:
        logger.info("→ MCP[%s]: executing tool", tool_name)


def log_tool_execution_success(tool_name: str, content_length: int) -> None:
    """
    Log successful tool execution with content length.

    Args:
        tool_name: Name of the executed tool
        content_length: Length of the returned content
    """
    logger.info("← MCP[%s]: success, content length: %d", tool_name, content_length)


def log_tool_execution_error(tool_name: str, error_msg: str) -> None:
    """
    Log tool execution error with consistent formatting.

    Args:
        tool_name: Name of the tool that failed
        error_msg: Error message describing the failure
    """
    logger.error("← MCP[%s]: failed with error: %s", tool_name, error_msg)


def log_tool_args_error(tool_name: str, error: Exception) -> None:
    """
    Log malformed tool arguments warning.

    Args:
        tool_name: Name of the tool with malformed arguments
        error: The JSON decode or validation error
    """
    logger.error("Malformed JSON arguments for %s: %s", tool_name, error)


def log_directional_flow(
    direction: str, component: str, message: str, *args: Any
) -> None:
    """
    Log directional flow messages with consistent arrow formatting.

    Args:
        direction: Either "→" (outgoing) or "←" (incoming/completed)
        component: Component name (e.g., "LLM", "Repository", "Frontend", "MCP")
        message: Message template with optional format placeholders
        *args: Arguments for message formatting
    """
    formatted_msg = message % args if args else message
    logger.info(f"{direction} {component}: {formatted_msg}")


def log_error_with_context(context: str, error: Exception) -> None:
    """
    Log errors with consistent formatting across the application.

    Args:
        context: Descriptive context of where the error occurred
        error: The exception that was raised
    """
    logger.error(f"Error {context}: {error}")


@asynccontextmanager
async def log_performance(operation_name: str):
    """Context manager to log performance metrics for operations."""
    start_time = time.monotonic()
    try:
        yield
    finally:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(f"⏱️ {operation_name} completed in {elapsed_ms:.2f}ms")


def log_llm_request_start(request_id: str, provider: str, model: str) -> float:
    """Log the start of an LLM request and return start time."""
    start_time = time.monotonic()
    logger.info(f"🚀 LLM request started: request_id={request_id}, provider={provider}, model={model}")
    return start_time


def log_llm_request_complete(request_id: str, start_time: float, success: bool = True):
    """Log the completion of an LLM request with timing."""
    elapsed_ms = (time.monotonic() - start_time) * 1000
    status = "✅" if success else "❌"
    logger.info(f"{status} LLM request completed: request_id={request_id}, elapsed={elapsed_ms:.2f}ms")


def log_connection_pool_stats(active_connections: int, available_connections: int):
    """Log connection pool statistics."""
    logger.debug(f"🔌 Connection pool: active={active_connections}, available={available_connections}")


def log_tool_arguments(
    tool_name: str, arguments: dict[str, Any], context: str, truncate_length: int = 500
) -> None:
    """
    Log tool arguments being sent to MCP server.

    Args:
        tool_name: Name of the tool being called
        arguments: Arguments dictionary being sent to the tool
        context: Descriptive context for the log entry
        truncate_length: Maximum length for argument logging
    """
    if not should_log_feature("mcp", "tool_arguments"):
        logger.debug(f"Tool arguments logging disabled for {tool_name}")
        return

    # Convert arguments to string representation
    args_str = str(arguments)

    # Truncate if necessary
    if len(args_str) > truncate_length:
        args_str = args_str[:truncate_length] + "..."

    logger.info(f"→ MCP[{tool_name}]: arguments ({context}): {args_str}")


def log_tool_results(
    tool_name: str, results: Any, context: str, truncate_length: int = 200
) -> None:
    """
    Log tool results received from MCP server.

    Args:
        tool_name: Name of the tool that was called
        results: Results received from the tool execution
        context: Descriptive context for the log entry
        truncate_length: Maximum length for results logging
    """
    if not should_log_feature("mcp", "tool_results"):
        return

    # Convert results to string representation
    results_str = str(results)

    # Truncate if necessary
    if len(results_str) > truncate_length:
        results_str = results_str[:truncate_length] + "..."

    logger.info(f"← MCP[{tool_name}]: results ({context}): {results_str}")
