"""
Chat Orchestrator

Main coordination layer for the modular chat system.
This is a thin orchestration layer that delegates to appropriate handlers
based on the operation type (streaming vs non-streaming).

Keeps the main class simple, just coordinating between modules.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from src.clients.mcp_client import MCPClient
from src.config import Configuration
from src.history import ChatEvent
from src.tool_schema_manager import ToolSchemaManager

from .models import (
    AssistantMessage,
    ChatMessage,
    ConversationHistory,
    SystemMessage,
    ToolDefinition,
    ToolMessage,
    UserMessage,
)
from .resource_loader import ResourceLoader
from .simple_chat_handler import SimpleChatHandler
from .streaming_handler import StreamingHandler
from .tool_executor import ToolExecutor

if TYPE_CHECKING:
    from src.clients.llm_client import LLMClient

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    """
    Conversation orchestrator - coordinates between specialized handlers
    1. Takes your message
    2. Figures out what tools might be needed
    3. Delegates to streaming or non-streaming handler
    4. Sends you back the response
    """

    class ChatOrchestratorConfig(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        clients: list[MCPClient]
        llm_client: "LLMClient"
        config: dict[str, Any]
        repo: Any  # ChatRepository protocol - use Any to avoid pydantic issues
        configuration: Configuration
        ctx_window: int = 4000

    def __init__(
        self,
        service_config: ChatOrchestratorConfig,
    ):
        self.clients = service_config.clients
        self.llm_client = service_config.llm_client
        self.config = service_config.config
        self.repo = service_config.repo
        self.configuration = service_config.configuration
        self._ctx_window = service_config.ctx_window
        self.chat_conf = self.config.get("chat", {}).get("service", {})

        # Core components
        self.tool_mgr: ToolSchemaManager | None = None
        self.resource_loader: ResourceLoader | None = None
        self.tool_executor: ToolExecutor | None = None
        self.streaming_handler: StreamingHandler | None = None
        self.simple_chat_handler: SimpleChatHandler | None = None

        # Initialization state
        self._init_lock = asyncio.Lock()
        self._ready = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize the chat orchestrator and all components."""
        async with self._init_lock:
            if self._ready.is_set():
                logger.debug("Chat orchestrator already initialized")
                return

            logger.info("→ Orchestrator: initializing chat orchestrator")

            # Connect to all clients and collect results
            logger.info("→ Orchestrator: connecting to MCP clients")
            # Use semaphore to limit concurrent connection attempts and avoid overwhelming system
            connection_semaphore = asyncio.Semaphore(5)  # Max 5 concurrent connections

            async def connect_with_semaphore(client):
                async with connection_semaphore:
                    return await client.connect()

            connection_results = await asyncio.gather(
                *(connect_with_semaphore(c) for c in self.clients),
                return_exceptions=True,
            )

            # Filter out only successfully connected clients
            connected_clients: list[MCPClient] = []
            for i, result in enumerate(connection_results):
                if isinstance(result, Exception):
                    logger.warning(
                        "Client '%s' failed to connect: %s",
                        self.clients[i].name,
                        result,
                    )
                else:
                    connected_clients.append(self.clients[i])

            if not connected_clients:
                logger.warning("No MCP clients connected - running with basic functionality")
            else:
                logger.info(
                    "← Orchestrator: connected to %d out of %d MCP clients",
                    len(connected_clients),
                    len(self.clients),
                )

            # Initialize tool manager (empty list is acceptable)
            logger.info("→ Orchestrator: initializing tool manager")
            self.tool_mgr = ToolSchemaManager(connected_clients)
            await self.tool_mgr.initialize()
            logger.info("← Orchestrator: tool manager ready")

            # Initialize resource loader and get system prompt
            logger.info("→ Orchestrator: initializing resource loader")
            self.resource_loader = ResourceLoader(self.tool_mgr, self.configuration)
            system_prompt = await self.resource_loader.initialize()
            logger.info("← Orchestrator: resource loader ready")

            # Initialize tool executor
            logger.info("→ Orchestrator: initializing tool executor")
            self.tool_executor = ToolExecutor(self.tool_mgr, self.configuration)
            logger.info("← Orchestrator: tool executor ready")

            # Initialize handlers
            logger.info("→ Orchestrator: initializing handlers")
            self.streaming_handler = StreamingHandler(
                self.llm_client,
                self.tool_executor,
                self.repo,
                self.resource_loader,
                self.chat_conf,
            )
            self.simple_chat_handler = SimpleChatHandler(
                self.llm_client,
                self.tool_executor,
                self.repo,
                self.resource_loader,
                self.chat_conf,
            )
            logger.info("← Orchestrator: handlers ready")

            # Log initialization summary
            logger.info(
                "← Orchestrator: ready - %d tools, %d resources, %d prompts",
                len(self.tool_mgr.get_openai_tools()),
                self.resource_loader.get_resource_count(),
                len(self.tool_mgr.list_available_prompts()),
            )

            # Configurable system prompt logging
            if self.chat_conf.get("logging", {}).get("system_prompt", True):
                logger.info("System prompt being used:\n%s", system_prompt)
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
        Process a user message with streaming response.

        This is the main entry point for streaming chat interactions.
        Delegates to streaming handler after initialization and validation.
        """
        await self._ready.wait()

        # Validate components are ready
        if not all(
            [
                self.streaming_handler,
                self.tool_executor,
                self.tool_mgr,
            ]
        ):
            raise RuntimeError("Chat orchestrator components not properly initialized")

        # Type assertions for mypy/pylance
        assert self.streaming_handler is not None
        assert self.resource_loader is not None
        assert self.tool_executor is not None
        assert self.tool_mgr is not None

        # Validate streaming support
        self.streaming_handler.validate_streaming_support()

        logger.info("→ Orchestrator: processing streaming message for request_id=%s", request_id)

        # Handle idempotency and user message persistence
        should_continue = await self.repo.handle_user_message_persistence(conversation_id, user_msg, request_id)
        if not should_continue:
            logger.info("→ Orchestrator: returning cached response")
            async for msg in self.streaming_handler.yield_existing_response(conversation_id, request_id):
                yield msg
            return

        # Build conversation and generate response
        system_prompt = await self.resource_loader.make_system_prompt()
        conv_dict = await self.repo.build_llm_conversation(conversation_id, user_msg, system_prompt)

        # Convert to typed ConversationHistory for efficiency
        conv = self._convert_to_conversation_history(conv_dict)

        tools_payload = self.tool_mgr.get_openai_tools()

        # Convert to typed ToolDefinition list
        typed_tools = [ToolDefinition.model_validate(tool) for tool in tools_payload]

        # Stream and handle tool calls
        async for msg in self.streaming_handler.stream_and_handle_tools(conv, typed_tools, conversation_id, request_id):
            yield msg

        logger.info("← Orchestrator: completed streaming message processing")

    async def chat_once(
        self,
        conversation_id: str,
        user_msg: str,
        request_id: str,
    ) -> ChatEvent:
        """
        Non-streaming chat with consistent history management.

        This is the main entry point for simple, non-streaming chat interactions.
        Delegates to simple chat handler after initialization and validation.
        """
        await self._ready.wait()

        # Validate components are ready
        if not all(
            [
                self.simple_chat_handler,
                self.tool_executor,
                self.tool_mgr,
            ]
        ):
            raise RuntimeError("Chat orchestrator components not properly initialized")

        # Type assertions for mypy/pylance
        assert self.simple_chat_handler is not None
        assert self.resource_loader is not None
        assert self.tool_executor is not None
        assert self.tool_mgr is not None

        logger.info(
            "→ Orchestrator: processing non-streaming chat for request_id=%s",
            request_id,
        )

        tools_payload = self.tool_mgr.get_openai_tools()

        # Convert to typed ToolDefinition list
        typed_tools = [ToolDefinition.model_validate(tool) for tool in tools_payload]

        # Delegate to simple chat handler
        result = await self.simple_chat_handler.chat_once(conversation_id, user_msg, request_id, typed_tools)

        logger.info("← Orchestrator: completed non-streaming chat processing")
        return result

    async def apply_prompt(self, name: str, args: dict[str, str]) -> list[dict[str, str]]:
        """Apply a parameterized prompt and return conversation messages."""
        await self._ready.wait()

        if not self.resource_loader:
            raise RuntimeError("Resource loader not initialized")

        logger.info("→ Orchestrator: applying prompt '%s'", name)
        result: list[dict[str, str]] = await self.resource_loader.apply_prompt(name, args)
        logger.info("← Orchestrator: prompt applied successfully")
        return result

    async def cleanup(self) -> None:
        """Clean up resources by closing all connected MCP clients and LLM client."""
        logger.info("→ Orchestrator: starting cleanup")

        if not self.tool_mgr:
            logger.warning("Tool manager not initialized during cleanup")
            return

        # Get connected clients from tool manager
        for client in self.tool_mgr.clients:
            try:
                await client.close()
                logger.debug("Closed MCP client: %s", client.name)
            except Exception as e:
                logger.warning("Error closing client %s: %s", client.name, e)

        # Close LLM HTTP client to prevent dangling connections
        try:
            await self.llm_client.close()
            logger.info("LLM client closed successfully")
        except Exception as e:
            logger.warning("Error closing LLM client: %s", e)

        logger.info("← Orchestrator: cleanup completed")

    # Properties for compatibility with existing code
    @property
    def ctx_window(self) -> int:
        """Context window size."""
        return self._ctx_window

    def get_tool_count(self) -> int:
        """Get number of available tools."""
        if not self.tool_mgr:
            return 0
        return len(self.tool_mgr.get_openai_tools())

    def get_resource_count(self) -> int:
        """Get number of available resources."""
        if not self.resource_loader:
            return 0
        return self.resource_loader.get_resource_count()

    def get_prompt_count(self) -> int:
        """Get number of available prompts."""
        if not self.tool_mgr:
            return 0
        return len(self.tool_mgr.list_available_prompts())

    def _convert_to_conversation_history(self, conv_dict: list[dict[str, Any]]) -> ConversationHistory:
        """Convert dictionary conversation to typed ConversationHistory."""
        history = ConversationHistory()

        for msg_dict in conv_dict:
            role = msg_dict.get("role")
            if role == "system":
                history.system_prompt = SystemMessage(role="system", content=msg_dict.get("content", ""))
            elif role == "user":
                history.add_message(UserMessage(role="user", content=msg_dict.get("content", "")))
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
