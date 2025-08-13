"""Lean Tool Schema Manager for MCP Platform

This module provides a lightweight registry for MCP servers that:
- Discovers tools/prompts/resources via the MCP SDK
- Emits OpenAI-compatible tool definitions on demand (minimal wrapper)
- Calls tools through the MCP SDK (server-side validation only)

Design goals:
- No client-side schema conversion beyond the minimal OpenAI wrapper
- No client-side parameter validation (rely on server-side validation)
- Keep MCP-native benefits (discovery, prompts, resources, error semantics)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mcp import McpError, types

if TYPE_CHECKING:
    from src.mcp_client import MCPClient

logger = logging.getLogger(__name__)


class ToolSchemaManager:
    """
    Manages MCP-discovered entities and produces OpenAI tool definitions on demand.

    Key characteristics:
    - Lean: does not create Pydantic models or transform schemas
    - Minimal OpenAI wrapper: {"type": "function", "function": {name, description,
      parameters}}
    - Conflict-safe: prefixes tool names with client name when conflicts occur
    - Adheres to MCP: retains full MCP metadata and calls via the MCP SDK
    """

    def __init__(self, clients: list[MCPClient]) -> None:
        """Initialize the tool schema manager with MCP clients."""
        self.clients = clients
        self._tool_registry: dict[str, ToolInfo] = {}
        self._prompt_registry: dict[str, PromptInfo] = {}
        self._resource_registry: dict[str, ResourceInfo] = {}

    async def initialize(self) -> None:
        """Initialize registries by collecting tools, prompts, and resources."""
        self._tool_registry.clear()
        self._prompt_registry.clear()
        self._resource_registry.clear()

        for client in self.clients:
            await self._register_client_tools(client)
            await self._register_client_prompts(client)
            await self._register_client_resources(client)

        logger.info(
            f"Initialized registries with {len(self._tool_registry)} tools, "
            f"{len(self._prompt_registry)} prompts, "
            f"{len(self._resource_registry)} resources"
        )

    async def _register_client_tools(self, client: MCPClient) -> None:
        """
        Register all tools from a specific MCP client with conflict resolution.

        - Skips disconnected clients
        - Handles name conflicts by prefixing with client name
        - Stores ToolInfo with original MCP tool metadata (no schema transformation)
        """
        if not client.is_connected:
            logger.warning(
                f"Skipping tool registration for disconnected client '{client.name}'"
            )
            return

        try:
            tools = await client.list_tools()
            for tool in tools:
                registry_name = tool.name

                # Handle name conflicts by prefixing with client name
                if registry_name in self._tool_registry:
                    logger.warning(
                        f"Tool name conflict: '{registry_name}' already exists"
                    )
                    registry_name = f"{client.name}_{registry_name}"

                # Store complete tool information (no conversion)
                self._tool_registry[registry_name] = ToolInfo(tool, client)

            logger.info(f"Registered {len(tools)} tools from client '{client.name}'")
        except Exception as e:
            logger.error(f"Error registering tools from client '{client.name}': {e}")
            raise

    async def _register_client_prompts(self, client: MCPClient) -> None:
        """Register prompts from a specific MCP client."""
        if not client.is_connected:
            logger.warning(
                f"Skipping prompt registration for disconnected client '{client.name}'"
            )
            return

        try:
            prompts = await client.list_prompts()
            for prompt in prompts:
                prompt_name = prompt.name
                if prompt_name in self._prompt_registry:
                    logger.warning(
                        f"Prompt name conflict: '{prompt_name}' already exists"
                    )
                    prompt_name = f"{client.name}_{prompt_name}"

                prompt_info = PromptInfo(prompt, client)
                self._prompt_registry[prompt_name] = prompt_info

            logger.info(
                f"Registered {len(prompts)} prompts from client '{client.name}'"
            )
        except Exception as e:
            logger.error(f"Error registering prompts from client '{client.name}': {e}")
            raise

    async def _register_client_resources(self, client: MCPClient) -> None:
        """Register resources from a specific MCP client."""
        if not client.is_connected:
            logger.warning(
                f"Skipping resource registration for disconnected "
                f"client '{client.name}'"
            )
            return

        try:
            resources = await client.list_resources()
            for resource in resources:
                resource_uri = str(resource.uri)
                if resource_uri in self._resource_registry:
                    logger.warning(
                        f"Resource URI conflict: '{resource_uri}' already exists"
                    )
                    resource_uri = f"{client.name}::{resource_uri}"

                resource_info = ResourceInfo(resource, client)
                self._resource_registry[resource_uri] = resource_info

            logger.info(
                f"Registered {len(resources)} resources from client '{client.name}'"
            )
        except Exception as e:
            logger.error(
                f"Error registering resources from client '{client.name}': {e}"
            )
            raise

    def _to_openai_tool(self, registry_name: str, tool: types.Tool) -> dict[str, Any]:
        """
        Produce a minimal OpenAI tool wrapper for an MCP tool.

        Notes:
        - Uses the resolved registry name for the function.name so LLM calls map
          back 1:1
        - Sends only fields OpenAI cares about (name, description, parameters)
        - Relies on MCP's inputSchema (JSON Schema) as-is
        """
        if not tool.inputSchema:
            raise ValueError(f"Tool {tool.name} has no input schema")

        return {
            "type": "function",
            "function": {
                "name": registry_name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        }

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Build the OpenAI tools list on demand from the current registry."""
        tools: list[dict[str, Any]] = []
        for registry_name, info in self._tool_registry.items():
            try:
                tools.append(self._to_openai_tool(registry_name, info.tool))
            except Exception as e:
                logger.error(
                    f"Skipping tool '{registry_name}' due to schema error: {e}"
                )
        return tools

    def get_tool_info(self, tool_name: str) -> ToolInfo:
        """Get detailed information about a specific tool by registry name."""
        tool_info = self._tool_registry.get(tool_name)
        if not tool_info:
            raise McpError(
                error=types.ErrorData(
                    code=types.INVALID_PARAMS,
                    message=f"Tool '{tool_name}' not found",
                )
            )
        return tool_info

    def get_prompt_info(self, prompt_name: str) -> PromptInfo:
        """Get detailed information about a specific prompt."""
        prompt_info = self._prompt_registry.get(prompt_name)
        if not prompt_info:
            raise McpError(
                error=types.ErrorData(
                    code=types.INVALID_PARAMS,
                    message=f"Prompt '{prompt_name}' not found",
                )
            )
        return prompt_info

    def get_resource_info(self, resource_uri: str) -> ResourceInfo:
        """Get detailed information about a specific resource."""
        resource_info = self._resource_registry.get(resource_uri)
        if not resource_info:
            raise McpError(
                error=types.ErrorData(
                    code=types.INVALID_PARAMS,
                    message=f"Resource '{resource_uri}' not found",
                )
            )
        return resource_info

    def list_available_prompts(self) -> list[str]:
        """Get list of all available prompt names."""
        return list(self._prompt_registry.keys())

    def list_available_resources(self) -> list[str]:
        """Get list of all available resource URIs."""
        return list(self._resource_registry.keys())

    async def call_tool(
        self, tool_name: str, parameters: dict[str, Any]
    ) -> types.CallToolResult:
        """
        Call a tool with raw parameters.

        No client-side validation is performed. We rely on the MCP server for
        schema validation and error semantics.
        """
        tool_info = self.get_tool_info(tool_name)
        # tool_info.tool.name is the original (server-side) name
        return await tool_info.client.call_tool(tool_info.tool.name, parameters)

    async def get_prompt(
        self, prompt_name: str, arguments: dict[str, Any] | None = None
    ) -> types.GetPromptResult:
        """Get a prompt by name with optional arguments."""
        prompt_info = self.get_prompt_info(prompt_name)
        return await prompt_info.client.get_prompt(prompt_info.prompt.name, arguments)

    async def read_resource(self, resource_uri: str) -> types.ReadResourceResult:
        """Read a resource by URI."""
        resource_info = self.get_resource_info(resource_uri)
        return await resource_info.client.read_resource(resource_uri)


class ToolInfo:
    """Information about a registered tool."""

    def __init__(self, tool: types.Tool, client: MCPClient):
        self.tool = tool
        self.client = client


class PromptInfo:
    """Information about a registered prompt."""

    def __init__(self, prompt: types.Prompt, client: MCPClient):
        self.prompt = prompt
        self.client = client


class ResourceInfo:
    """Information about a registered resource."""

    def __init__(self, resource: types.Resource, client: MCPClient):
        self.resource = resource
        self.client = client
