"""
Resource Loading Handler

Handles fragile resource-related operations:
- Resource availability checking
- Resource catalog management  
- System prompt construction with resources
- Prompt listing and application

Resource loading fails often when MCP servers are down, so this module
is isolated for better error handling and logging.
"""

import logging
from typing import Any

from mcp import types

logger = logging.getLogger(__name__)


class ResourceLoader:
    """Handles resource loading and system prompt construction."""

    def __init__(self, tool_mgr, configuration):
        self.tool_mgr = tool_mgr
        self.configuration = configuration  # Use Configuration object instead of static dict
        self._resource_catalog: list[str] = []

    async def initialize(self) -> str:
        """Initialize resource catalog and build system prompt."""
        logger.info("→ Resources: initializing resource loader")
        
        # Update resource catalog to only include available resources
        await self.update_resource_catalog_on_availability()
        
        # Build system prompt with available resources
        system_prompt = await self.make_system_prompt()
        
        logger.info("← Resources: initialization completed")
        return system_prompt

    async def update_resource_catalog_on_availability(self) -> None:
        """
        Update the resource catalog to reflect current availability.

        This implements a circuit-breaker-like pattern where we periodically
        check if previously failed resources have become available again.
        """
        if not self.tool_mgr:
            logger.warning("No tool manager available for resource catalog update")
            return

        logger.debug("→ Resources: checking resource availability")

        # Get all registered resources from the tool manager
        all_resource_uris = self.tool_mgr.list_available_resources()

        # Filter to only include resources that are actually available
        available_uris = []
        for uri in all_resource_uris:
            try:
                resource_result = await self.tool_mgr.read_resource(uri)
                if resource_result.contents:
                    available_uris.append(uri)
                    logger.debug("→ Resources: %s is available", uri)
                else:
                    logger.debug("→ Resources: %s has no content, skipping", uri)
            except Exception as e:
                # Skip unavailable resources silently in normal operation
                logger.debug("→ Resources: %s is unavailable: %s", uri, e)
                continue

        # Update the catalog to only include working resources
        self._resource_catalog = available_uris
        logger.info(
            "← Resources: catalog updated - %d of %d resources available",
            len(available_uris), len(all_resource_uris)
        )

    async def make_system_prompt(self) -> str:
        """Build the system prompt with actual resource contents and prompts."""
        logger.debug("→ Resources: building system prompt")
        
        # Get system prompt from current configuration (runtime-aware)
        chat_service_config = self.configuration.get_chat_service_config()
        base = chat_service_config.get("system_prompt", "You are a helpful assistant.").rstrip()

        if not self.tool_mgr:
            logger.warning("No tool manager available for system prompt construction")
            return base

        # Only include resources that are actually available
        available_resources = await self.get_available_resources()
        if available_resources:
            logger.info("→ Resources: including %d resources in system prompt", len(available_resources))
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

        # Add available prompts section
        prompt_names = self.tool_mgr.list_available_prompts()
        if prompt_names:
            logger.info("→ Resources: including %d prompts in system prompt", len(prompt_names))
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

        logger.debug("← Resources: system prompt built, length=%d chars", len(base))
        return base

    async def get_available_resources(
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
            logger.debug("No resources in catalog or no tool manager available")
            return available_resources

        logger.debug("→ Resources: checking availability of %d resources", len(self._resource_catalog))

        for uri in self._resource_catalog:
            try:
                resource_result = await self.tool_mgr.read_resource(uri)
                if resource_result.contents:
                    # Only include resources that have actual content
                    available_resources[uri] = resource_result.contents
                    logger.debug("→ Resources: %s loaded successfully", uri)
                else:
                    logger.debug("→ Resources: %s has no content, skipping", uri)
            except Exception as e:
                # Log the error but don't include in system prompt
                # This prevents the LLM from being told about broken resources
                logger.warning(
                    "→ Resources: %s is unavailable and excluded from system prompt: %s",
                    uri, e
                )
                continue

        if available_resources:
            logger.info(
                "← Resources: %d resources are available for system prompt",
                len(available_resources)
            )
        else:
            logger.info(
                "← Resources: no resources are currently available - "
                "system prompt will not include resource section"
            )

        return available_resources

    async def apply_prompt(self, name: str, args: dict[str, str]) -> list[dict]:
        """Apply a parameterized prompt and return conversation messages."""
        if not self.tool_mgr:
            raise RuntimeError("Tool manager not initialized")

        logger.info("→ Resources: applying prompt '%s' with args: %s", name, args)

        try:
            res = await self.tool_mgr.get_prompt(name, args)
            messages = [
                {"role": m.role, "content": m.content.text}
                for m in res.messages
                if isinstance(m.content, types.TextContent)
            ]
            logger.info("← Resources: prompt applied successfully, %d messages", len(messages))
            return messages
        except Exception as e:
            logger.error("← Resources: failed to apply prompt '%s': %s", name, e)
            raise

    def get_resource_catalog(self) -> list[str]:
        """Get the current resource catalog."""
        return self._resource_catalog.copy()

    def get_resource_count(self) -> int:
        """Get the number of available resources."""
        return len(self._resource_catalog)
