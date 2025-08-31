"""
Model capability manager for OpenRouter API.

Fetches and caches model capability information from OpenRouter to enable
dynamic parameter filtering based on what each model actually supports.
"""

from __future__ import annotations

import asyncio
import logging

import httpx


class ModelCapabilities:
    """
    Fetches and caches model capability information from OpenRouter.

    It exposes supported_parameters and input_modalities so that the LLM client
    can filter unsupported request keys (e.g. 'tools', 'tool_choice') at runtime.
    """

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, set[str]]] = {}
        self._lock = asyncio.Lock()

    async def _fetch_model_info(self, model_slug: str) -> dict[str, set[str]] | None:
        """Fetch model information from OpenRouter API."""
        url = "https://openrouter.ai/api/v1/models"
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json().get("data", [])
            except Exception as e:
                logging.error(f"Failed to fetch model list from OpenRouter: {e}")
                return None

        # Search for model by ID or canonical_slug
        for m in data:
            if m.get("id") == model_slug or m.get("canonical_slug") == model_slug:
                # Extract capability information
                supported_params = set(m.get("supported_parameters", []))
                architecture = m.get("architecture", {})
                input_modalities = set(architecture.get("input_modalities", []))
                output_modalities = set(architecture.get("output_modalities", []))

                return {
                    "supported_parameters": supported_params,
                    "input_modalities": input_modalities,
                    "output_modalities": output_modalities,
                }

        logging.warning(f"Model '{model_slug}' not found in OpenRouter model list")
        return None

    async def get(self, model_slug: str) -> dict[str, set[str]]:
        """Get capability information for a model, with caching."""
        async with self._lock:
            if model_slug in self._cache:
                return self._cache[model_slug]

            info = await self._fetch_model_info(model_slug)
            if not info:
                # Default to empty sets so that unsupported fields are dropped
                info = {"supported_parameters": set(), "input_modalities": set(), "output_modalities": set()}
                logging.info(f"Using empty capability set for model '{model_slug}' (not found or API error)")
            else:
                logging.info(
                    f"Cached capabilities for model '{model_slug}': {len(info['supported_parameters'])} parameters"
                )

            self._cache[model_slug] = info
            return info

    async def supported_parameters(self, model_slug: str) -> set[str]:
        """Get the set of supported parameters for a model."""
        capabilities = await self.get(model_slug)
        return capabilities.get("supported_parameters", set())

    async def input_modalities(self, model_slug: str) -> set[str]:
        """Get the set of input modalities supported by a model."""
        capabilities = await self.get(model_slug)
        return capabilities.get("input_modalities", set())

    async def output_modalities(self, model_slug: str) -> set[str]:
        """Get the set of output modalities supported by a model."""
        capabilities = await self.get(model_slug)
        return capabilities.get("output_modalities", set())
