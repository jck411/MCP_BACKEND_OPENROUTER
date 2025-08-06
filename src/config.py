"""Configuration management for the MCP client."""

import json
import os
from typing import Any

import yaml
from dotenv import load_dotenv


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration from YAML and environment variables."""
        self.load_env()  # Load .env for API keys
        self._config = self._load_yaml_config()  # Load YAML config

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    def _load_yaml_config(self) -> dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_path) as file:
            config = yaml.safe_load(file)
            if not isinstance(config, dict):
                raise ValueError("Configuration file must contain a dictionary")
            return config

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path) as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the API key for the active LLM provider.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        # Get active provider
        llm_config = self._config.get("llm", {})
        active_provider = llm_config.get("active", "openai")

        # Map provider names to environment variable names
        provider_key_map = {
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }

        env_key = provider_key_map.get(active_provider)
        if not env_key:
            raise ValueError(
                f"Unknown provider '{active_provider}' - no API key mapping found"
            )

        api_key = os.getenv(env_key)
        if not api_key:
            raise ValueError(
                f"API key '{env_key}' not found in environment variables "
                f"for provider '{active_provider}'"
            )

        return api_key

    def get_config_dict(self) -> dict[str, Any]:
        """Get the full configuration dictionary for websocket server.

        Returns:
            The complete configuration dictionary.
        """
        return self._config

    def get_llm_config(self) -> dict[str, Any]:
        """Get active LLM provider configuration from YAML.

        Returns:
            Active LLM provider configuration dictionary.
        """
        llm_config = self._config.get("llm", {})
        active_provider = llm_config.get("active", "openai")
        providers = llm_config.get("providers", {})

        if active_provider not in providers:
            raise ValueError(
                f"Active provider '{active_provider}' not found in providers config"
            )

        return providers[active_provider]

    def get_websocket_config(self) -> dict[str, Any]:
        """Get WebSocket configuration from YAML.

        Returns:
            WebSocket configuration dictionary.
        """
        return self._config.get("chat", {}).get("websocket", {})

    def get_logging_config(self) -> dict[str, Any]:
        """Get logging configuration from YAML.

        Returns:
            Logging configuration dictionary.
        """
        return self._config.get("logging", {})

    def get_chat_service_config(self) -> dict[str, Any]:
        """Get chat service configuration from YAML.

        Returns:
            Chat service configuration dictionary.
        """
        return self._config.get("chat", {}).get("service", {})

    def get_max_tool_hops(self) -> int:
        """Get the maximum number of tool hops allowed.

        Returns:
            Maximum number of tool hops (default: 8).
        """
        service_config = self.get_chat_service_config()
        max_hops = service_config.get("max_tool_hops", 8)

        # Validate that it's a positive integer
        if not isinstance(max_hops, int) or max_hops < 1:
            raise ValueError("max_tool_hops must be a positive integer")

        return max_hops

    def get_mcp_connection_config(self) -> dict[str, Any]:
        """Get MCP connection configuration from YAML.

        Returns:
            MCP connection configuration dictionary with validated defaults.
        """
        mcp_config = self._config.get("mcp", {})
        connection_config = mcp_config.get("connection", {})

        # Get values with defaults
        max_attempts = connection_config.get("max_reconnect_attempts", 5)
        initial_delay = connection_config.get("initial_reconnect_delay", 1.0)
        max_delay = connection_config.get("max_reconnect_delay", 30.0)
        connection_timeout = connection_config.get("connection_timeout", 30.0)
        ping_timeout = connection_config.get("ping_timeout", 10.0)

        # Validate configuration values
        if max_attempts < 1:
            raise ValueError("max_reconnect_attempts must be at least 1")
        if initial_delay <= 0:
            raise ValueError("initial_reconnect_delay must be positive")
        if max_delay < initial_delay:
            raise ValueError("max_reconnect_delay must be >= initial_reconnect_delay")
        if connection_timeout <= 0:
            raise ValueError("connection_timeout must be positive")
        if ping_timeout <= 0:
            raise ValueError("ping_timeout must be positive")

        return {
            "max_reconnect_attempts": max_attempts,
            "initial_reconnect_delay": initial_delay,
            "max_reconnect_delay": max_delay,
            "connection_timeout": connection_timeout,
            "ping_timeout": ping_timeout,
        }
