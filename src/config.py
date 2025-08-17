"""Configuration management for the MCP client."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from typing import Any, cast

import yaml
from dotenv import load_dotenv


class Configuration:
    """Event-driven configuration manager with observer pattern."""

    def __init__(self) -> None:
        """Initialize configuration from YAML and environment variables."""
        self.load_env()  # Load .env for API keys
        # Load default YAML config (reference only)
        self._default_config = self._load_yaml_config()
        self._runtime_config_path = os.path.join(
            os.path.dirname(__file__), "runtime_config.yaml"
        )
        self._runtime_config_mtime: float | None = None
        self._current_config: dict[str, Any] = {}

        # Event-driven observer pattern
        self._config_change_callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._watch_task: asyncio.Task[None] | None = None

        # Initialize persistent runtime configuration
        self._initialize_runtime_config()
        self._reload_config()

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
            return cast(dict[str, Any], config)

    def _initialize_runtime_config(self) -> None:
        """Initialize runtime configuration file if it doesn't exist."""
        if not os.path.exists(self._runtime_config_path):
            # Create runtime_config.yaml from defaults on first run
            initial_config = self._default_config.copy()
            initial_config["_runtime_config"] = {
                "last_modified": time.time(),
                "version": 1,
                "is_runtime_config": True,
                "default_config_path": "config.yaml",
                "created_from_defaults": True,
            }

            with open(self._runtime_config_path, "w") as file:
                yaml.safe_dump(initial_config, file, default_flow_style=False, indent=2)

    def _load_runtime_config(self) -> dict[str, Any]:
        """Load runtime configuration from YAML file."""
        try:
            with open(self._runtime_config_path) as file:
                config = yaml.safe_load(file)
                if not isinstance(config, dict):
                    # If corrupted, recreate from defaults
                    self._initialize_runtime_config()
                    return self._load_runtime_config()
                return cast(dict[str, Any], config)
        except (yaml.YAMLError, OSError):
            # If runtime config is corrupted or unreadable, recreate from defaults
            self._initialize_runtime_config()
            return self._load_runtime_config()

    def _deep_merge(
        self, base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence."""
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(
                    cast(dict[str, Any], result[key]), cast(dict[str, Any], value)
                )
            else:
                result[key] = value

        return result

    def _reload_config(self) -> bool:
        """Reload configuration from runtime config if it has been modified.

        Returns:
            True if config was actually reloaded, False if no changes.
        """
        # Check if runtime config file exists and get its modification time
        current_mtime = None
        if os.path.exists(self._runtime_config_path):
            current_mtime = os.path.getmtime(self._runtime_config_path)

        # If modification time changed or this is the first load, reload config
        if current_mtime != self._runtime_config_mtime:
            old_config = self._current_config.copy()
            self._runtime_config_mtime = current_mtime
            runtime_config = self._load_runtime_config()

            # Runtime config IS the current config (with defaults for missing values)
            # Remove runtime metadata before using as current config
            self._current_config = {
                k: v
                for k, v in runtime_config.items()
                if not k.startswith("_runtime_config")
            }

            # Notify observers if config actually changed (not just first load)
            if old_config and self._current_config != old_config:
                self._notify_config_change()

            return True
        return False

    def _get_current_config(self) -> dict[str, Any]:
        """Get current configuration (cached, no file system access)."""
        return self._current_config

    def _notify_config_change(self) -> None:
        """Notify all registered observers of configuration changes."""
        for callback in self._config_change_callbacks:
            try:
                callback(self._current_config.copy())
            except Exception as e:
                logging.error(f"Error in config change callback: {e}")

    def subscribe_to_changes(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Subscribe to configuration change events.

        Args:
            callback: Function to call when config changes. Receives new
                config as argument.
        """
        if callback not in self._config_change_callbacks:
            self._config_change_callbacks.append(callback)

    def unsubscribe_from_changes(
        self, callback: Callable[[dict[str, Any]], None]
    ) -> None:
        """Unsubscribe from configuration change events.

        Args:
            callback: Function to remove from notifications.
        """
        if callback in self._config_change_callbacks:
            self._config_change_callbacks.remove(callback)

    async def start_watching(self) -> None:
        """Start the async file watching task for automatic config updates."""
        if self._watch_task is not None:
            return  # Already watching

        self._watch_task = asyncio.create_task(self._watch_config_file())
        logging.info("Started watching runtime configuration file for changes")

    async def stop_watching(self) -> None:
        """Stop the async file watching task."""
        if self._watch_task is not None:
            self._watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watch_task
            self._watch_task = None
            logging.info("Stopped watching runtime configuration file")

    async def _watch_config_file(self) -> None:
        """Async task that watches for config file changes."""
        while True:
            try:
                await asyncio.sleep(1)  # Check every second - still efficient
                if self._reload_config():
                    logging.info("Runtime configuration file changed - config reloaded")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error watching config file: {e}")
                await asyncio.sleep(5)  # Back off on errors

    def _get_config_value(self, path: list[str], default: Any = None) -> Any:
        """Get a configuration value by path, with fallback to defaults."""
        # Try to get from current runtime config first
        current: Any = self._get_current_config()
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]  # type: ignore[assignment]
            else:
                # Fall back to default config
                default_current: Any = self._default_config
                for default_key in path:
                    if (
                        isinstance(default_current, dict)
                        and default_key in default_current
                    ):
                        default_current = default_current[default_key]  # type: ignore[assignment]
                    else:
                        return default
                return default_current  # type: ignore[return-value]
        return current  # type: ignore[return-value]

    def reload_runtime_config(self) -> bool:
        """Manually reload runtime configuration.

        Returns:
            True if configuration was reloaded, False if no changes detected.
        """
        old_mtime = self._runtime_config_mtime
        self._reload_config()
        return old_mtime != self._runtime_config_mtime

    def save_runtime_config(self, config: dict[str, Any]) -> None:
        """Save configuration to runtime config file.

        Args:
            config: Configuration dictionary to save.
        """
        # Get current version from existing runtime config
        current_runtime_config: dict[str, Any] = {}
        if os.path.exists(self._runtime_config_path):
            try:
                with open(self._runtime_config_path) as f:
                    loaded_config = yaml.safe_load(f)
                    if isinstance(loaded_config, dict):
                        current_runtime_config = cast(dict[str, Any], loaded_config)
            except (yaml.YAMLError, OSError):
                pass

        current_version = current_runtime_config.get("_runtime_config", {}).get(
            "version", 0
        )

        # Add runtime metadata
        runtime_config = config.copy()
        runtime_config["_runtime_config"] = {
            "last_modified": time.time(),
            "version": current_version + 1,
            "is_runtime_config": True,
            "default_config_path": "config.yaml",
        }

        with open(self._runtime_config_path, "w") as file:
            yaml.safe_dump(runtime_config, file, default_flow_style=False, indent=2)

        # Reload the configuration
        self._reload_config()

    def get_runtime_metadata(self) -> dict[str, Any]:
        """Get runtime configuration metadata.

        Returns:
            Runtime configuration metadata dictionary.
        """
        if os.path.exists(self._runtime_config_path):
            try:
                with open(self._runtime_config_path) as f:
                    loaded_config = yaml.safe_load(f)
                    if isinstance(loaded_config, dict):
                        runtime_config = cast(dict[str, Any], loaded_config)
                        return runtime_config.get("_runtime_config", {})
            except (yaml.YAMLError, OSError):
                pass
        return {}

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
        # Get active provider from current config
        config = self._get_current_config()
        llm_config = config.get("llm", {})
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
        return self._get_current_config()

    def get_llm_config(self) -> dict[str, Any]:
        """Get active LLM provider configuration from YAML.

        Returns:
            Active LLM provider configuration dictionary.
        """
        config = self._get_current_config()
        llm_config = config.get("llm", {})
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
        config = self._get_current_config()
        return config.get("chat", {}).get("websocket", {})

    def get_logging_config(self) -> dict[str, Any]:
        """Get logging configuration from YAML.

        Returns:
            Logging configuration dictionary.
        """
        config = self._get_current_config()
        return config.get("logging", {})

    def get_chat_service_config(self) -> dict[str, Any]:
        """Get chat service configuration from YAML.

        Returns:
            Chat service configuration dictionary.
        """
        config = self._get_current_config()
        return config.get("chat", {}).get("service", {})

    def get_chat_storage_config(self) -> dict[str, Any]:
        """Get chat storage configuration from YAML.

        Returns:
            Chat storage configuration dictionary.
        """
        config = self._get_current_config()
        return config.get("chat", {}).get("storage", {})

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
        config = self._get_current_config()
        mcp_config = config.get("mcp", {})
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

    def reset_to_defaults(self) -> None:
        """Reset runtime_config.yaml to the defaults from config.yaml."""
        # Save a copy of the default config as the new runtime config.
        # save_runtime_config will add runtime metadata and increment version
        # automatically.
        self.save_runtime_config(self._default_config.copy())


def reset_runtime_config_cli() -> None:
    """Console script that resets src/runtime_config.yaml to defaults."""
    try:
        cfg = Configuration()
        cfg.reset_to_defaults()
        logging.info("✓ runtime_config.yaml reset to defaults from config.yaml")
    except Exception as e:
        logging.error(f"Error resetting runtime configuration: {e}")
        sys.exit(1)
