"""
FastMCP Desktop Example

A simple example that exposes the desktop directory as a resource,
demonstrates tools, and includes example prompts.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, cast

import yaml
from mcp.server.fastmcp import FastMCP

# Tool configuration - easily toggle tools on/off
TOOL_CONFIG = {
    "math_tools": True,
    "advanced_math": True,
    "conversation_prompts": False,
    "desktop_resources": True,
    "config_tools": True,
}

# Configuration validation constants
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MAX_TOKENS_LIMIT = 50000
MIN_PENALTY = -2.0
MAX_PENALTY = 2.0

# Create server
mcp = FastMCP("Demo")


# Path helpers
HOME_DIR = Path.home()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESOURCES_DIR = HOME_DIR / "Documents" / "MCP.resources"
RUNTIME_CONFIG_PATH = PROJECT_ROOT / "src" / "runtime_config.yaml"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "src" / "config.yaml"


# File resources
if TOOL_CONFIG["desktop_resources"]:

    @mcp.resource(
        "resource://desktop-files",
        name="DesktopListing",
        description="List of files under ~/Documents/MCP.resources",
        mime_type="text/plain",
    )
    def desktop() -> str:
        """List the files in the MCP resources directory"""
        try:
            if not RESOURCES_DIR.exists() or not RESOURCES_DIR.is_dir():
                return f"Resource directory not found: {RESOURCES_DIR}"
            files = [f.name for f in RESOURCES_DIR.iterdir() if f.is_file()]
            return "\n".join(f"- {file}" for file in sorted(files))
        except Exception as exc:
            logging.exception("Failed to list resources directory")
            return f"Error reading resource directory {RESOURCES_DIR}: {exc}"


# Basic math tools
if TOOL_CONFIG["math_tools"]:

    @mcp.tool()
    def sum(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    @mcp.tool()
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b

    @mcp.tool()
    def subtract(a: int, b: int) -> int:
        """Subtract second number from first"""
        return a - b


# Advanced math tools (disabled by default)
if TOOL_CONFIG["advanced_math"]:

    @mcp.tool()
    def power(a: int, b: int) -> int:
        """Raise first number to the power of second"""
        return a**b

    @mcp.tool()
    def divide(a: float, b: float) -> float:
        """Divide first number by second"""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b


# Configuration tools
if TOOL_CONFIG["config_tools"]:

    def _load_config() -> dict[str, Any]:
        """Helper to load and parse the runtime config"""
        try:
            config_text = RUNTIME_CONFIG_PATH.read_text()
        except FileNotFoundError:
            logging.info(
                "Runtime config not found at %s; returning empty config",
                RUNTIME_CONFIG_PATH,
            )
            return {}
        except Exception:
            logging.exception(
                "Failed reading runtime config from %s", RUNTIME_CONFIG_PATH
            )
            return {}
        result = yaml.safe_load(config_text)
        return cast(dict[str, Any], result) if isinstance(result, dict) else {}

    def _get_active_provider_config() -> tuple[str, dict[str, Any]]:
        """Helper to get active provider name and its config"""
        config: dict[str, Any] = _load_config()
        llm_config: dict[str, Any] = cast(dict[str, Any], config.get("llm", {}))
        active_provider: str = cast(str, llm_config.get("active", "groq"))
        providers: dict[str, Any] = cast(
            dict[str, Any], llm_config.get("providers", {})
        )
        provider_config: dict[str, Any] = cast(
            dict[str, Any], providers.get(active_provider, {})
        )
        return active_provider, provider_config

    def _save_config(config: dict[str, Any]) -> None:
        """Helper to save the runtime config"""
        try:
            RUNTIME_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with RUNTIME_CONFIG_PATH.open("w") as f:
                yaml.safe_dump(config, f, default_flow_style=False, indent=2)
        except Exception:
            logging.exception(
                "Failed writing runtime config to %s", RUNTIME_CONFIG_PATH
            )

    @mcp.tool()
    def get_system_prompt() -> str:
        """Get the current system prompt configuration"""
        config: dict[str, Any] = _load_config()
        chat: dict[str, Any] = cast(dict[str, Any], config.get("chat", {}))
        service: dict[str, Any] = cast(dict[str, Any], chat.get("service", {}))
        system_prompt: str = cast(
            str, service.get("system_prompt", "No system prompt configured")
        )
        return f"System prompt: {system_prompt}"

    @mcp.tool()
    def get_sampling_parameters() -> str:
        """Get sampling parameters (temperature, top_p, top_k) for active provider"""
        active_provider, provider_config = _get_active_provider_config()

        temperature = provider_config.get("temperature", "not set")
        top_p = provider_config.get("top_p", "not set")
        top_k = provider_config.get("top_k", "not set")

        return (
            f"Sampling parameters for {active_provider}:\n"
            f"- Temperature: {temperature}\n"
            f"- Top-p: {top_p}\n"
            f"- Top-k: {top_k}"
        )

    @mcp.tool()
    def get_length_parameters() -> str:
        """Get length parameters (max_tokens, min_length) for active provider"""
        active_provider, provider_config = _get_active_provider_config()

        max_tokens = provider_config.get("max_tokens", "not set")
        min_length = provider_config.get("min_length", "not set")
        max_completion_tokens = provider_config.get("max_completion_tokens", "not set")

        return (
            f"Length parameters for {active_provider}:\n"
            f"- Max tokens: {max_tokens}\n"
            f"- Min length: {min_length}\n"
            f"- Max completion tokens: {max_completion_tokens}"
        )

    @mcp.tool()
    def get_penalty_parameters() -> str:
        """Get penalty parameters for active provider"""
        active_provider, provider_config = _get_active_provider_config()

        presence_penalty = provider_config.get("presence_penalty", "not set")
        frequency_penalty = provider_config.get("frequency_penalty", "not set")
        repetition_penalty = provider_config.get("repetition_penalty", "not set")

        return (
            f"Penalty parameters for {active_provider}:\n"
            f"- Presence penalty: {presence_penalty}\n"
            f"- Frequency penalty: {frequency_penalty}\n"
            f"- Repetition penalty: {repetition_penalty}"
        )

    @mcp.tool()
    def get_stopping_criteria() -> str:
        """Get stopping criteria (stop sequences, max tokens) for active provider"""
        active_provider, provider_config = _get_active_provider_config()

        stop_sequences = provider_config.get("stop", "not set")
        stop_tokens = provider_config.get("stop_tokens", "not set")
        end_of_text = provider_config.get("end_of_text", "not set")

        return (
            f"Stopping criteria for {active_provider}:\n"
            f"- Stop sequences: {stop_sequences}\n"
            f"- Stop tokens: {stop_tokens}\n"
            f"- End of text: {end_of_text}"
        )

    # Configuration editing tools
    @mcp.tool()
    def set_system_prompt(prompt: str) -> str:
        """
        Set the system prompt configuration but ensure it always begins with
        'You are a helpful assistant.'
        """
        if not prompt.strip():
            return "Error: System prompt cannot be empty"

        config: dict[str, Any] = _load_config()
        # Ensure chat.service structure exists
        if "chat" not in config:
            config["chat"] = {}
        if "service" not in config["chat"]:
            config["chat"]["service"] = {}

        user_prompt = prompt.strip()
        prefix = "You are a helpful assistant."
        # Avoid duplicating the prefix if the user already included it
        if user_prompt.lower().startswith(prefix.lower()):
            combined = user_prompt
        else:
            combined = f"{prefix} {user_prompt}"

        config["chat"]["service"]["system_prompt"] = combined
        _save_config(config)
        return "System prompt updated successfully"

    @mcp.tool()
    def set_temperature(temperature: float) -> str:
        """Set the temperature for the active LLM provider (0.0-2.0)"""
        if not MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE:
            return (
                f"Error: Temperature must be between {MIN_TEMPERATURE} "
                f"and {MAX_TEMPERATURE}"
            )

        config: dict[str, Any] = _load_config()
        active_provider, _ = _get_active_provider_config()

        # Update the active provider's temperature
        config["llm"]["providers"][active_provider]["temperature"] = temperature
        _save_config(config)
        return f"Temperature set to {temperature} for {active_provider}"

    @mcp.tool()
    def set_max_tokens(max_tokens: int) -> str:
        """Set the max_tokens for the active LLM provider"""
        if max_tokens <= 0:
            return "Error: max_tokens must be positive"
        if max_tokens > MAX_TOKENS_LIMIT:
            return f"Error: max_tokens too large (max {MAX_TOKENS_LIMIT})"

        config: dict[str, Any] = _load_config()
        active_provider, _ = _get_active_provider_config()

        config["llm"]["providers"][active_provider]["max_tokens"] = max_tokens
        _save_config(config)
        return f"Max tokens set to {max_tokens} for {active_provider}"

    @mcp.tool()
    def set_top_p(top_p: float) -> str:
        """Set the top_p sampling parameter for the active LLM provider (0.0-1.0)"""
        if not 0.0 <= top_p <= 1.0:
            return "Error: top_p must be between 0.0 and 1.0"

        config: dict[str, Any] = _load_config()
        active_provider, _ = _get_active_provider_config()

        config["llm"]["providers"][active_provider]["top_p"] = top_p
        _save_config(config)
        return f"Top-p set to {top_p} for {active_provider}"

    @mcp.tool()
    def switch_llm_provider(provider: str) -> str:
        """Switch to a different LLM provider (groq, openai, openrouter, etc.)"""
        config: dict[str, Any] = _load_config()
        providers: dict[str, Any] = cast(
            dict[str, Any],
            cast(dict[str, Any], config.get("llm", {})).get("providers", {}),
        )
        available_providers = list(providers.keys())

        if provider not in available_providers:
            available_str = ", ".join(available_providers)
            return f"Error: Provider '{provider}' not found. Available: {available_str}"

        config["llm"]["active"] = provider
        _save_config(config)
        return f"Switched to LLM provider: {provider}"

    @mcp.tool()
    def set_presence_penalty(penalty: float) -> str:
        """Set the presence_penalty for the active LLM provider (-2.0 to 2.0)"""
        if not MIN_PENALTY <= penalty <= MAX_PENALTY:
            return (
                f"Error: presence_penalty must be between {MIN_PENALTY} "
                f"and {MAX_PENALTY}"
            )

        config: dict[str, Any] = _load_config()
        active_provider, _ = _get_active_provider_config()

        config["llm"]["providers"][active_provider]["presence_penalty"] = penalty
        _save_config(config)
        return f"Presence penalty set to {penalty} for {active_provider}"

    @mcp.tool()
    def set_frequency_penalty(penalty: float) -> str:
        """Set the frequency_penalty for the active LLM provider (-2.0 to 2.0)"""
        if not MIN_PENALTY <= penalty <= MAX_PENALTY:
            return (
                f"Error: frequency_penalty must be between {MIN_PENALTY} "
                f"and {MAX_PENALTY}"
            )

        config = _load_config()
        active_provider, _ = _get_active_provider_config()

        config["llm"]["providers"][active_provider]["frequency_penalty"] = penalty
        _save_config(config)
        return f"Frequency penalty set to {penalty} for {active_provider}"

    @mcp.tool()
    def reset_runtime_config_to_defaults() -> str:
        """
        Reset runtime_config.yaml to defaults from config.yaml.
        Writes defaults and updates metadata to trigger reload.
        """
        try:
            default_path = DEFAULT_CONFIG_PATH
            runtime_path = RUNTIME_CONFIG_PATH

            if not default_path.exists():
                return f"Error: Default config file not found at {default_path}"

            default_cfg = yaml.safe_load(default_path.read_text())
            if not isinstance(default_cfg, dict):
                return "Error: Default config is not a valid mapping"

            # Preserve and bump version if present to aid change tracking
            current_version: int = 0
            try:
                existing_raw = yaml.safe_load(runtime_path.read_text())
                existing: dict[str, Any] = (
                    existing_raw if isinstance(existing_raw, dict) else {}
                )
                runtime_meta: dict[str, Any] = cast(
                    dict[str, Any], existing.get("_runtime_config", {})
                )
                current_version = cast(int, runtime_meta.get("version", 0))
            except Exception:
                pass

            default_cfg["_runtime_config"] = {
                "last_modified": time.time(),
                "version": current_version + 1,
                "is_runtime_config": True,
                "default_config_path": "config.yaml",
                "created_from_defaults": True,
            }

            runtime_path.parent.mkdir(parents=True, exist_ok=True)
            with runtime_path.open("w") as f:
                yaml.safe_dump(default_cfg, f, default_flow_style=False, indent=2)

            return "✓ runtime_config.yaml reset to defaults from config.yaml"
        except Exception as e:
            logging.exception("Failed to reset runtime configuration")
            return f"Error resetting runtime configuration: {e}"


# Conversation prompts
if TOOL_CONFIG["conversation_prompts"]:

    @mcp.prompt()
    def summarize_conversation() -> str:
        """Create a summary of the current conversation"""
        return (
            "Please provide a concise summary of our conversation so far, "
            "highlighting the main topics and any important conclusions."
        )

    @mcp.prompt()
    def generate_questions() -> str:
        """Generate follow-up questions about the conversation"""
        return (
            "Based on our conversation, please generate 3-5 thoughtful "
            "follow-up questions that could help continue or deepen "
            "the discussion."
        )


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Log which features are enabled
    enabled_features = [k for k, v in TOOL_CONFIG.items() if v]
    logging.info(f"Demo Server starting with features: {', '.join(enabled_features)}")
    mcp.run()
