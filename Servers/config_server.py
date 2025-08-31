"""
FastMCP Configuration Server

A simple example that provides LLM provider configuration management tools.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import yaml
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# Per-tool toggles (default True). Disable any individual tool by name.
TOOL_TOGGLES = {
    # Configuration tools
    "get_current_llm_provider": True,
    "get_system_prompt": True,
    "get_sampling_parameters": True,
    "get_length_parameters": True,
    "get_penalty_parameters": True,
    "get_stopping_criteria": True,
    "set_system_prompt": True,
    "set_temperature": True,
    "set_max_tokens": True,
    "set_top_p": True,
    "switch_llm_provider": True,
    "set_presence_penalty": True,
    "set_frequency_penalty": True,
    "set_repetition_penalty": True,
    "set_top_k": True,
    "set_seed": True,
    "set_min_p": True,
    "set_top_a": True,
    "set_stop_sequences": True,
    "set_response_format": True,
    "set_structured_outputs": True,
    "set_include_reasoning": True,
    "set_reasoning": True,
    "reset_runtime_config_to_defaults": True,
    # Logging configuration tools
    "get_logging_config": True,
    "set_global_log_level": True,
    "set_module_log_level": True,
    "set_logging_feature": True,
    "set_logging_format": True,
    "set_advanced_logging_option": True,
}

F = TypeVar("F", bound=Callable[..., Any])


def _identity_decorator[F](func: F) -> F:
    """Identity decorator that does nothing - used when tools are disabled."""
    return func


# Conditional decorators that become no-ops when a tool is disabled.
def tool_if(toggle_key: str) -> Callable[[F], F]:
    """Conditional tool decorator - only registers if toggle is True."""
    if TOOL_TOGGLES.get(toggle_key, True):
        return cast(Callable[[F], F], mcp.tool())
    return _identity_decorator


# Configuration tools - all tools are enabled by default


class ConfigServerConfig(BaseModel):
    """Configuration for the config server with caching and validation."""

    # Server paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    runtime_config_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "src" / "runtime_config.yaml"
    )
    default_config_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "src" / "config.yaml"
    )

    # Cached config data
    _config_cache: dict[str, Any] | None = None
    _config_mtime: float | None = None

    def get_config(self) -> dict[str, Any]:
        """Get cached configuration with automatic reload on file changes."""
        if self.runtime_config_path.exists():
            current_mtime = self.runtime_config_path.stat().st_mtime
            if self._config_cache is None or self._config_mtime != current_mtime:
                try:
                    with open(self.runtime_config_path, encoding="utf-8") as f:
                        config_text = f.read()
                    self._config_cache = yaml.safe_load(config_text) or {}
                    self._config_mtime = current_mtime
                except Exception:
                    logging.exception("Failed to load config")
                    self._config_cache = {}
            return self._config_cache.copy()
        return {}

    def save_config(self, config: dict[str, Any]) -> None:
        """Save configuration and invalidate cache."""
        try:
            self.runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.runtime_config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, default_flow_style=False, indent=2)
            self._config_cache = None  # Invalidate cache
            self._config_mtime = None
        except Exception:
            logging.exception("Failed to save config")

    def get_active_provider_config(self) -> tuple[str, dict[str, Any]]:
        """Helper to get active provider name and its config."""
        config = self.get_config()
        llm_config = config.get("llm", {})
        active_provider = llm_config.get("active", "groq")
        providers = llm_config.get("providers", {})
        provider_config = providers.get(active_provider, {})
        return active_provider, provider_config


# Create single instance for the entire server
config_manager = ConfigServerConfig()

# Create server
mcp = FastMCP("Config")

# Configuration validation constants
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MAX_TOKENS_LIMIT = 50000
MIN_PENALTY = -2.0
MAX_PENALTY = 2.0

# Path helpers
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Configuration tools - now using cached config manager


@tool_if("get_current_llm_provider")
def get_current_llm_provider() -> str:
    """Get the currently active LLM provider and its details"""
    active_provider, provider_config = config_manager.get_active_provider_config()

    # Get available providers for context
    config = config_manager.get_config()
    providers = list(config.get("llm", {}).get("providers", {}).keys())

    # Format provider details
    details = []
    if "model" in provider_config:
        details.append(f"Model: {provider_config['model']}")
    if "base_url" in provider_config:
        # Mask API keys in URLs if present
        url = provider_config["base_url"]
        if "api.openai.com" in url:
            details.append("Provider: OpenAI")
        elif "api.groq.com" in url:
            details.append("Provider: Groq")
        elif "openrouter.ai" in url:
            details.append("Provider: OpenRouter")
        else:
            details.append(f"Base URL: {url}")

    details_str = " | ".join(details) if details else "No additional details"

    return (
        f"Current LLM Provider: {active_provider}\nDetails: {details_str}\nAvailable providers: {', '.join(providers)}"
    )


@tool_if("get_system_prompt")
def get_system_prompt() -> str:
    """Get the current system prompt configuration"""
    config = config_manager.get_config()
    chat = config.get("chat", {})
    service = chat.get("service", {})
    system_prompt = service.get("system_prompt", "No system prompt configured")
    return f"System prompt: {system_prompt}"


@tool_if("get_sampling_parameters")
def get_sampling_parameters() -> str:
    """Get sampling parameters (temperature, top_p, top_k) for active provider"""
    active_provider, provider_config = config_manager.get_active_provider_config()

    temperature = provider_config.get("temperature", "not set")
    top_p = provider_config.get("top_p", "not set")
    top_k = provider_config.get("top_k", "not set")

    return (
        f"Sampling parameters for {active_provider}:\n- Temperature: {temperature}\n- Top-p: {top_p}\n- Top-k: {top_k}"
    )


@tool_if("get_length_parameters")
def get_length_parameters() -> str:
    """Get length parameters (max_tokens, min_length) for active provider"""
    active_provider, provider_config = config_manager.get_active_provider_config()

    max_tokens = provider_config.get("max_tokens", "not set")
    min_length = provider_config.get("min_length", "not set")
    max_completion_tokens = provider_config.get("max_completion_tokens", "not set")

    return (
        f"Length parameters for {active_provider}:\n"
        f"- Max tokens: {max_tokens}\n"
        f"- Min length: {min_length}\n"
        f"- Max completion tokens: {max_completion_tokens}"
    )


@tool_if("get_penalty_parameters")
def get_penalty_parameters() -> str:
    """Get penalty parameters for active provider"""
    active_provider, provider_config = config_manager.get_active_provider_config()

    presence_penalty = provider_config.get("presence_penalty", "not set")
    frequency_penalty = provider_config.get("frequency_penalty", "not set")
    repetition_penalty = provider_config.get("repetition_penalty", "not set")

    return (
        f"Penalty parameters for {active_provider}:\n"
        f"- Presence penalty: {presence_penalty}\n"
        f"- Frequency penalty: {frequency_penalty}\n"
        f"- Repetition penalty: {repetition_penalty}"
    )


@tool_if("get_stopping_criteria")
def get_stopping_criteria() -> str:
    """Get stopping criteria (stop sequences, max tokens) for active provider"""
    active_provider, provider_config = config_manager.get_active_provider_config()

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


@tool_if("set_system_prompt")
def set_system_prompt(prompt: str) -> str:
    """
    Set the system prompt configuration but ensure it always begins with
    'You are a helpful assistant.'
    """
    if not prompt.strip():
        return "Error: System prompt cannot be empty"

    config = config_manager.get_config()
    # Ensure chat.service structure exists
    if "chat" not in config:
        config["chat"] = {}
    if "service" not in config["chat"]:
        config["chat"]["service"] = {}

    user_prompt = prompt.strip()
    prefix = "You are a helpful assistant."
    # Avoid duplicating the prefix if the user already included it
    combined = user_prompt if user_prompt.lower().startswith(prefix.lower()) else f"{prefix} {user_prompt}"

    config["chat"]["service"]["system_prompt"] = combined
    config_manager.save_config(config)
    return "System prompt updated successfully"


@tool_if("set_temperature")
def set_temperature(temperature: float) -> str:
    """Set the temperature for the active LLM provider (0.0-2.0)"""
    if not MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE:
        return f"Error: Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}"

    config = config_manager.get_config()
    active_provider, _ = config_manager.get_active_provider_config()

    # Update the active provider's temperature
    config["llm"]["providers"][active_provider]["temperature"] = temperature
    config_manager.save_config(config)
    return f"Temperature set to {temperature} for {active_provider}"


@tool_if("set_max_tokens")
def set_max_tokens(max_tokens: int) -> str:
    """Set the max_tokens for the active LLM provider"""
    if max_tokens <= 0:
        return "Error: max_tokens must be positive"
    if max_tokens > MAX_TOKENS_LIMIT:
        return f"Error: max_tokens too large (max {MAX_TOKENS_LIMIT})"

    config = config_manager.get_config()
    active_provider, _ = config_manager.get_active_provider_config()

    config["llm"]["providers"][active_provider]["max_tokens"] = max_tokens
    config_manager.save_config(config)
    return f"Max tokens set to {max_tokens} for {active_provider}"


@tool_if("set_top_p")
def set_top_p(top_p: float) -> str:
    """Set the top_p sampling parameter for the active LLM provider (0.0-1.0)"""
    if not 0.0 <= top_p <= 1.0:
        return "Error: top_p must be between 0.0 and 1.0"

    config = config_manager.get_config()
    active_provider, _ = config_manager.get_active_provider_config()

    config["llm"]["providers"][active_provider]["top_p"] = top_p
    config_manager.save_config(config)
    return f"Top-p set to {top_p} for {active_provider}"


@tool_if("switch_llm_provider")
def switch_llm_provider(provider: str) -> str:
    """Switch to a different LLM provider configuration (groq, openai, openrouter, etc.). Use this to switch between different provider setups, not to change models within a provider."""
    config = config_manager.get_config()
    providers: dict[str, Any] = cast(
        dict[str, Any],
        cast(dict[str, Any], config.get("llm", {})).get("providers", {}),
    )
    available_providers = list(providers.keys())

    if provider not in available_providers:
        available_str = ", ".join(available_providers)
        return f"Error: Provider '{provider}' not found. Available: {available_str}"

    config["llm"]["active"] = provider
    config_manager.save_config(config)
    return f"Switched to LLM provider: {provider}"


@tool_if("set_model")
def set_model(model: str) -> str:
    """Change the model within the currently active LLM provider. Use this to switch to different models (like nousresearch/hermes-4-405b, openai/gpt-4o, etc.) without changing the provider."""
    if not model.strip():
        return "Error: Model name cannot be empty"

    config = config_manager.get_config()
    active_provider, _ = config_manager.get_active_provider_config()

    config["llm"]["providers"][active_provider]["model"] = model.strip()
    config_manager.save_config(config)
    return f"Model set to '{model.strip()}' for {active_provider}"


@tool_if("set_presence_penalty")
def set_presence_penalty(penalty: float) -> str:
    """Set the presence_penalty for the active LLM provider (-2.0 to 2.0)"""
    if not MIN_PENALTY <= penalty <= MAX_PENALTY:
        return f"Error: presence_penalty must be between {MIN_PENALTY} and {MAX_PENALTY}"

    config = config_manager.get_config()
    active_provider, _ = config_manager.get_active_provider_config()

    config["llm"]["providers"][active_provider]["presence_penalty"] = penalty
    config_manager.save_config(config)
    return f"Presence penalty set to {penalty} for {active_provider}"


@tool_if("set_frequency_penalty")
def set_frequency_penalty(penalty: float) -> str:
    """Set the frequency_penalty for the active LLM provider (-2.0 to 2.0)"""
    if not MIN_PENALTY <= penalty <= MAX_PENALTY:
        return f"Error: frequency_penalty must be between {MIN_PENALTY} and {MAX_PENALTY}"

    config = config_manager.get_config()
    active_provider, _ = config_manager.get_active_provider_config()

    config["llm"]["providers"][active_provider]["frequency_penalty"] = penalty
    config_manager.save_config(config)
    return f"Frequency penalty set to {penalty} for {active_provider}"


@tool_if("set_repetition_penalty")
def set_repetition_penalty(penalty: float) -> str:
    """Set repetition_penalty for the active provider (> 0.0 recommended)"""
    if penalty <= 0:
        return "Error: repetition_penalty must be > 0"

    config = config_manager.get_config()
    active_provider, _ = config_manager.get_active_provider_config()

    config["llm"]["providers"][active_provider]["repetition_penalty"] = penalty
    config_manager.save_config(config)
    return f"Repetition penalty set to {penalty} for {active_provider}"


@tool_if("set_top_k")
def set_top_k(top_k: int) -> str:
    """Set top_k sampling parameter for the active provider (>= 0)"""
    if top_k < 0:
        return "Error: top_k must be >= 0"

    config = config_manager.get_config()
    active_provider, _ = config_manager.get_active_provider_config()

    config["llm"]["providers"][active_provider]["top_k"] = top_k
    config_manager.save_config(config)
    return f"Top-k set to {top_k} for {active_provider}"


@tool_if("set_seed")
def set_seed(seed: int) -> str:
    """Set sampling seed for the active provider (>= 0)"""
    if seed < 0:
        return "Error: seed must be >= 0"

    config = config_manager.get_config()
    active_provider, _ = config_manager.get_active_provider_config()

    config["llm"]["providers"][active_provider]["seed"] = seed
    config_manager.save_config(config)
    return f"Seed set to {seed} for {active_provider}"


@tool_if("set_min_p")
def set_min_p(min_p: float) -> str:
    """Set min_p sampling parameter for the active provider (0.0-1.0)"""
    if not 0.0 <= min_p <= 1.0:
        return "Error: min_p must be between 0.0 and 1.0"

    config = config_manager.get_config()
    active_provider, _ = config_manager.get_active_provider_config()

    config["llm"]["providers"][active_provider]["min_p"] = min_p
    config_manager.save_config(config)
    return f"min_p set to {min_p} for {active_provider}"


@tool_if("set_top_a")
def set_top_a(top_a: float) -> str:
    """Set top_a sampling parameter for the active provider (0.0-1.0)"""
    if not 0.0 <= top_a <= 1.0:
        return "Error: top_a must be between 0.0 and 1.0"

    config = config_manager.get_config()
    active_provider, _ = config_manager.get_active_provider_config()

    config["llm"]["providers"][active_provider]["top_a"] = top_a
    config_manager.save_config(config)
    return f"top_a set to {top_a} for {active_provider}"


@tool_if("set_stop_sequences")
def set_stop_sequences(sequences: str) -> str:
    """
    Set stop sequences for the active provider.

    Accepts either a JSON array string (e.g., "[\"END\", \"STOP\"]")
    or a comma/pipe-separated string (e.g., "END, STOP" or "END|STOP").
    A single token/string is also accepted.
    """
    parsed: Any
    text = sequences.strip()
    try:
        if text.startswith("["):
            parsed = json.loads(text)
        elif "," in text:
            parsed = [s.strip() for s in text.split(",") if s.strip()]
        elif "|" in text:
            parsed = [s.strip() for s in text.split("|") if s.strip()]
        else:
            parsed = text
    except Exception:
        return "Error: could not parse stop sequences; provide JSON array or comma-separated list"

    config = config_manager.get_config()
    active_provider, _ = config_manager.get_active_provider_config()
    config["llm"]["providers"][active_provider]["stop"] = parsed
    config_manager.save_config(config)
    return f"Stop sequences updated for {active_provider}: {parsed}"


@tool_if("set_response_format")
def set_response_format(format_type: str) -> str:
    """
    Set response_format for the active provider. Common values:
    - text
    - json_object
    """
    allowed = {"text", "json_object"}
    if format_type not in allowed:
        return f"Error: format_type must be one of {sorted(allowed)}"

    config = config_manager.get_config()
    active_provider, _ = config_manager.get_active_provider_config()
    config["llm"]["providers"][active_provider]["response_format"] = {"type": format_type}
    config_manager.save_config(config)
    return f"response_format set to {format_type} for {active_provider}"


@tool_if("set_structured_outputs")
def set_structured_outputs(enabled: bool) -> str:
    """Enable/disable structured_outputs for the active provider"""
    config = config_manager.get_config()
    active_provider, _ = config_manager.get_active_provider_config()
    config["llm"]["providers"][active_provider]["structured_outputs"] = enabled
    config_manager.save_config(config)
    return f"structured_outputs set to {enabled} for {active_provider}"


@tool_if("set_include_reasoning")
def set_include_reasoning(include: bool) -> str:
    """Enable/disable include_reasoning for the active provider"""
    config = config_manager.get_config()
    active_provider, _ = config_manager.get_active_provider_config()
    config["llm"]["providers"][active_provider]["include_reasoning"] = include
    config_manager.save_config(config)
    return f"include_reasoning set to {include} for {active_provider}"


@tool_if("set_reasoning")
def set_reasoning(level: str) -> str:
    """
    Set reasoning option for the active provider (provider-specific).
    Typical values: low, medium, high.
    """
    normalized = level.strip().lower()
    if not normalized:
        return "Error: reasoning level cannot be empty"

    config = config_manager.get_config()
    active_provider, _ = config_manager.get_active_provider_config()
    config["llm"]["providers"][active_provider]["reasoning"] = normalized
    config_manager.save_config(config)
    return f"reasoning set to {normalized} for {active_provider}"


@tool_if("reset_runtime_config_to_defaults")
def reset_runtime_config_to_defaults() -> str:
    """
    Reset runtime_config.yaml to defaults from config.yaml.
    Writes defaults and updates metadata to trigger reload.
    """
    try:
        default_path = config_manager.default_config_path
        runtime_path = config_manager.runtime_config_path

        if not default_path.exists():
            return f"Error: Default config file not found at {default_path}"

        default_cfg = yaml.safe_load(default_path.read_text())
        if not isinstance(default_cfg, dict):
            return "Error: Default config is not a valid mapping"

        # Preserve and bump version if present to aid change tracking
        current_version: int = 0
        try:
            existing_raw = yaml.safe_load(runtime_path.read_text())
            existing: dict[str, Any] = existing_raw if isinstance(existing_raw, dict) else {}
            runtime_meta: dict[str, Any] = cast(dict[str, Any], existing.get("_runtime_config", {}))
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


# Logging configuration tools


@tool_if("get_logging_config")
def get_logging_config() -> str:
    """Get the current logging configuration"""
    config = config_manager.get_config()
    logging_config = config.get("logging", {})

    if not logging_config:
        return "No logging configuration found"

    # Format global logging settings
    global_settings = []
    if "level" in logging_config:
        global_settings.append(f"Global level: {logging_config['level']}")
    if "format" in logging_config:
        global_settings.append(f"Format: {logging_config['format']}")

    # Format advanced settings
    advanced = logging_config.get("advanced", {})
    if advanced:
        advanced_settings = []
        for key, value in advanced.items():
            advanced_settings.append(f"  {key}: {value}")
        if advanced_settings:
            global_settings.append("Advanced settings:")
            global_settings.extend(advanced_settings)

    # Format module-specific settings
    modules = logging_config.get("modules", {})
    module_settings = []
    for module_name, module_config in modules.items():
        module_settings.append(f"Module '{module_name}':")
        if "level" in module_config:
            module_settings.append(f"  Level: {module_config['level']}")

        features = module_config.get("enable_features", {})
        if features:
            enabled_features = [f for f, enabled in features.items() if enabled]
            if enabled_features:
                module_settings.append(f"  Enabled features: {', '.join(enabled_features)}")

        truncate_lengths = module_config.get("truncate_lengths", {})
        if truncate_lengths:
            truncates = [f"{k}: {v}" for k, v in truncate_lengths.items()]
            module_settings.append(f"  Truncate lengths: {', '.join(truncates)}")

        # Add other module-specific settings
        for key, value in module_config.items():
            if key not in ["level", "enable_features", "truncate_lengths"]:
                module_settings.append(f"  {key}: {value}")

    # Combine all sections
    result_parts = []
    if global_settings:
        result_parts.append("Global logging settings:")
        result_parts.extend(global_settings)

    if module_settings:
        if result_parts:
            result_parts.append("")
        result_parts.append("Module-specific settings:")
        result_parts.extend(module_settings)

    return "\n".join(result_parts) if result_parts else "Empty logging configuration"


@tool_if("set_global_log_level")
def set_global_log_level(level: str) -> str:
    """
    Set the global logging level.

    Valid levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    """
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    normalized_level = level.upper().strip()

    if normalized_level not in valid_levels:
        return f"Error: Invalid log level '{level}'. Valid levels: {', '.join(sorted(valid_levels))}"

    config = config_manager.get_config()

    # Ensure logging section exists
    if "logging" not in config:
        config["logging"] = {}

    config["logging"]["level"] = normalized_level
    config_manager.save_config(config)
    return f"Global log level set to {normalized_level}"


@tool_if("set_module_log_level")
def set_module_log_level(module: str, level: str) -> str:
    """
    Set the logging level for a specific module.

    Args:
        module: Module name (e.g., 'chat', 'mcp', 'connection_pool')
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    normalized_level = level.upper().strip()

    if normalized_level not in valid_levels:
        return f"Error: Invalid log level '{level}'. Valid levels: {', '.join(sorted(valid_levels))}"

    if not module.strip():
        return "Error: Module name cannot be empty"

    normalized_module = module.strip()

    config = config_manager.get_config()

    # Ensure logging section exists
    if "logging" not in config:
        config["logging"] = {}

    # Ensure modules section exists
    if "modules" not in config["logging"]:
        config["logging"]["modules"] = {}

    # Ensure the specific module section exists
    if normalized_module not in config["logging"]["modules"]:
        config["logging"]["modules"][normalized_module] = {}

    config["logging"]["modules"][normalized_module]["level"] = normalized_level
    config_manager.save_config(config)

    # Refresh feature flags in the logging module for immediate effect
    _refresh_logging_feature_flags(config["logging"])

    return f"Log level for module '{normalized_module}' set to {normalized_level}"


def _refresh_logging_feature_flags(logging_config: dict) -> None:
    """Refresh the feature flags in the logging module for immediate effect."""

    # Module-to-logger mapping (copied from main.py for consistency)

    modules_config = logging_config.get("modules", {})

    # Refresh feature flags for each module
    for module_name, module_config in modules_config.items():
        if not isinstance(module_config, dict):
            continue

        # Store feature flags for runtime checking
        if not hasattr(logging, "_module_features"):
            logging._module_features = {}
        logging._module_features[module_name] = module_config.get("enable_features", {})


@tool_if("set_logging_feature")
def set_logging_feature(module: str, feature: str, enabled: bool) -> str:
    """
    Enable or disable a specific logging feature for a module.

    Args:
        module: Module name (e.g., 'chat', 'mcp', 'connection_pool')
        feature: Feature name (e.g., 'llm_replies', 'tool_arguments', 'tool_results')
        enabled: True to enable, False to disable
    """
    if not module.strip():
        return "Error: Module name cannot be empty"

    if not feature.strip():
        return "Error: Feature name cannot be empty"

    normalized_module = module.strip()
    normalized_feature = feature.strip()

    config = config_manager.get_config()

    # Ensure logging section exists
    if "logging" not in config:
        config["logging"] = {}

    # Ensure modules section exists
    if "modules" not in config["logging"]:
        config["logging"]["modules"] = {}

    # Ensure the specific module section exists
    if normalized_module not in config["logging"]["modules"]:
        config["logging"]["modules"][normalized_module] = {}

    # Ensure enable_features section exists
    if "enable_features" not in config["logging"]["modules"][normalized_module]:
        config["logging"]["modules"][normalized_module]["enable_features"] = {}

    config["logging"]["modules"][normalized_module]["enable_features"][normalized_feature] = enabled
    config_manager.save_config(config)

    # Refresh feature flags in the logging module for immediate effect
    _refresh_logging_feature_flags(config["logging"])

    action = "enabled" if enabled else "disabled"
    return f"Feature '{normalized_feature}' {action} for module '{normalized_module}'"


@tool_if("set_logging_format")
def set_logging_format(format_string: str) -> str:
    """
    Set the global logging format string.

    The format string uses Python's logging format syntax.
    Common placeholders: %(asctime)s, %(levelname)s, %(message)s, %(name)s, etc.

    Example: '%(asctime)s - %(levelname)s - %(message)s'
    """
    if not format_string.strip():
        return "Error: Format string cannot be empty"

    config = config_manager.get_config()

    # Ensure logging section exists
    if "logging" not in config:
        config["logging"] = {}

    config["logging"]["format"] = format_string.strip()
    config_manager.save_config(config)
    return f"Global log format set to: {format_string.strip()}"


@tool_if("set_advanced_logging_option")
def set_advanced_logging_option(option: str, value: str) -> str:
    """
    Set an advanced logging option.

    Args:
        option: Option name (e.g., 'async_logging', 'buffer_size', 'log_to_file', 'structured_logging')
        value: Option value (will be parsed as bool, int, or string as appropriate)

    Examples:
        set_advanced_logging_option('async_logging', 'true')
        set_advanced_logging_option('buffer_size', '1000')
        set_advanced_logging_option('log_to_file', 'false')
    """
    if not option.strip():
        return "Error: Option name cannot be empty"

    if value is None:
        return "Error: Option value cannot be None"

    normalized_option = option.strip()
    value_str = str(value).strip()

    # Parse the value based on common patterns
    parsed_value: Any
    lower_value = value_str.lower()

    # Handle boolean values
    if lower_value in ("true", "false"):
        parsed_value = lower_value == "true"
    # Handle integer values
    elif value_str.isdigit() or (value_str.startswith("-") and value_str[1:].isdigit()):
        try:
            parsed_value = int(value_str)
        except ValueError:
            parsed_value = value_str  # Fall back to string
    # Handle string values
    else:
        parsed_value = value_str

    config = config_manager.get_config()

    # Ensure logging section exists
    if "logging" not in config:
        config["logging"] = {}

    # Ensure advanced section exists
    if "advanced" not in config["logging"]:
        config["logging"]["advanced"] = {}

    config["logging"]["advanced"][normalized_option] = parsed_value
    config_manager.save_config(config)
    return f"Advanced logging option '{normalized_option}' set to {parsed_value}"


if __name__ == "__main__":
    # Configure logging
    # Logging is already configured in main.py

    # Log which tools are enabled
    enabled_tools = [k for k, v in TOOL_TOGGLES.items() if v]
    logging.info(f"Config Server starting with {len(enabled_tools)} enabled tools: {', '.join(enabled_tools)}")
    mcp.run()
