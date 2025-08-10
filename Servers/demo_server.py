"""
FastMCP Desktop Example

A simple example that exposes the desktop directory as a resource,
demonstrates tools, and includes example prompts.
"""

from __future__ import annotations

import logging
from pathlib import Path

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

# Create server
mcp = FastMCP("Demo")


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
        desktop = Path("/home/jack/Documents/MCP.resources")
        files = [f.name for f in desktop.iterdir() if f.is_file()]
        return "\n".join(f"- {file}" for file in sorted(files))


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
        return a ** b

    @mcp.tool()
    def divide(a: float, b: float) -> float:
        """Divide first number by second"""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b


# Configuration tools
if TOOL_CONFIG["config_tools"]:
    def _load_config() -> dict:
        """Helper to load and parse the runtime config"""
        config_path = Path("/home/jack/MCP_BACKEND_OPENROUTER/src/runtime_config.yaml")
        config_text = config_path.read_text()
        result = yaml.safe_load(config_text)
        return result if isinstance(result, dict) else {}

    def _get_active_provider_config() -> tuple[str, dict]:
        """Helper to get active provider name and its config"""
        config = _load_config()
        llm_config = config.get("llm", {}) if isinstance(config, dict) else {}
        active_provider = llm_config.get("active", "groq")
        providers = llm_config.get("providers", {})
        provider_config = providers.get(active_provider, {})
        return active_provider, provider_config

    @mcp.tool()
    def get_system_prompt() -> str:
        """Get the current system prompt configuration"""
        config = _load_config()
        chat_config = (
            config.get("chat", {}).get("service", {})
            if isinstance(config, dict) else {}
        )
        system_prompt = chat_config.get("system_prompt", "No system prompt configured")
        return f"System prompt: {system_prompt}"

    @mcp.tool()
    def get_sampling_parameters() -> str:
        """Get sampling parameters (temperature, top_p, top_k) for active provider"""
        active_provider, provider_config = _get_active_provider_config()

        temperature = provider_config.get("temperature", "not set")
        top_p = provider_config.get("top_p", "not set")
        top_k = provider_config.get("top_k", "not set")

        return (f"Sampling parameters for {active_provider}:\n"
                f"- Temperature: {temperature}\n"
                f"- Top-p: {top_p}\n"
                f"- Top-k: {top_k}")

    @mcp.tool()
    def get_length_parameters() -> str:
        """Get length parameters (max_tokens, min_length) for active provider"""
        active_provider, provider_config = _get_active_provider_config()

        max_tokens = provider_config.get("max_tokens", "not set")
        min_length = provider_config.get("min_length", "not set")
        max_completion_tokens = provider_config.get("max_completion_tokens", "not set")

        return (f"Length parameters for {active_provider}:\n"
                f"- Max tokens: {max_tokens}\n"
                f"- Min length: {min_length}\n"
                f"- Max completion tokens: {max_completion_tokens}")

    @mcp.tool()
    def get_penalty_parameters() -> str:
        """Get penalty parameters for active provider"""
        active_provider, provider_config = _get_active_provider_config()

        presence_penalty = provider_config.get("presence_penalty", "not set")
        frequency_penalty = provider_config.get("frequency_penalty", "not set")
        repetition_penalty = provider_config.get("repetition_penalty", "not set")

        return (f"Penalty parameters for {active_provider}:\n"
                f"- Presence penalty: {presence_penalty}\n"
                f"- Frequency penalty: {frequency_penalty}\n"
                f"- Repetition penalty: {repetition_penalty}")

    @mcp.tool()
    def get_stopping_criteria() -> str:
        """Get stopping criteria (stop sequences, max tokens) for active provider"""
        active_provider, provider_config = _get_active_provider_config()

        stop_sequences = provider_config.get("stop", "not set")
        stop_tokens = provider_config.get("stop_tokens", "not set")
        end_of_text = provider_config.get("end_of_text", "not set")

        return (f"Stopping criteria for {active_provider}:\n"
                f"- Stop sequences: {stop_sequences}\n"
                f"- Stop tokens: {stop_tokens}\n"
                f"- End of text: {end_of_text}")


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
