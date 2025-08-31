"""
Chat Service Data Models

Comprehensive data structures for chat functionality including LLM API types,
tool definitions, streaming models, and internal orchestration models.
All strongly typed with Pydantic for validation and type safety.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator

if TYPE_CHECKING:
    pass


# ==============================================================================
# CORE CHAT MESSAGES (LLM API Types)
# ==============================================================================


class BaseChatMessage(BaseModel):
    """Base class for all chat messages."""

    role: str
    content: str | None = None


class SystemMessage(BaseModel):
    """System message for setting context."""

    role: Literal["system"] = "system"
    content: str


class UserMessage(BaseModel):
    """User message."""

    role: Literal["user"] = "user"
    content: str


class FunctionCall(BaseModel):
    """Function call within a tool call."""

    name: str
    arguments: str = Field(default="{}")  # JSON string


class ToolCall(BaseModel):
    """Tool call from LLM."""

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class AssistantMessage(BaseModel):
    """Assistant message with optional tool calls."""

    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None

    @field_validator("tool_calls")
    @classmethod
    def validate_tool_calls(cls, v: list[ToolCall] | None) -> list[ToolCall] | None:
        """Convert empty tool_calls list to None to avoid API errors."""
        if v is not None and len(v) == 0:
            return None
        return v

    @field_validator("content", "tool_calls")
    @classmethod
    def validate_content_or_tool_calls(cls, v: Any, _info: ValidationInfo) -> Any:
        """Ensure either content or tool_calls is present (relaxed validation)."""
        # Skip validation during model creation for flexibility
        return v

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssistantMessage:
        """Create AssistantMessage from dict (for LLM client compatibility)."""
        # Handle tool_calls conversion if present
        tool_calls = None
        if data.get("tool_calls"):
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    type=tc.get("type", "function"),
                    function=FunctionCall(
                        name=tc["function"]["name"],
                        arguments=tc["function"].get("arguments", "{}"),
                    ),
                )
                for tc in data["tool_calls"]
            ]

        return cls(
            content=data.get("content"),
            tool_calls=tool_calls,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict format for LLM client compatibility."""
        result: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        return result


class ToolMessage(BaseModel):
    """Tool response message."""

    role: Literal["tool"] = "tool"
    content: str
    tool_call_id: str


# Union of all message types for conversation
ChatCompletionMessage = SystemMessage | UserMessage | AssistantMessage | ToolMessage


# ==============================================================================
# TOOL DEFINITIONS AND SCHEMAS
# ==============================================================================


class ToolFunctionParameters(BaseModel):
    """Function parameters schema for tools."""

    type: Literal["object"] = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)
    additionalProperties: bool = False


class ToolFunctionDefinition(BaseModel):
    """Tool function definition."""

    name: str
    description: str
    parameters: ToolFunctionParameters


class ToolDefinition(BaseModel):
    """Complete tool definition for OpenAI API."""

    type: Literal["function"] = "function"
    function: ToolFunctionDefinition


class ToolExecutionResult(BaseModel):
    """Result of tool execution."""

    tool_call_id: str
    content: str
    success: bool = True
    error: str | None = None


class ToolCallAccumulation(BaseModel):
    """Accumulated tool calls during streaming."""

    calls: list[ToolCall] = Field(default_factory=list)  # type: ignore
    complete: bool = False

    def add_delta(self, tool_calls: list[ToolCall]) -> None:
        """Add tool call delta to accumulation."""
        for new_call in tool_calls:
            # Find existing call with same ID or add new one
            existing_call = next((call for call in self.calls if call.id == new_call.id), None)
            if existing_call:
                # Update existing call
                if new_call.function.name:
                    existing_call.function.name = new_call.function.name
                if new_call.function.arguments:
                    existing_call.function.arguments += new_call.function.arguments
            else:
                # Add new call
                self.calls.append(new_call)


# ==============================================================================
# REQUEST/RESPONSE MODELS
# ==============================================================================


class ChatCompletionRequest(BaseModel):
    """Chat completion request payload."""

    model: str
    messages: list[ChatCompletionMessage]
    tools: list[ToolDefinition] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stop: list[str] | str | None = None

    # Additional model-specific parameters
    top_k: int | None = None
    repetition_penalty: float | None = None

    # Model configuration for various providers
    model_config = {"extra": "allow"}  # Allow additional provider-specific parameters


class ChatCompletionChoice(BaseModel):
    """Single choice in chat completion response."""

    index: int
    message: AssistantMessage
    finish_reason: str | None = None


class TokenUsage(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Complete chat completion response."""

    id: str | None = None
    object: Literal["chat.completion"] = "chat.completion"
    created: int | None = None
    model: str
    choices: list[ChatCompletionChoice]
    usage: TokenUsage | None = None
    thinking: str | None = None  # For reasoning models


class LLMResponseData(BaseModel):
    """Structured LLM response data."""

    message: AssistantMessage
    finish_reason: str | None = None
    index: int = 0
    model: str
    thinking: str | None = None  # For reasoning models


# ==============================================================================
# STREAMING MODELS
# ==============================================================================


class FunctionCallDelta(BaseModel):
    """Partial function call data in streaming response."""

    name: str | None = None
    arguments: str | None = None


class ToolCallDelta(BaseModel):
    """Partial tool call data in streaming response."""

    index: int | None = None
    id: str | None = None
    type: Literal["function"] | None = None
    function: FunctionCallDelta | None = None


class StreamingDelta(BaseModel):
    """Delta content in streaming response."""

    role: str | None = None
    content: str | None = None
    tool_calls: list[ToolCallDelta] | None = None

    # Additional reasoning fields
    thinking: str | None = None
    reasoning: str | None = None


class StreamingChoice(BaseModel):
    """Single choice in streaming response."""

    index: int
    delta: StreamingDelta
    finish_reason: str | None = None


class StreamingChunk(BaseModel):
    """Single chunk in streaming response."""

    id: str | None = None
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int | None = None
    model: str | None = None
    choices: list[StreamingChoice]

    # Additional fields for reasoning models
    thinking_complete: bool | None = None


class ThinkingChunk(BaseModel):
    """Thinking/reasoning content chunk."""

    type: Literal["thinking"] = "thinking"
    content: str
    complete: bool = False
    full_thinking: str | None = None  # Complete thinking when done


# Union type for streaming responses
StreamingResponse = StreamingChunk | ThinkingChunk


# ==============================================================================
# CONVERSATION MANAGEMENT
# ==============================================================================


class ConversationHistory(BaseModel):
    """Complete conversation history."""

    system_prompt: SystemMessage | None = None
    messages: list[ChatCompletionMessage] = Field(default_factory=list)  # type: ignore

    def add_message(self, message: ChatCompletionMessage) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)

    def get_api_format(self) -> list[ChatCompletionMessage]:
        """Get conversation in API format."""
        result: list[ChatCompletionMessage] = []
        if self.system_prompt:
            result.append(self.system_prompt)
        result.extend(self.messages)
        return result

    def get_dict_format(self) -> list[dict[str, Any]]:
        """Get conversation in dictionary format for LLM client."""
        result: list[dict[str, Any]] = []
        if self.system_prompt:
            result.append(self.system_prompt.model_dump())

        for msg in self.messages:
            # Use custom to_dict() for AssistantMessage to handle empty tool_calls
            if isinstance(msg, AssistantMessage):
                result.append(msg.to_dict())
            else:
                result.append(msg.model_dump())
        return result


class ChatSessionContext(BaseModel):
    """Context for a chat session."""

    conversation_id: str
    request_id: str
    user_message: str
    conversation: ConversationHistory
    tools: list[ToolDefinition] = Field(default_factory=list)  # type: ignore
    max_tool_hops: int = 8
    current_hop: int = 0


# ==============================================================================
# INTERNAL ORCHESTRATION MODELS
# ==============================================================================


class ChatMessage(BaseModel):
    """
    Frontend/UI message type for streaming responses.
    Different from ChatCompletionMessage which is for LLM API.
    """

    type: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallContext(BaseModel):
    """Parameters for tool call iteration handling."""

    conv: ConversationHistory
    tools_payload: list[ToolDefinition]
    conversation_id: str
    request_id: str
    assistant_msg: AssistantMessage
    full_content: str


class ToolExecutionContext(BaseModel):
    """Context for executing tool calls."""

    conversation: ConversationHistory
    tools: list[ToolDefinition]
    conversation_id: str
    request_id: str
    max_hops: int = 8
    current_hop: int = 0


class ChatResponseMetadata(BaseModel):
    """Metadata for chat responses."""

    model: str
    finish_reason: str | None = None
    thinking: str | None = None  # For reasoning models
    token_usage: dict[str, int] | None = None
    tool_calls_made: int = 0
    total_hops: int = 0


# ==============================================================================
# CONFIGURATION MODELS
# ==============================================================================


class LLMClientConfig(BaseModel):
    """LLM client configuration."""

    model: str
    base_url: str
    timeout: float = 60.0
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None

    # Provider-specific settings
    model_config = {"extra": "allow"}
