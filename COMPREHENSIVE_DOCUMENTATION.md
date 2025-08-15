# MCP Backend OpenRouter - Comprehensive Documentation

This document provides a complete guide to the MCP (Model Context Protocol) chatbot platform, covering architecture, configuration, frontend integration, and usage.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Runtime Configuration System](#runtime-configuration-system)
4. [Flexible LLM Client](#flexible-llm-client)
5. [Frontend Integration](#frontend-integration)
6. [Development Guidelines](#development-guidelines)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The MCP Backend OpenRouter is a sophisticated chatbot platform that connects to various MCP servers and uses LLM APIs for responses. It features:

- **Modular Architecture**: Clear separation of concerns with isolated complexity
- **Dynamic Configuration**: Runtime configuration changes without restarts
- **Flexible LLM Support**: Automatic parameter pass-through for any model
- **Real-time Communication**: WebSocket-based frontend integration
- **Persistent Storage**: Automatic conversation persistence with configurable retention

### Key Technologies

- **MCP SDK**: Model Context Protocol for tool integration
- **Pydantic**: Data validation and type safety
- **WebSocket**: Real-time communication
- **SQLite**: Chat history persistence
- **YAML**: Configuration management

---

## Architecture

### Core Principle: Complexity Isolation

The system follows a modular design where complex/fragile modules are isolated for better error handling, while mundane/stable modules handle straightforward business logic.

### Project Structure

```
src/
├── application.py              # Main application entry point
├── config.py                   # Configuration management with dynamic reload
├── tool_schema_manager.py      # Tool schema validation and conversion
├── websocket_server.py         # WebSocket communication layer
├── config.yaml                 # Default configuration (reference)
├── runtime_config.yaml         # Runtime configuration (persistent)
└── servers_config.json         # MCP server configurations

src/chat/                       # Modular chat system
├── __init__.py                 # Exports: ChatOrchestrator, ChatMessage, ToolCallContext
├── models.py                   # Stable data structures
├── conversation_manager.py     # REMOVED - Logic moved to history module
├── tool_executor.py            # Complex/fragile tool execution
├── streaming_handler.py        # Complex/fragile streaming logic
├── simple_chat_handler.py      # Mundane non-streaming operations
├── resource_loader.py          # Fragile resource loading
└── chat_orchestrator.py        # Main coordinator

src/clients/                    # Client integrations
├── __init__.py                 # Client exports
├── llm_client.py              # Flexible LLM API client
└── mcp_client.py              # MCP server client

src/history/                    # Storage system
├── __init__.py                 # Repository factory and exports
├── models.py                  # Data models
├── repository.py              # Storage interface
├── sqlite_repo.py             # SQLite implementation
├── memory_repo.py             # In-memory implementation
├── auto_persist_repo.py       # Auto-persistence wrapper
├── factory.py                 # Repository factory
└── utils.py                   # Storage utilities

Servers/                       # Ready for new FastMCP servers
├── demo_server.py             # Demo MCP server
├── demo_prompt_server.py      # Prompt-only demo server
└── prompt_options.template.md # Prompt template

Scripts/                       # Utility scripts
└── format.py                  # Code formatting script
```

### Chat Module Responsibilities

#### 1. **models.py** (Stable)
- Core data structures: `ChatMessage`, `ToolCallContext`
- Usage statistics conversion
- Pure data models with high stability

#### 2. **conversation_manager.py** (REMOVED)
- **Previous Purpose**: User message persistence, conversation history building, repository interactions
- **Status**: ❌ **ELIMINATED** - All functionality moved to history repository layer  
- **Replacement**: Direct repository usage in ChatOrchestrator and handlers

#### 3. **tool_executor.py** (Complex/Fragile)
- Tool call accumulation from streaming deltas
- Tool execution with error handling
- Tool result formatting and hop limit checking
- **Logging**: `→ MCP[server]: calling tool` / `← MCP[server]: result`

#### 4. **streaming_handler.py** (Complex/Fragile)
- LLM response streaming with delta persistence
- Tool call iteration handling
- Hybrid message yielding (ChatMessage + dict)
- **Logging**: `→ Frontend: streaming content` / `→ LLM: streaming request`

#### 5. **simple_chat_handler.py** (Mundane/Stable)
- Non-streaming request/response flow
- Basic tool call execution without delta complexity
- Straightforward LLM interactions

#### 6. **resource_loader.py** (Fragile)
- Resource availability checking and catalog management
- System prompt building with resources
- **Logging**: `→ Resources: checking availability`

#### 7. **chat_orchestrator.py** (Main Coordinator)
- Component initialization and request routing
- High-level flow control
- **Logging**: `→ Orchestrator:` / `← Orchestrator:`

---

## Runtime Configuration System

### Overview

The system uses a **persistent runtime configuration architecture** with:
- `src/config.yaml` - Reference/Default Configuration (never modified)
- `src/runtime_config.yaml` - Persistent Runtime Configuration (primary source)

### Key Features

- **🔄 Dynamic Updates**: Configuration changes applied immediately without restart
- **💾 Persistent Changes**: All changes persist across system restarts
- **🔍 Automatic Detection**: Automatic file modification detection and reload
- **🔗 Deep Merging**: Partial configuration updates supported
- **🛡️ Automatic Recovery**: Auto-recreation from defaults if corrupted
- **📊 Metadata Tracking**: Version numbers and modification timestamps

### Configuration Files

#### Default Configuration (`src/config.yaml`)
```yaml
# MCP Chatbot Configuration
chat:
  websocket:
    host: "localhost"
    port: 8000
    endpoint: "/ws/chat"
    allow_origins: ["*"]
    allow_credentials: true
    max_message_size: 16777216  # 16MB
    ping_interval: 20
    ping_timeout: 10

  storage:
    type: "auto_persist"
    persistence:
      db_path: "chat_history.db"
      retention:
        max_age_hours: 24
        max_messages: 1000
        max_sessions: 2
        cleanup_interval_minutes: 2
    saved_sessions:
      enabled: true
      retention_days: null
      max_saved: 50

  service:
    system_prompt: |
      You are a helpful assistant with a sense of humor.
      You have access to to a list of tools like setting your own configuration.
    streaming:
      enabled: true
    max_tool_hops: 8
    tool_notifications:
      enabled: true
      show_args: true
      icon: "🔧"
      format: "{icon} Executing tool: {tool_name}"

llm:
  active: "openrouter"
  providers:
    openrouter:
      base_url: "https://openrouter.ai/api/v1"
      model: "openai/gpt-4o-mini"
      temperature: 0.7
      max_tokens: 4096
      top_p: 1.0
      transforms: ["middle-out"]

mcp:
  config_file: "servers_config.json"
  connection:
    max_reconnect_attempts: 5
    initial_reconnect_delay: 1.0
    max_reconnect_delay: 30.0
    connection_timeout: 30.0
    ping_timeout: 10.0
```

#### Runtime Configuration (`src/runtime_config.yaml`)
```yaml
# Runtime overrides - only specify what you want to change
chat:
  websocket:
    port: 8080  # Override port
  service:
    max_tool_hops: 12  # Override max tool hops

llm:
  active: "groq"  # Switch to different provider

# Metadata (automatically managed)
_runtime_config:
  last_modified: 1755070820.785298
  version: 11
  is_runtime_config: true
  default_config_path: "config.yaml"
  created_from_defaults: true
```

### Usage Examples

#### Programmatic Updates
```python
from src.config import Configuration

config = Configuration()

# Modify specific values
current_config = config.get_config_dict()
current_config['chat']['websocket']['port'] = 9000
current_config['llm']['active'] = 'groq'

# Save changes (automatically reloads)
config.save_runtime_config(current_config)
```

#### Manual File Editing
Directly edit `src/runtime_config.yaml` - changes are automatically detected and applied.

---

## Flexible LLM Client

### Overview

The `LLMClient` is designed to be "ready for anything" - it automatically passes through all configuration parameters to any OpenAI-compatible API, making it compatible with new models without requiring code changes.

### Key Features

- **🔄 Automatic Parameter Pass-Through**: Any config parameter is automatically sent to the API
- **🧠 Reasoning Model Support**: Automatic detection and extraction of thinking/reasoning content
- **🎯 Provider Flexibility**: Works with OpenAI, OpenRouter, Groq, and any OpenAI-compatible provider
- **🚀 Future-Proof**: Compatible with models that don't exist yet

### Configuration Examples

#### Standard Model
```yaml
openai:
  base_url: "https://api.openai.com/v1"
  model: "gpt-4o-mini"
  temperature: 0.7
  max_tokens: 4096
  top_p: 1.0
```

#### Reasoning Model (o1/o3)
```yaml
openai_reasoning:
  base_url: "https://api.openai.com/v1"
  model: "o1-preview"
  max_completion_tokens: 8192  # o1 specific parameter
  # No temperature - o1 uses fixed temperature
```

#### Future Model with Custom Parameters
```yaml
future_model:
  base_url: "https://api.newprovider.com/v1"
  model: "amazing-model-v5"
  temperature: 0.8
  max_tokens: 16384
  # Any new parameters automatically passed through
  new_parameter: "value"
  reasoning_mode: "advanced"
  custom_setting: 42
```

### How It Works

#### Parameter Pass-Through
```python
# The client automatically includes ALL config parameters
payload = {
    "model": self.config["model"],
    "messages": messages,
}

# Add ALL other parameters from config
for key, value in self.config.items():
    if key not in ["base_url", "model"]:  # Skip infrastructure keys
        payload[key] = value
```

#### Reasoning Content Extraction
```python
# Checks multiple possible locations for thinking content
possible_reasoning_fields = [
    "thinking", "reasoning", "thought_process", 
    "internal_thoughts", "chain_of_thought", "rationale"
]
```

### Response Format
```python
{
    "message": {...},           # Standard message
    "thinking": "...",          # Reasoning content (if present)
    "usage": {...},            # Token usage
    "model": "...",            # Model used
    "finish_reason": "..."     # Completion reason
}
```

### Adding New Models

1. **Add to Config**: Just add new parameters to your YAML configuration
2. **That's It**: The client automatically handles everything else

---

## Frontend Integration

### Overview

The system provides WebSocket-based real-time communication with automatic conversation persistence and user-controlled session management.

### Storage Architecture

- **Auto-Persist Mode**: Automatic conversation persistence with configurable retention
- **User-Controlled Sessions**: No automatic session clearing on reconnect
- **Manual Session Management**: User can clear history or save important conversations

### WebSocket Message Types

#### 1. Clear History (Start New Session)

**Message Format**:
```json
{
  "type": "clear_history"
}
```

**Frontend Implementation**:
```javascript
function clearHistory() {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({
            "type": "clear_history"
        }));
        
        clearChatMessages();
        showNotification("New session started");
    }
}
```

#### 2. Save Session (Future Implementation)

**Message Format**:
```json
{
  "type": "save_session",
  "name": "My Important Conversation",
  "conversation_id": "current-conv-id"
}
```

#### 3. Standard Chat Message

**Message Format**:
```json
{
  "type": "user_message",
  "message": "Hello, how can you help me today?"
}
```

**Response Format**:
```json
{
  "type": "assistant_message",
  "message": "I'm here to help! What would you like to know?",
  "thinking": "The user is greeting me...",  // Optional reasoning
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 12,
    "total_tokens": 27
  }
}
```

### UI Implementation Examples

#### Clear History Button
```html
<div class="chat-controls">
    <button id="clearHistoryBtn" class="btn-clear" title="Start New Conversation">
        <span class="icon">🔄</span>
        <span class="text">New Chat</span>
    </button>
</div>
```

#### WebSocket Message Handling
```javascript
websocket.onmessage = function(event) {
    const message = JSON.parse(event.data);
    
    switch (message.type) {
        case 'session_cleared':
            clearChatMessages();
            showNotification('New conversation started');
            break;
            
        case 'assistant_message':
            displayMessage(message.message, 'assistant');
            if (message.thinking) {
                displayThinking(message.thinking);
            }
            break;
            
        case 'tool_execution':
            showToolExecution(message.tool_name, message.status);
            break;
            
        case 'error':
            handleError(message.error_message);
            break;
    }
};
```

### Error Handling
```javascript
websocket.onmessage = function(event) {
    const message = JSON.parse(event.data);
    
    if (message.type === 'error') {
        switch (message.error_code) {
            case 'STORAGE_FULL':
                showError('Storage full - please contact administrator');
                break;
            case 'SESSION_NOT_FOUND':
                showError('Saved session not found - it may have expired');
                break;
            case 'MCP_SERVER_UNAVAILABLE':
                showError('Some tools are temporarily unavailable');
                break;
            default:
                showError(message.error_message || 'An error occurred');
        }
    }
};
```

---

## Development Guidelines

### Package Management & Scripts

The project uses `uv` for dependency management and defines several entry points:

#### Available Scripts
```bash
# Start the MCP platform
uv run python src/application.py

# Reset runtime configuration to defaults
uv run mcp-reset-config

# Format code using ruff
uv run python scripts/format.py
```

#### Package Management
- **ALWAYS** use `uv` for dependency management
- **ALWAYS** use `uv run` to execute Python commands
- Never use `pip` or other package managers

### Code Quality Standards
- **REQUIRED**: Use Pydantic for all data models and validation
- **REQUIRED**: Include comprehensive type hints
- **REQUIRED**: Follow fail-fast principle - no fallbacks or silent error handling
- **REQUIRED**: Remove all legacy/deprecated code when adding new code

### MCP SDK Integration
- Use official MCP SDK types from `mcp.types`
- Use modern union syntax: `str | None` instead of `Union[str, None]`
- Error handling with `mcp.types.INVALID_PARAMS`, `mcp.types.INTERNAL_ERROR`
- Always chain exceptions with `from e` or `from None`

### Import Pattern
```python
from __future__ import annotations  # Always first
import standard_library
import third_party
import mcp.types as types
from local_modules import items
```

### Error Handling Pattern
```python
try:
    # operation
except SpecificError as e:
    logger.error(f"Specific error: {e}")
    raise McpError(
        error=types.ErrorData(
            code=types.INVALID_PARAMS,
            message="Clear error message",
        )
    ) from e
```

### Function Signatures
```python
async def function_name(
    param1: str,
    param2: int | None = None,
) -> ReturnType:
    """Clear docstring describing purpose."""
```

### Testing Philosophy
- Create test code only when explicitly requested
- **ALWAYS** delete test code immediately after verification
- Never commit temporary test files

---

## Troubleshooting

### Common Issues

#### Configuration Not Updating
1. Check file permissions on `src/runtime_config.yaml`
2. Verify YAML syntax using a validator
3. Check system logs for error messages
4. Try manual reload: `config.reload_runtime_config()`

#### WebSocket Connection Issues
1. Verify server is running: `uv run python src/application.py`
2. Check port configuration in runtime config
3. Ensure firewall allows WebSocket connections
4. Verify frontend WebSocket URL matches server config

#### MCP Server Connection Failures
1. Check `servers_config.json` for correct server configurations
2. Verify MCP servers are running and accessible
3. Check tool schema validation in logs
4. Use `tool_schema_manager.py` to debug schema issues

#### LLM API Issues
1. Verify API keys in environment variables
2. Check base URL configuration for your provider
3. Validate model names and parameter compatibility
4. Monitor rate limits and quotas

### Debugging Tips

#### Enable Detailed Logging
```yaml
# In runtime_config.yaml
logging:
  level: "DEBUG"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

#### Test Components Individually
```python
# Test configuration
from src.config import Configuration
config = Configuration()
print(config.get_config_dict())

# Test LLM client
from src.clients.llm_client import LLMClient
llm = LLMClient(config.get_llm_config())

# Test MCP connection
from src.clients.mcp_client import MCPClient
mcp = MCPClient()
```

#### Monitor Resource Usage
```bash
# Check database size
ls -lh chat_history.db

# Monitor WebSocket connections
netstat -an | grep 8000

# Check memory usage
ps aux | grep python
```

### Performance Optimization

#### Configuration Access
- File modification time checking has minimal overhead
- Configuration is cached between checks
- Consider access frequency in high-throughput scenarios

#### Storage Management
- Configure retention policies in chat storage config
- Monitor database growth
- Use auto-persist repository for optimal performance

#### Memory Management
- Streaming handlers release resources automatically
- Tool execution contexts are cleaned up after use
- Consider message history limits for long conversations

---

#### MCP Server Configuration (`servers_config.json`)
```json
{
  "_comment": "MCP Platform Server Configuration - Primary configuration location",
  "_usage": "Add or remove MCP servers here. Tool configuration is handled by each server",
  "mcpServers": {
    "demo": {
      "enabled": true,
      "command": "uv",
      "args": ["run", "python", "Servers/demo_server.py"],
      "cwd": "/home/jack/MCP_PLATFORM_FastMCP"
    },
    "demo_prompt": {
      "_comment": "Prompt-only server - no tools or resources",
      "enabled": false,
      "command": "uv",
      "args": ["run", "python", "Servers/demo_prompt_server.py"],
      "cwd": "/home/jack/MCP_PLATFORM_FastMCP"
    }
  },
  "settings": {
    "defaultTimeout": 30,
    "maxRetries": 3,
    "retryDelay": 1.0,
    "logLevel": "INFO",
    "autoReconnect": true
  }
}
```

### Backend Setup
- [ ] Install dependencies: `uv install`
- [ ] Configure environment variables in `.env`
- [ ] Review and customize `src/config.yaml`
- [ ] Start the server: `uv run python src/application.py`

### Frontend Integration
- [ ] Implement WebSocket connection to configured port
- [ ] Add clear history button with confirmation
- [ ] Handle all message types in WebSocket handler
- [ ] Implement error handling for common scenarios
- [ ] Test streaming message display
- [ ] Add tool execution notifications

### Testing
- [ ] Verify configuration reloading works
- [ ] Test chat functionality with multiple turns
- [ ] Confirm tool execution works with your MCP servers
- [ ] Test WebSocket reconnection handling
- [ ] Validate error scenarios

### Production Deployment
- [ ] Set appropriate logging levels
- [ ] Configure retention policies
- [ ] Set up monitoring for WebSocket connections
- [ ] Monitor database growth
- [ ] Configure appropriate timeouts
- [ ] Set up backup procedures for chat history

---

This comprehensive documentation covers all aspects of the MCP Backend OpenRouter platform. For specific implementation details, refer to the source code and inline documentation.

## Quick Start Checklist
