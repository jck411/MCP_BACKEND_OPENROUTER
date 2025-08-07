# Modular Chat Architecture

This document describes the new modular chat architecture that replaced the monolithic `chat_service.py` file.

## Architecture Overview

The chat system has been refactored into focused modules with clear separation of concerns:

### Core Principle: Complexity Isolation

- **Complex/Fragile modules** are isolated for better error handling and logging
- **Mundane/Stable modules** handle straightforward business logic
- **Code duplication is acceptable** between streaming/non-streaming for clarity

## Module Structure

```
src/chat/
├── __init__.py                 # Exports: ChatOrchestrator, ChatMessage, etc.
├── models.py                   # Stable data structures
├── conversation_manager.py     # Stable conversation operations
├── tool_executor.py           # Complex/fragile tool execution
├── streaming_handler.py       # Complex/fragile streaming logic
├── simple_chat_handler.py     # Mundane non-streaming operations
├── resource_loader.py         # Fragile resource loading
└── chat_orchestrator.py       # Main coordinator
```

## Module Responsibilities

### 1. **models.py** (Stable)
- **Purpose**: Core data structures that rarely change
- **Contents**:
  - `ChatMessage` - Message structure with metadata
  - `ToolCallContext` - Parameters for tool call iterations
  - `convert_usage()` - Usage statistics conversion
- **Stability**: High - pure data models

### 2. **conversation_manager.py** (Mundane/Stable)
- **Purpose**: Straightforward conversation operations
- **Contents**:
  - User message persistence with idempotency checks
  - Conversation history building
  - Basic repository interactions
- **Logging**: `→ Repository:` / `← Repository:` for database operations
- **Stability**: High - simple business logic

### 3. **tool_executor.py** (Complex/Fragile)
- **Purpose**: Tool execution and MCP client interactions
- **Contents**:
  - Tool call accumulation from streaming deltas
  - Tool execution with error handling
  - Tool result formatting
  - Tool hop limit checking
- **Logging**: 
  - `→ MCP[server_name]: calling tool_name`
  - `← MCP[server_name]: success/failure`
- **Fragility**: HIGH - prone to breaking when MCP servers change

### 4. **streaming_handler.py** (Complex/Fragile)
- **Purpose**: Most complex streaming response handling
- **Contents**:
  - LLM response streaming with delta persistence
  - Tool call iteration handling
  - Hybrid message yielding (ChatMessage + dict)
  - Frontend streaming coordination
- **Logging**:
  - `→ Frontend: streaming content delta`
  - `→ Frontend: tool execution notification`  
  - `→ LLM: starting streaming request`
- **Fragility**: HIGHEST - streaming bugs are hard to debug

### 5. **simple_chat_handler.py** (Mundane/Stable)
- **Purpose**: Non-streaming chat operations
- **Contents**:
  - Simple request/response flow
  - Basic tool call execution (no delta complexity)
  - Straightforward LLM interactions
- **Code Duplication**: Acceptable with streaming handler for clarity
- **Stability**: High - simple patterns

### 6. **resource_loader.py** (Fragile)
- **Purpose**: Resource loading and system prompt construction
- **Contents**:
  - Resource availability checking
  - Resource catalog management
  - System prompt building with resources
  - Prompt application
- **Logging**:
  - `→ Resources: checking availability`
  - `→ Resources: resource_uri is available/unavailable`
- **Fragility**: HIGH - fails often when MCP servers are down

### 7. **chat_orchestrator.py** (Main Coordinator)
- **Purpose**: Thin coordination layer
- **Contents**:
  - Component initialization
  - Request routing to appropriate handlers
  - High-level flow control
- **Logging**: `→ Orchestrator:` / `← Orchestrator:` for coordination
- **Design**: Keeps main class simple, just delegates

## Import Structure

### Current Usage (Direct Import)
```python
# main.py and websocket_server.py
from src.chat import ChatOrchestrator

# Create orchestrator directly
orchestrator = ChatOrchestrator(config)
```

### Module Exports
```python
# src/chat/__init__.py
from .chat_orchestrator import ChatOrchestrator
from .models import ChatMessage, ToolCallContext, convert_usage

__all__ = ["ChatOrchestrator", "ChatMessage", "ToolCallContext", "convert_usage"]
```

## Logging Strategy

Each module has its own logger with directional flow indicators:

- **`→ Frontend:`** Messages sent to user interface
- **`→ MCP[server]:`** Calls to MCP servers
- **`← MCP[server]:`** Responses from MCP servers  
- **`→ Repository:`** Database operations
- **`→ Resources:`** Resource loading operations
- **`→ LLM:`** Language model API calls
- **`← LLM:`** Language model responses
- **`→ Orchestrator:`** High-level coordination

Example:
```python
logger.info("→ MCP[git]: calling git_status")
logger.info("← MCP[git]: success, 3 files changed")
logger.info("→ Frontend: streaming content delta, length=25")
```

## Benefits Achieved

### ✅ Complexity Isolation
- Streaming complexity completely separated from simple chat
- Tool execution isolated for better MCP error handling
- Resource loading isolated for availability failures

### ✅ Clear Failure Boundaries  
- When streaming breaks, simple chat still works
- When one MCP server fails, others continue working
- When resources fail to load, chat continues without them

### ✅ Maintainability
- Each module has single responsibility
- Clear logging boundaries for debugging
- Easy to test components independently

### ✅ No Massive Files
- Largest module is ~300 lines vs original 1000+ lines
- Each file focused on specific concern
- Easier to navigate and understand

## Migration Notes

### What Changed
- ❌ **Removed**: `src/chat_service.py` (deleted entirely)
- ✅ **Added**: Modular structure under `src/chat/`
- ✅ **Updated**: Direct imports to `ChatOrchestrator`

### Breaking Changes
- **Import change**: `from src.chat_service import ChatService` → `from src.chat import ChatOrchestrator`
- **Class name**: `ChatService` → `ChatOrchestrator`
- **Config class**: `ChatServiceConfig` → `ChatOrchestratorConfig`

### Interface Compatibility
All methods remain the same:
- `initialize()` 
- `process_message()` - streaming
- `chat_once()` - non-streaming
- `apply_prompt()`
- `cleanup()`

## Testing the Architecture

```python
# Verify imports work
from src.chat import ChatOrchestrator, ChatMessage, ToolCallContext, convert_usage

# Verify interface
orchestrator = ChatOrchestrator(config)
await orchestrator.initialize()
```

The modular architecture maintains full functionality while providing clear separation of concerns and better maintainability.
