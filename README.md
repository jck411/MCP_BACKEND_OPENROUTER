# MCP Platform

High-performance Model Context Protocol (MCP) client with WebSocket API, SQLite storage, and support for OpenRouter, OpenAI, and Groq.

## 🚀 Quick Start

```bash
# Install and run
uv sync
export OPENROUTER_API_KEY="your_key_here"  # or OPENAI_API_KEY, GROQ_API_KEY
./run.sh
```

**Connect:** `ws://localhost:8000/ws/chat`

## 📡 WebSocket API

**Send:**
```json
{
  "action": "chat",
  "request_id": "unique-id",
  "payload": {
    "text": "Hello",
    "streaming": true
  }
}
```

**Receive:**
```json
{
  "request_id": "unique-id",
  "status": "chunk",
  "chunk": {
    "type": "text",
    "data": "Response...",
    "metadata": {}
  }
}
```

## ⚙️ Configuration

### LLM Provider (`src/config.yaml`)
```yaml
llm:
  active: "openrouter"  # or "openai", "groq"
  providers:
    openrouter:
      model: "openai/gpt-4o-mini"
    openai:
      model: "gpt-4o-mini"
    groq:
      model: "llama-3.3-70b-versatile"
```

### MCP Servers (`src/servers_config.json`)
```json
{
  "mcpServers": {
    "my_server": {
      "enabled": true,
      "command": "python",
      "args": ["path/to/server.py"]
    }
  }
}
```

## 📁 Key Files
- `src/config.yaml` - LLM providers and settings
- `src/servers_config.json` - MCP server configurations
- `chat_history.db` - SQLite database for conversation history

## 🛠️ Development

### Code Formatting
```bash
# Quick format (ignores line length issues)
./format.sh

# Full check including line length
uv run ruff check src/

# Format specific files
uv run ruff format src/mcp/client.py src/llm/client.py
```

### Project Structure
```
src/
├── main.py              # Entry point
├── application.py       # App startup
├── mcp/                 # MCP client module
│   └── client.py        # MCPClient class
├── llm/                 # LLM client module
│   └── client.py        # LLMClient class
└── chat/                # Chat orchestration
```

## ✅ Features
- **Full MCP Protocol** - Tools, prompts, resources
- **High Performance** - SQLite with WAL mode, optimized indexes
- **Real-time Streaming** - WebSocket with delta persistence
- **Multi-Provider** - OpenRouter (100+ models), OpenAI, Groq
- **Type Safe** - Pydantic validation throughout

---
**Requirements:** Python 3.13+, `request_id` required in WebSocket messages
