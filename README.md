# MCP Backend OpenRouter

A high-performance chatbot platform connecting MCP servers with LLM APIs for intelligent tool execution.

## 🚀 Quick Start

```bash
# Install dependencies
uv install

# Start the platform
uv run python src/main.py

# Reset configuration to defaults
uv run mcp-reset-config
```

**Connect:** `ws://localhost:8000/ws/chat`

## 📡 WebSocket API

### Send Messages
```json
{
  "type": "user_message",
  "message": "Hello, how can you help me today?"
}
```

### Receive Responses
```json
{
  "type": "assistant_message",
  "message": "I'm here to help! What would you like to know?",
  "thinking": "The user is greeting me...",
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 12,
    "total_tokens": 27
  }
}
```

### Message Types

| Type | Purpose | Payload |
|------|---------|---------|
| `user_message` | Send user input | `{"type": "user_message", "message": "text"}` |
| `clear_history` | Start new session | `{"type": "clear_history"}` |
| `assistant_message` | AI response | `{"type": "assistant_message", "message": "text", "thinking": "reasoning"}` |
| `tool_execution` | Tool status | `{"type": "tool_execution", "tool_name": "name", "status": "executing"}` |

## ⚙️ Configuration

### Essential Settings (`src/runtime_config.yaml`)
```yaml
chat:
  websocket:
    port: 8000  # WebSocket server port
    
  service:
    max_tool_hops: 8  # Maximum tool call iterations
    streaming:
      enabled: true  # Enable streaming responses
      
  storage:
    persistence:
      db_path: "chat_history.db"
      retention:
        max_age_hours: 24
        max_messages: 1000

llm:
  active: "openrouter"  # Active LLM provider
  providers:
    openrouter:
      base_url: "https://openrouter.ai/api/v1"
      model: "openai/gpt-4o-mini"
      temperature: 0.7
      max_tokens: 4096
```

### MCP Servers (`servers_config.json`)
```json
{
  "mcpServers": {
    "demo": {
      "enabled": true,
      "command": "uv",
      "args": ["run", "python", "Servers/config_server.py"],
      "cwd": "/path/to/your/project"
    }
  }
}
```

## 🔧 Performance Tuning

### Streaming Optimization
```yaml
chat:
  service:
    streaming:
      persistence:
        persist_deltas: false      # Maximum speed (no DB writes during streaming)
        interval_ms: 200           # Flush every 200ms
        min_chars: 1024           # Or when buffer reaches 1024 chars
```

### HTTP/2 Support
```bash
uv add h2  # Required for HTTP/2 optimization
```

## 🛠️ Development

### Code Standards
- Use `uv` for package management
- Pydantic for data validation
- Type hints required
- Fail-fast error handling

### Available Scripts
```bash
uv run python src/main.py          # Start platform
uv run python scripts/format.py    # Format code
uv run mcp-reset-config            # Reset configuration
```

### Code Formatting
```bash
# Quick format (ignores line length issues)
./format.sh

# Full check including line length
uv run ruff check src/

# Format specific files
uv run ruff format src/chat/ src/clients/
```

## 📁 Project Structure

```
MCP_BACKEND_OPENROUTER/
├── src/                    # Main source code
│   ├── main.py            # Application entry point
│   ├── config.py          # Configuration management
│   ├── websocket_server.py # WebSocket communication
│   ├── chat/              # Chat system modules
│   ├── clients/           # LLM and MCP clients
│   └── history/           # Storage and persistence
├── Servers/               # MCP server implementations
├── config.yaml            # Default configuration
├── runtime_config.yaml    # Runtime overrides
├── servers_config.json    # MCP server config
└── uv.lock               # Dependency lock file
```

## 🔑 Environment Variables

```bash
# Required for LLM APIs
export OPENAI_API_KEY="your-key"
export OPENROUTER_API_KEY="your-key"
export GROQ_API_KEY="your-key"
```

## 🚨 Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| Configuration not updating | Check file permissions on `runtime_config.yaml` |
| WebSocket connection fails | Verify server is running and port is correct |
| MCP server errors | Check `servers_config.json` and server availability |
| LLM API issues | Verify API keys and model configuration |

### Debug Mode
```yaml
# In runtime_config.yaml
logging:
  level: "DEBUG"
```

### Component Testing
```python
# Test configuration
from src.config import Configuration
config = Configuration()
print(config.get_config_dict())

# Test LLM client
from src.clients.llm_client import LLMClient
llm = LLMClient(config.get_llm_config())
```

## ✅ Features

- **Full MCP Protocol** - Tools, prompts, resources
- **High Performance** - SQLite with WAL mode, optimized indexes
- **Real-time Streaming** - WebSocket with delta persistence
- **Multi-Provider** - OpenRouter (100+ models), OpenAI, Groq
- **Type Safe** - Pydantic validation throughout
- **Dynamic Configuration** - Runtime changes without restart
- **Auto-Persistence** - Automatic conversation storage

## 📚 Quick Reference

| Command | Purpose |
|---------|---------|
| `uv run python src/main.py` | Start the platform |
| `uv run mcp-reset-config` | Reset to default config |
| Edit `runtime_config.yaml` | Change settings (auto-reload) |
| Edit `servers_config.json` | Configure MCP servers |

## 🆘 Support

- Check logs for detailed error messages
- Verify configuration syntax with YAML validator
- Test individual components for isolation
- Monitor WebSocket connections and database size

---

**Requirements:** Python 3.13+, `uv` package manager
