# Dynamic Runtime Configuration System

This document describes the dynamic runtime configuration system that allows configuration changes during runtime without restarting the system.

## Overview

The system uses a **persistent runtime configuration architecture** with two files:
- `src/config.yaml` - **Reference/Default Configuration**: Contains default values (never modified by the system)
- `src/runtime_config.yaml` - **Persistent Runtime Configuration**: The primary configuration file that persists changes across restarts

## Key Features

### 🔄 Dynamic Updates
Configuration changes are applied immediately without requiring a system restart.

### 💾 Persistent Changes
All runtime configuration changes persist across system restarts - no reset to defaults.

### 🔍 Automatic Detection
The system automatically detects when the runtime configuration file is modified and reloads it.

### 🔗 Deep Merging
Partial configuration updates are supported - you only need to specify the values you want to change.

### 🛡️ Automatic Recovery
If the runtime configuration is missing or corrupted, the system automatically recreates it from defaults.

### 📊 Metadata Tracking
Tracks version numbers, modification timestamps, and configuration history.

### ⚡ Safe Operations
Robust error handling for corrupted YAML files and invalid configurations.

## How It Works

1. **Initialization**: System loads `config.yaml` as reference defaults and creates `runtime_config.yaml` if it doesn't exist
2. **Runtime Check**: On each configuration access, checks if `runtime_config.yaml` has been modified
3. **Primary Source**: `runtime_config.yaml` serves as the primary configuration source
4. **Persistence**: All changes are saved to `runtime_config.yaml` and persist across restarts
5. **Auto-reload**: Automatically detects file changes and reloads configuration

## File Structure

### Default Configuration (`src/config.yaml`)
```yaml
# MCP Chatbot Configuration
chat:
  websocket:
    host: "localhost"
    port: 8000
  service:
    max_tool_hops: 8
llm:
  active: "openrouter"
  providers:
    openrouter:
      model: "openai/gpt-4o-mini"
      temperature: 0.7
```

### Runtime Configuration (`src/runtime_config.yaml`)
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
  last_modified: 1704661200.0
  version: 5
  is_runtime_config: true
  default_config_path: "config.yaml"
```

## Usage Examples

### Programmatic Configuration Updates

```python
from src.config import Configuration

# Initialize configuration
config = Configuration()

# Get current configuration
current_config = config.get_config_dict()

# Modify specific values
current_config['chat']['websocket']['port'] = 9000
current_config['llm']['active'] = 'groq'

# Save changes (automatically reloads)
config.save_runtime_config(current_config)

# Verify changes
print(f"New port: {config.get_websocket_config()['port']}")
```

### Manual File Editing

You can directly edit `src/runtime_config.yaml`:

```yaml
# Only specify what you want to override
chat:
  service:
    system_prompt: |
      You are a specialized assistant with enhanced capabilities.
      You can adapt your behavior based on runtime configuration.

llm:
  providers:
    openrouter:
      temperature: 0.9  # Make responses more creative
      max_tokens: 8192  # Increase token limit
```

The changes will be automatically detected and applied.

### Partial Updates

```python
# Update only specific nested values
partial_config = {
    'chat': {
        'service': {
            'max_tool_hops': 20
        }
    }
}

# Merge with existing config
merged_config = config._deep_merge(config.get_config_dict(), partial_config)
config.save_runtime_config(merged_config)
```

## API Reference

### Configuration Class Methods

#### `reload_runtime_config() -> bool`
Manually reload runtime configuration.
- **Returns**: `True` if configuration was reloaded, `False` if no changes detected.

#### `save_runtime_config(config: dict) -> None`
Save configuration to runtime config file.
- **Args**: `config` - Configuration dictionary to save
- **Effect**: Automatically adds metadata and reloads configuration

#### `get_config_dict() -> dict`
Get the complete merged configuration dictionary.
- **Returns**: Current configuration with runtime overrides applied

### Automatic Reload
All existing configuration methods automatically check for runtime updates:
- `get_llm_config()`
- `get_websocket_config()`
- `get_chat_service_config()`
- `get_chat_storage_config()`
- `get_logging_config()`
- `get_mcp_connection_config()`

## Runtime Configuration Metadata

The system automatically manages metadata in the `_runtime_config` section:

```yaml
_runtime_config:
  last_modified: 1704661200.0    # Unix timestamp of last modification
  version: 5                     # Incremental version number
  is_runtime_config: true        # Flag indicating this is a runtime config
  default_config_path: "config.yaml"  # Path to default configuration
```

## Error Handling

### Corrupted Runtime Config
If `runtime_config.yaml` becomes corrupted or unreadable:
- System automatically falls back to default configuration
- Logs the error for debugging
- Continues operation without interruption

### Missing Runtime Config
If `runtime_config.yaml` doesn't exist:
- System uses default configuration from `config.yaml`
- No errors or warnings generated
- Normal operation continues

### Invalid Configuration Values
Configuration validation occurs at the point of use:
- Invalid values trigger appropriate error messages
- System maintains stability through validation
- Default values are used when possible

## Best Practices

### 1. Incremental Updates
Only specify the configuration values you want to change in the runtime config.

```yaml
# Good: Only override what's needed
chat:
  service:
    max_tool_hops: 15

# Avoid: Copying entire default config
```

### 2. Backup Important Changes
Before making significant runtime configuration changes:
```bash
cp src/runtime_config.yaml src/runtime_config.yaml.backup
```

### 3. Validate Changes
Test configuration changes in a development environment first:
```python
config = Configuration()
print("Current config working:", config.get_llm_config())
```

### 4. Monitor Configuration
Use the metadata to track configuration changes:
```python
if config._runtime_config:
    meta = config._runtime_config.get('_runtime_config', {})
    print(f"Config version: {meta.get('version')}")
    print(f"Last modified: {meta.get('last_modified')}")
```

## Demo Script

Run the demonstration script to see the system in action:

```bash
python example_runtime_config_usage.py
```

This script demonstrates:
- Loading initial configuration
- Making runtime changes
- Automatic reload detection
- Partial configuration updates
- Metadata tracking

## Troubleshooting

### Configuration Not Updating
1. Check file permissions on `src/runtime_config.yaml`
2. Verify YAML syntax using a validator
3. Check system logs for error messages
4. Try manual reload: `config.reload_runtime_config()`

### Performance Considerations
- File modification time checking has minimal overhead
- Configuration is cached between modification time checks
- Deep merging is efficient for typical configuration sizes
- Consider the frequency of configuration access in high-throughput scenarios

### Concurrent Access
- File-based configuration is safe for single-process applications
- For multi-process scenarios, consider implementing file locking
- Database-backed configuration may be preferred for complex distributed systems

## Integration Notes

The dynamic runtime configuration system is fully backward compatible:
- Existing code continues to work without modification
- All existing configuration methods are enhanced automatically
- No breaking changes to the Configuration class API
- Default behavior remains unchanged when no runtime config exists

This system provides a robust foundation for runtime configuration management while maintaining simplicity and reliability.
