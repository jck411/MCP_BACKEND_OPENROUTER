# MAX_TOOL_HOPS Configuration

## Overview

The `max_tool_hops` configuration parameter controls the maximum number of recursive tool calls allowed in a single conversation turn. This limit prevents infinite loops when tools call other tools in a chain.

## Configuration

Add or modify the `max_tool_hops` parameter in your `config.yaml` file:

```yaml
chat:
  service:
    # Maximum number of recursive tool calls to prevent infinite loops
    max_tool_hops: 8  # Default value
```

## Default Value

- **Default**: 8 recursive tool calls
- **Minimum**: 1 (must be a positive integer)
- **No maximum limit** (though very high values are not recommended)

## Use Cases

### Higher Values (12-20)
Consider increasing the limit for complex workflows that require:
- Multi-step data processing pipelines
- Complex analysis requiring multiple tool interactions
- Workflows with legitimate tool chaining needs

### Lower Values (3-5)
Consider decreasing the limit for:
- Simple use cases with minimal tool interaction
- Environments where you want to prevent complex tool chains
- Testing scenarios where you want stricter limits

## Configuration Example

```yaml
chat:
  service:
    system_prompt: |
      You are a helpful assistant with access to tools.
    
    streaming:
      enabled: true
    
    # Allow up to 15 tool hops for complex workflows
    max_tool_hops: 15
    
    tool_notifications:
      enabled: true
      show_args: true
```

## Behavior When Limit is Reached

When the maximum number of tool hops is reached:

1. The system logs a warning message
2. Tool execution stops immediately
3. A user-visible warning is displayed:
   ```
   ⚠️ Reached maximum tool call limit (8). Stopping to prevent infinite recursion.
   ```
4. The conversation continues with the partial results obtained

## Validation

The system validates the `max_tool_hops` value:
- Must be a positive integer (≥ 1)
- Invalid values will raise a `ValueError` during startup

## Migration Notes

This change is backward compatible:
- If `max_tool_hops` is not specified, the default value of 8 is used
- Existing configurations will continue to work without modification
- The previous hard-coded limit of 8 is now the configurable default
