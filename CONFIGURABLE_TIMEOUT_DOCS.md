# Configurable Timeout and Retry Logic

The MCP Backend now supports configurable timeout and retry logic for MCP client connections. This allows operators to tune connection behavior without code changes through YAML configuration.

## Configuration

Add the following section to your `src/config.yaml` file:

```yaml
# MCP Server Configuration
mcp:
  config_file: "servers_config.json"
  
  # Connection and retry configuration
  connection:
    # Maximum number of reconnection attempts per server
    max_reconnect_attempts: 5
    
    # Initial delay between reconnection attempts (seconds)
    # Uses exponential backoff up to max_reconnect_delay
    initial_reconnect_delay: 1.0
    
    # Maximum delay between reconnection attempts (seconds)
    max_reconnect_delay: 30.0
    
    # Connection timeout for initial server connection (seconds)
    connection_timeout: 30.0
    
    # Ping timeout for connection health checks (seconds)
    ping_timeout: 10.0
```

## Parameters

### max_reconnect_attempts
- **Type**: Integer (minimum: 1)
- **Default**: 5
- **Description**: Maximum number of reconnection attempts before giving up. Each server will retry connection this many times before marking as failed.

### initial_reconnect_delay
- **Type**: Float (positive number)
- **Default**: 1.0
- **Description**: Initial delay in seconds before the first reconnection attempt. This value doubles with each retry (exponential backoff).

### max_reconnect_delay
- **Type**: Float (must be >= initial_reconnect_delay)
- **Default**: 30.0
- **Description**: Maximum delay between reconnection attempts. Prevents excessively long waits due to exponential backoff.

### connection_timeout
- **Type**: Float (positive number)
- **Default**: 30.0
- **Description**: Timeout in seconds for the initial MCP server connection and handshake process.

### ping_timeout
- **Type**: Float (positive number)
- **Default**: 10.0
- **Description**: Timeout in seconds for connection health check pings.

## Behavior

### Exponential Backoff
The system implements exponential backoff for reconnection attempts:

1. First retry: `initial_reconnect_delay` seconds
2. Second retry: `initial_reconnect_delay * 2` seconds
3. Third retry: `initial_reconnect_delay * 4` seconds
4. And so on, until `max_reconnect_delay` is reached

After reaching the maximum delay, all subsequent retries use `max_reconnect_delay`.

### Example Timing
With default settings (initial_delay=1.0s, max_delay=30.0s, max_attempts=5):

- Attempt 1: Immediate
- Attempt 2: Wait 1.0s
- Attempt 3: Wait 2.0s
- Attempt 4: Wait 4.0s
- Attempt 5: Wait 8.0s

Total time before giving up: ~15 seconds

### High-Availability Scenarios
For production environments requiring faster recovery:

```yaml
mcp:
  connection:
    max_reconnect_attempts: 10
    initial_reconnect_delay: 0.5
    max_reconnect_delay: 5.0
    connection_timeout: 10.0
    ping_timeout: 3.0
```

### Development/Testing
For development where you want to fail fast:

```yaml
mcp:
  connection:
    max_reconnect_attempts: 2
    initial_reconnect_delay: 0.1
    max_reconnect_delay: 1.0
    connection_timeout: 5.0
    ping_timeout: 2.0
```

## Validation

The configuration system validates all parameters:

- `max_reconnect_attempts` must be at least 1
- All timeout values must be positive
- `max_reconnect_delay` must be greater than or equal to `initial_reconnect_delay`

Invalid configurations will raise a `ValueError` with a descriptive error message.

## Logging

The system logs connection configuration at startup:

```
INFO - MCP client 'demo' configured with: max_attempts=5, initial_delay=1.0s, max_delay=30.0s, connection_timeout=30.0s, ping_timeout=10.0s
```

Retry attempts are logged as warnings:

```
WARNING - Connection attempt 2 failed for demo_server: Connection refused. Retrying in 2.0s...
```

Final failures are logged as errors:

```
ERROR - Failed to connect to demo_server after 5 attempts: Connection refused
```

## Backward Compatibility

All parameters have sensible defaults, so existing configurations without the `mcp.connection` section will continue to work with the default values.
