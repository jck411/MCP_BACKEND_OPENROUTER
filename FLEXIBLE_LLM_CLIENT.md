# Flexible LLM Client Documentation

## Overview

The updated `LLMClient` in `src/main.py` is designed to be "ready for anything" - it automatically passes through all configuration parameters to the API, making it compatible with new models and their specific parameters without requiring code changes.

## Key Features

### 🔄 **Automatic Parameter Pass-Through**
- Any parameter you add to the config is automatically sent to the API
- No code changes needed for new model parameters
- Works with any OpenAI-compatible API

### 🧠 **Reasoning Model Support**
- Automatically detects and extracts thinking/reasoning content
- Supports various field names: `thinking`, `reasoning`, `thought_process`, etc.
- Works in both streaming and non-streaming modes
- Compatible with o1, o3, and future reasoning models

### 🎯 **Provider Flexibility**
- Works with OpenAI, OpenRouter, Groq, and any OpenAI-compatible provider
- Provider detection for potential future optimizations
- Graceful handling of provider-specific quirks

## Configuration Examples

### Standard Model
```yaml
openai:
  base_url: "https://api.openai.com/v1"
  model: "gpt-4o-mini"
  temperature: 0.7
  max_tokens: 4096
  top_p: 1.0
```

### Reasoning Model (o1)
```yaml
openai_reasoning:
  base_url: "https://api.openai.com/v1"
  model: "o1-preview"
  max_completion_tokens: 8192  # o1 specific parameter
  # No temperature - o1 uses fixed temperature
```

### Future Model with Custom Parameters
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

## How It Works

### Parameter Pass-Through
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

### Reasoning Content Extraction
```python
# Checks multiple possible locations for thinking content
possible_reasoning_fields = [
    "thinking", "reasoning", "thought_process", 
    "internal_thoughts", "chain_of_thought", "rationale"
]
```

### Usage in Your Code
The chat service automatically uses the flexible client - no changes needed to your existing code. The client:

1. **Builds the request** with all your config parameters
2. **Sends to the API** (any OpenAI-compatible endpoint)  
3. **Extracts thinking content** if present
4. **Returns standardized response** with optional `thinking` field

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

## Adding New Models/Parameters

### Step 1: Add to Config
```yaml
llm:
  active: "my_new_provider"
  providers:
    my_new_provider:
      base_url: "https://api.newprovider.com/v1"
      model: "new-amazing-model"
      # Add ANY parameters the model supports
      new_param1: "value1"
      experimental_feature: true
      custom_settings:
        nested_param: "nested_value"
```

### Step 2: That's It!
The client automatically:
- ✅ Passes all parameters to the API
- ✅ Handles any response format
- ✅ Extracts reasoning content if present
- ✅ Maintains compatibility with your existing code

## Streaming Support

Reasoning models are fully supported in streaming mode:
- Thinking content is buffered and yielded separately
- Regular content streams normally
- End-of-thinking is automatically detected
- Compatible with your existing streaming handlers

## Benefits

1. **Zero Code Changes** for new models/parameters
2. **Future-Proof** - works with models that don't exist yet
3. **Reasoning Compatible** - handles thinking modes transparently  
4. **Provider Agnostic** - works with any OpenAI-compatible API
5. **Backward Compatible** - existing configurations continue to work

Your LLMClient is now ready for anything! 🚀
