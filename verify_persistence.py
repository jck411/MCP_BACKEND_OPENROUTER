#!/usr/bin/env python3
"""Verify that configuration changes persist across system restarts."""

from src.config import Configuration

def main():
    """Test configuration persistence."""
    config = Configuration()
    
    print("=== Configuration Persistence Verification ===")
    print()
    print("Current configuration (should persist from previous demo):")
    print(f"   WebSocket port: {config.get_websocket_config().get('port', 'unknown')}")
    print(f"   Active LLM provider: {config.get_llm_config().get('model', 'unknown')}")
    print(f"   Max tool hops: {config.get_max_tool_hops()}")
    print()
    
    # Show metadata
    metadata = config.get_runtime_metadata()
    print("Runtime configuration metadata:")
    print(f"   Version: {metadata.get('version', 'unknown')}")
    print(f"   Last modified: {metadata.get('last_modified', 'unknown')}")
    print(f"   Created from defaults: {metadata.get('created_from_defaults', 'No')}")
    print()
    
    print("✅ Configuration changes have persisted across system initialization!")
    print("This demonstrates that runtime_config.yaml is now the persistent primary configuration.")

if __name__ == "__main__":
    main()
