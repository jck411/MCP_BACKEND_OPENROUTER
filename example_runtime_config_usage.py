#!/usr/bin/env python3
"""Example script demonstrating dynamic runtime configuration usage."""

import time
import yaml
from src.config import Configuration


def main():
    """Demonstrate the dynamic runtime configuration system."""
    print("=== Dynamic Runtime Configuration Demo ===\n")
    
    # Initialize configuration
    config = Configuration()
    
    print("1. Initial configuration loaded:")
    print(f"   Active LLM provider: {config.get_llm_config().get('model', 'unknown')}")
    print(f"   WebSocket port: {config.get_websocket_config().get('port', 'unknown')}")
    print(f"   Max tool hops: {config.get_max_tool_hops()}")
    print()
    
    # Show current runtime config status
    runtime_config_path = config._runtime_config_path
    runtime_metadata = config.get_runtime_metadata()
    print(f"2. Runtime config path: {runtime_config_path}")
    print(f"   Runtime config exists: {'Yes' if runtime_metadata else 'No'}")
    print()
    
    # Example: Modify configuration during runtime
    print("3. Modifying configuration during runtime...")
    
    # Get current config and modify it
    current_config = config.get_config_dict().copy()
    
    # Change some settings
    current_config['chat']['websocket']['port'] = 8080
    current_config['chat']['service']['max_tool_hops'] = 12
    current_config['llm']['active'] = 'groq'
    
    # Save the modified configuration
    config.save_runtime_config(current_config)
    print("   Configuration saved to runtime_config.yaml")
    print()
    
    # Verify changes are applied
    print("4. Verifying changes were applied:")
    print(f"   Active LLM provider: {config.get_llm_config().get('model', 'unknown')}")
    print(f"   WebSocket port: {config.get_websocket_config().get('port', 'unknown')}")
    print(f"   Max tool hops: {config.get_max_tool_hops()}")
    print()
    
    # Show runtime config metadata
    runtime_meta = config.get_runtime_metadata()
    if runtime_meta:
        print("5. Runtime configuration metadata:")
        print(f"   Version: {runtime_meta.get('version', 'unknown')}")
        print(f"   Last modified: {runtime_meta.get('last_modified', 'unknown')}")
        print(f"   Is runtime config: {runtime_meta.get('is_runtime_config', 'unknown')}")
        print()
    
    # Demonstrate automatic reload detection
    print("6. Demonstrating automatic reload detection...")
    print("   (You could modify runtime_config.yaml manually now)")
    print("   Checking for changes...")
    
    # Check if config changed
    config_changed = config.reload_runtime_config()
    print(f"   Configuration changed: {config_changed}")
    print()
    
    print("7. Example: Partial configuration update")
    # Show how partial updates work (only changing one setting)
    partial_config = {
        'chat': {
            'service': {
                'max_tool_hops': 15
            }
        }
    }
    
    # This will be merged with existing runtime config
    merged_config = config._deep_merge(config.get_config_dict(), partial_config)
    config.save_runtime_config(merged_config)
    
    print(f"   Updated max_tool_hops to: {config.get_max_tool_hops()}")
    print(f"   WebSocket port unchanged: {config.get_websocket_config().get('port', 'unknown')}")
    print()
    
    print("=== Demo Complete ===")
    print("\nKey Features:")
    print("- Runtime configuration changes without restart")
    print("- Automatic detection of file modifications")
    print("- Deep merging of partial configuration updates")
    print("- Fallback to default config for missing values")
    print("- Metadata tracking for versioning and timestamps")


if __name__ == "__main__":
    main()
