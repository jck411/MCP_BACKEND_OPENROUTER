#!/usr/bin/env python3
"""
Test script to verify that MAX_TOOL_HOPS configuration works correctly.
"""

import asyncio
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import Configuration


def test_max_tool_hops_config():
    """Test that MAX_TOOL_HOPS can be configured."""
    print("Testing MAX_TOOL_HOPS configuration...")
    
    # Test with default config
    config = Configuration()
    default_hops = config.get_max_tool_hops()
    print(f"Default max_tool_hops: {default_hops}")
    assert default_hops == 8, f"Expected 8, got {default_hops}"
    
    # Test with custom value by creating a temporary config
    import tempfile
    import yaml
    
    custom_config = {
        "chat": {
            "service": {
                "max_tool_hops": 15
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(custom_config, f)
        temp_config_path = f.name
    
    try:
        # Temporarily replace the config file path
        original_config_path = os.path.join(os.path.dirname(__file__), "src", "config.yaml")
        backup_path = original_config_path + ".backup"
        
        # Backup original config
        os.rename(original_config_path, backup_path)
        
        # Copy temp config to expected location
        import shutil
        shutil.copy2(temp_config_path, original_config_path)
        
        # Create new configuration instance
        config = Configuration()
        custom_hops = config.get_max_tool_hops()
        print(f"Custom max_tool_hops: {custom_hops}")
        assert custom_hops == 15, f"Expected 15, got {custom_hops}"
        
    finally:
        # Restore original config
        if os.path.exists(backup_path):
            os.rename(backup_path, original_config_path)
        os.unlink(temp_config_path)
    
    print("✅ MAX_TOOL_HOPS configuration test passed!")


def test_validation():
    """Test that invalid values are rejected."""
    print("Testing validation...")
    
    import tempfile
    import yaml
    
    # Test invalid value (0)
    invalid_config = {
        "chat": {
            "service": {
                "max_tool_hops": 0
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(invalid_config, f)
        temp_config_path = f.name
    
    try:
        original_config_path = os.path.join(os.path.dirname(__file__), "src", "config.yaml")
        backup_path = original_config_path + ".backup"
        
        # Backup original config
        os.rename(original_config_path, backup_path)
        
        # Copy temp config to expected location
        import shutil
        shutil.copy2(temp_config_path, original_config_path)
        
        # Create new configuration instance
        config = Configuration()
        try:
            config.get_max_tool_hops()
            assert False, "Should have raised ValueError for invalid value"
        except ValueError as e:
            print(f"✅ Correctly rejected invalid value: {e}")
        
    finally:
        # Restore original config
        if os.path.exists(backup_path):
            os.rename(backup_path, original_config_path)
        os.unlink(temp_config_path)
    
    print("✅ Validation test passed!")


if __name__ == "__main__":
    test_max_tool_hops_config()
    test_validation()
    print("\n🎉 All tests passed! MAX_TOOL_HOPS is now configurable.")
