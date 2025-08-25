#!/usr/bin/env python3
"""
Performance Test Script

This script demonstrates the performance improvements from our optimizations.
Run this to see the actual impact of the changes.
"""

import asyncio
import time
from src.clients.llm_client import LLMClient
from src.config import Configuration

async def test_connection_performance():
    """Test the performance of connection creation and reuse."""
    print("🚀 Testing Connection Performance")
    print("=" * 50)

    # Test 1: Configuration Loading Speed
    start_time = time.monotonic()
    config = Configuration()
    config_time = time.monotonic() - start_time
    print(f"⚙️  Configuration loaded in: {config_time:.3f}s")
    # Test 2: Connection Pool Initialization
    start_time = time.monotonic()
    async with LLMClient(config) as llm_client:
        init_time = time.monotonic() - start_time
        print(f"🔌 Connection pool initialized in: {init_time:.3f}s")
        # Test 3: Connection Pool Settings
        pool_config = llm_client._connection_pool_config
        print(f"📊 Connection Pool Config:")
        print(f"   Max Connections: {pool_config['max_connections']}")
        print(f"   Keepalive Connections: {pool_config['max_keepalive_connections']}")
        print(f"   Request Timeout: {pool_config['request_timeout_seconds']}s")

        # Test 4: Configuration Caching
        start_time = time.monotonic()
        for _ in range(100):
            _ = llm_client._connection_pool_config
        cache_time = time.monotonic() - start_time
        print(f"⚡ 100 config cache accesses in: {cache_time:.6f}s")
    print("\n✅ Performance test completed!")

if __name__ == "__main__":
    asyncio.run(test_connection_performance())
