#!/usr/bin/env python3
"""
Test script to demonstrate Python logging hierarchy behavior
"""
import logging
import sys
sys.path.insert(0, 'src')

def test_logging_hierarchy():
    """Test to show how logging levels work in Python's hierarchy"""
    print("=== Python Logging Hierarchy Test ===\n")

    # Clear any existing handlers
    for logger in [logging.getLogger(), logging.getLogger('src.chat'), logging.getLogger('src.clients')]:
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)

    # Set up console handler on root logger
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    # Test 1: Root at INFO, child at WARNING
    print("Test 1: Root logger = INFO, src.chat = WARNING")
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger('src.chat').setLevel(logging.WARNING)

    print(f"Root logger effective level: {logging.getLogger().getEffectiveLevel()} (INFO=20)")
    print(f"src.chat logger level: {logging.getLogger('src.chat').getEffectiveLevel()} (WARNING=30)")

    print("\nLogging from src.chat (should show WARNING only):")
    logging.getLogger('src.chat').info("This INFO message should NOT appear")
    logging.getLogger('src.chat').warning("This WARNING message SHOULD appear")

    print("\n" + "="*50 + "\n")

    # Test 2: Root at INFO, child at INFO
    print("Test 2: Root logger = INFO, src.chat = INFO")
    logging.getLogger('src.chat').setLevel(logging.INFO)

    print(f"Root logger effective level: {logging.getLogger().getEffectiveLevel()} (INFO=20)")
    print(f"src.chat logger level: {logging.getLogger('src.chat').getEffectiveLevel()} (INFO=20)")

    print("\nLogging from src.chat (should show INFO and above):")
    logging.getLogger('src.chat').info("This INFO message SHOULD appear")
    logging.getLogger('src.chat').warning("This WARNING message SHOULD appear")

    print("\n" + "="*50 + "\n")

    # Test 3: Root at WARNING, child at INFO (this is what you might expect to be overridden)
    print("Test 3: Root logger = WARNING, src.chat = INFO")
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger('src.chat').setLevel(logging.INFO)

    print(f"Root logger effective level: {logging.getLogger().getEffectiveLevel()} (WARNING=30)")
    print(f"src.chat logger level: {logging.getLogger('src.chat').getEffectiveLevel()} (INFO=20)")

    print("\nLogging from src.chat (should show INFO and above despite root being WARNING):")
    logging.getLogger('src.chat').info("This INFO message SHOULD appear")
    logging.getLogger('src.chat').warning("This WARNING message SHOULD appear")

    print("\n" + "="*50 + "\n")

    print("Key Points:")
    print("1. Child loggers with explicit levels are NOT overridden by parent levels")
    print("2. Only when a logger has level NOTSET does it inherit from its parent")
    print("3. More restrictive parent levels do NOT override less restrictive child levels")
    print("4. Less restrictive parent levels do NOT override more restrictive child levels")

if __name__ == "__main__":
    test_logging_hierarchy()
