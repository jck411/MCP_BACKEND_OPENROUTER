#!/bin/bash
# MCP Backend Test Runner
# Run this script to verify the system is working correctly

echo "🔬 Running MCP Backend System Tests..."
echo "======================================"

cd "$(dirname "$0")"

# Use the virtual environment if it exists
if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
elif command -v python3 > /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

# Run the test
$PYTHON_CMD test_system.py

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All tests passed! System is ready."
else
    echo ""
    echo "❌ Some tests failed. Check the output above."
    exit 1
fi
