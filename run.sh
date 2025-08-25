#!/bin/bash
# Enhanced script to run the MCP client with environment verification
set -e

cd "$(dirname "$0")"

# Verify uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ UV not found. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Verify virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Running setup..."
    ./setup.sh
fi

# Verify critical dependencies
echo "🔍 Verifying environment..."
if ! uv run python -c "import fastmcp; import mcp" &> /dev/null; then
    echo "❌ Missing dependencies. Syncing..."
    uv sync --extra dev
fi

echo "🚀 Starting MCP Platform..."
uv run python src/main.py
