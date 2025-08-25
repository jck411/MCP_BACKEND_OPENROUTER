#!/bin/bash
# Setup script for MCP Platform

set -e

echo "🚀 Setting up MCP Platform..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ UV is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Please create one with your API keys:"
    echo "   echo 'GROQ_API_KEY=your_groq_key_here' > .env"
    echo "   # OR use OPENAI_API_KEY, ANTHROPIC_API_KEY, etc."
    echo "   Continuing with setup..."
fi

# Install dependencies
echo "📦 Installing dependencies..."
uv sync --extra dev

# Create VS Code workspace settings to ensure correct Python environment
echo "🔧 Configuring VS Code environment..."
mkdir -p .vscode

# Create or update VS Code settings
cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "none",
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": "explicit",
            "source.organizeImports.ruff": "explicit"
        }
    }
}
EOF

# Create development helper scripts
echo "📝 Creating development helpers..."

# Create activate script for manual environment activation
cat > activate.sh << 'EOF'
#!/bin/bash
# Activate the uv virtual environment manually
# Source this file: source ./activate.sh
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
    echo "Python path: $(which python)"
else
    echo "❌ Virtual environment not found. Run ./setup.sh first"
fi
EOF
chmod +x activate.sh

# Make run script executable
chmod +x run.sh

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Edit .env with your API keys"
echo "2. Add your FastMCP servers to src/servers_config.json"
echo "3. Run: ./run.sh"
echo ""
echo "🛠️  Available commands:"
echo "  ./run.sh              # Start the MCP platform"
echo "  ./dev.sh check        # Check environment status"
echo "  ./dev.sh server       # Run demo FastMCP server"
echo "  ./dev.sh format       # Format code"
echo "  ./dev.sh lint         # Check code quality"
echo "  source ./activate.sh  # Manually activate venv"
echo ""
echo "💡 Pro tip: VS Code will now automatically use the correct Python environment!"
echo "   If VS Code is open, reload it with Ctrl+Shift+P -> 'Developer: Reload Window'"
