#!/bin/bash
# Development helper script
set -e

cd "$(dirname "$0")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

show_help() {
    echo -e "${BLUE}MCP Platform Development Helper${NC}"
    echo ""
    echo "Usage: ./dev.sh [command]"
    echo ""
    echo "Commands:"
    echo "  check     - Check environment and dependencies"
    echo "  lint      - Run linting with ruff"
    echo "  format    - Format code with ruff"
    echo "  server    - Run a demo FastMCP server"
    echo "  test      - Run tests (if any)"
    echo "  shell     - Open a shell with the virtual environment"
    echo "  clean     - Clean up cache and temporary files"
    echo "  install   - Install/update dependencies"
    echo "  vscode    - Configure VS Code settings"
    echo ""
}

check_env() {
    echo -e "${BLUE}🔍 Checking environment...${NC}"
    
    if [ ! -d ".venv" ]; then
        echo -e "${RED}❌ Virtual environment not found${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ Virtual environment found${NC}"
    
    echo "Python version:"
    uv run python --version
    
    echo ""
    echo "Key dependencies:"
    uv run python -c "
try:
    import fastmcp
    print(f'✅ FastMCP: {fastmcp.__version__}')
except ImportError:
    print('❌ FastMCP not found')

try:
    import mcp
    print('✅ MCP SDK: installed')
except ImportError:
    print('❌ MCP SDK not found')

try:
    import pydantic
    print(f'✅ Pydantic: {pydantic.__version__}')
except ImportError:
    print('❌ Pydantic not found')
"
}

run_lint() {
    echo -e "${BLUE}🔍 Running linter...${NC}"
    uv run ruff check .
}

run_format() {
    echo -e "${BLUE}🎨 Formatting code...${NC}"
    uv run ruff format .
    uv run ruff check --fix .
}

run_demo_server() {
    echo -e "${BLUE}🚀 Starting demo FastMCP server...${NC}"
    echo "Press Ctrl+C to stop"
    uv run python Servers/demo_server.py
}

run_shell() {
    echo -e "${BLUE}🐚 Opening shell with virtual environment...${NC}"
    echo "Virtual environment will be activated"
    echo "Type 'exit' to return"
    source .venv/bin/activate
    exec bash
}

clean_up() {
    echo -e "${BLUE}🧹 Cleaning up...${NC}"
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    rm -rf .pytest_cache/ 2>/dev/null || true
    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

install_deps() {
    echo -e "${BLUE}📦 Installing/updating dependencies...${NC}"
    uv sync --extra dev
}

configure_vscode() {
    echo -e "${BLUE}🔧 Configuring VS Code...${NC}"
    mkdir -p .vscode
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
    echo -e "${GREEN}✅ VS Code configured${NC}"
}

case "${1:-help}" in
    check)
        check_env
        ;;
    lint)
        run_lint
        ;;
    format)
        run_format
        ;;
    server)
        run_demo_server
        ;;
    test)
        echo -e "${YELLOW}⚠️  No tests configured yet${NC}"
        ;;
    shell)
        run_shell
        ;;
    clean)
        clean_up
        ;;
    install)
        install_deps
        ;;
    vscode)
        configure_vscode
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac
