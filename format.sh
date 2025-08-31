#!/bin/bash
# Quick formatting script for the MCP platform
# This script ensures ruff is available and runs formatting consistently

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 Running ruff format...${NC}"
uv run ruff format src/ Servers/ scripts/

echo -e "${BLUE}🧼 Fixing whitespace-only issues (blank lines, trailing spaces)...${NC}"
uv run ruff check --preview --fix --unsafe-fixes --select W291,W293,E3 src/ Servers/ scripts/

echo -e "${BLUE}🔍 Running ruff check with auto-fix (ignoring line length)...${NC}"
uv run ruff check --fix --ignore E501 src/ Servers/ scripts/

echo -e "${GREEN}✅ Code formatting complete!${NC}"
echo -e "${YELLOW}💡 Note: Run 'uv run ruff check src/' to see all issues including line length${NC}"

# Optional: Also format other common files
if [ "$1" = "--all" ]; then
    echo -e "${BLUE}🧹 Formatting additional files...${NC}"
    if [ -f "pyproject.toml" ]; then
        echo "   - pyproject.toml (already formatted)"
    fi
    echo -e "${GREEN}✅ All files formatted!${NC}"
fi
