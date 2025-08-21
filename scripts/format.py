"""Format script for MCP platform."""

import subprocess
import sys


def main():
    """Run ruff format and check on the codebase."""
    try:
        print("🔧 Formatting code with ruff...")
        subprocess.run(
            ["uv", "run", "ruff", "format", "src/", "Servers/", "scripts/"], check=True
        )

        print("🔍 Checking code with ruff (ignoring line length)...")
        subprocess.run(
            [
                "uv",
                "run",
                "ruff",
                "check",
                "--fix",
                "--ignore",
                "E501",
                "src/",
                "Servers/",
                "scripts/",
            ],
            check=True,
        )

        print("✅ Code formatting complete!")
        print(
            "💡 Note: Run 'uv run ruff check src/ Servers/ scripts/' to see all issues including line length"
        )

    except subprocess.CalledProcessError as e:
        print(f"❌ Error running ruff: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
