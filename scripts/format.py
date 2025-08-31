"""Format script for MCP platform."""

import subprocess
import sys
from pathlib import Path


def _paths() -> list[str]:
    return ["src/", "Servers/", "scripts/"]


def _py_files(paths: list[str]) -> list[str]:
    files: list[str] = []
    for root in paths:
        for p in Path(root).rglob("*.py"):
            files.append(str(p))
    return files


def main():
    """Run ruff format and targeted checks on the codebase."""
    try:
        targets = _paths()

        subprocess.run(["uv", "run", "ruff", "format", *targets], check=True)

        # Fast whitespace cleanups (preview + unsafe for whitespace-only)
        subprocess.run(
            [
                "uv",
                "run",
                "ruff",
                "check",
                "--preview",
                "--fix",
                "--unsafe-fixes",
                "--select",
                "W291,W293,E3",
                *targets,
            ],
            check=True,
        )

        subprocess.run(
            [
                "uv",
                "run",
                "ruff",
                "check",
                "--fix",
                "--ignore",
                "E501",
                *targets,
            ],
            check=True,
        )

    except subprocess.CalledProcessError:
        sys.exit(1)


if __name__ == "__main__":
    main()
