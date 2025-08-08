# sourcery skip: avoid-global-variables
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
CodeWeaver CLI package.

Provides command-line interface for CodeWeaver MCP server using cyclopts,
including MCP client operations, service management, auto-indexing, health
monitoring, and configuration management.

Maintainer's note:
- This module serves as the entry point for the CLI application.
- It's designed to be self-contained. All commands, types and utilities for the CLI are in the `codeweaver.cli` package.
"""

from codeweaver.cli.app import run_async_cli, run_cli


if __name__ == "__main__":
    import warnings

    # Suppress this specific pydantic warning. Every import in CodeWeaver is from the top-level `pydantic` package, so the warning is coming from a dependency.
    # This is a temporary workaround until all dependencies are updated.
    warnings.filterwarnings(
        "ignore",
        message=r".*`pydantic\.error_wrappers:ValidationError` has been moved to `pydantic:ValidationError`\.",
        category=UserWarning,
        module="pydantic._migration",
    )
    # If this module is run directly, start the CLI application
    # This allows for easy testing and running without needing to import
    # from another module.
    cli_app = run_async_cli()


__all__ = (
    "run_async_cli",
    # Main CLI functions
    "run_cli",
)
