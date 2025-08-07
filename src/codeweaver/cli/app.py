# sourcery skip: avoid-global-variables
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Main CLI application for CodeWeaver.

Provides a comprehensive command-line interface for CodeWeaver operations,
including MCP client capabilities, service management, auto-indexing,
health monitoring, and configuration management.
"""

import asyncio
import contextlib
import logging
import sys

from pathlib import Path
from typing import Annotated

import cyclopts

from cyclopts import App

# Import command modules (will be created next)
from codeweaver.cli.commands import (
    client_commands,
    config_commands,
    index_commands,
    services_commands,
    stats_commands,
)
from codeweaver.cli.types import OutputFormat
from codeweaver.cli.utils import ServerManager


logger = logging.getLogger(__name__)

# Create main CLI application
app = App(
    name="codeweaver",
    help="CodeWeaver CLI - Advanced MCP server with auto-indexing and intelligent code analysis",
    version="0.1.0",
    default_parameter=cyclopts.Parameter(show_default=True, show_choices=True),
)
# Add command groups
app.command(client_commands.app, name="client")
app.command(services_commands.app, name="services")
app.command(index_commands.app, name="index")
app.command(stats_commands.app, name="stats")
app.command(config_commands.app, name="config")


@app.default
def main(
    config: Annotated[
        Path | None, cyclopts.Parameter("--config", "-c", help="Path to configuration file")
    ] = None,
    verbose: Annotated[
        bool, cyclopts.Parameter("--verbose", "-v", help="Enable verbose logging")
    ] = False,
    quiet: Annotated[
        bool, cyclopts.Parameter("--quiet", "-q", help="Suppress non-error output")
    ] = False,
    fmt: Annotated[
        OutputFormat,
        cyclopts.Parameter(
            "--format", "-f", alias="format", help="Output format for command results"
        ),
    ] = OutputFormat.TEXT,
) -> None:
    """
    CodeWeaver CLI - Advanced MCP server with auto-indexing capabilities.

    Use 'codeweaver <command> --help' to see help for specific commands.

    Common usage:
      codeweaver services start          # Start all services
      codeweaver index start             # Start auto-indexing
      codeweaver client test <server>    # Test MCP connection
      codeweaver stats health            # Show health status
      codeweaver config generate         # Generate config file
    """
    # Configure logging based on verbosity
    if quiet:
        log_level = logging.ERROR
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Store global options for command access
    app.meta["config_path"] = str(config) if config else None
    app.meta["output_format"] = fmt
    app.meta["verbose"] = verbose
    app.meta["quiet"] = quiet

    # If no subcommand, show help
    print("Use 'codeweaver --help' or 'codeweaver <command> --help' for usage information.")


@app.command
def version() -> None:
    """Show CodeWeaver version information."""
    try:
        from codeweaver import __version__

        version_ = __version__
    except ImportError:
        version_ = "development"

    print(f"CodeWeaver CLI version {version_}")


@app.command
async def health(
    config: Annotated[
        Path | None, cyclopts.Parameter("--config", "-c", help="Path to configuration file")
    ] = None,
    fmt: Annotated[
        OutputFormat, cyclopts.Parameter("--format", "-f", alias="format", help="Output format")
    ] = OutputFormat.TEXT,
) -> None:
    """Quick health check of CodeWeaver server and services."""
    try:
        from codeweaver.cli.commands.stats_commands import format_health_output

        # Get server instance
        server = await ServerManager.ensure_server("health check", str(config) if config else None)

        # Collect health information
        health_data = {"server_running": ServerManager.is_running(), "services": {}}

        # Check services if available
        if server.services_manager:
            services = await server.services_manager.list_services()
            for service_name in services:
                try:
                    service = await server.services_manager.get_service(service_name)
                    health_data["services"][service_name] = {
                        "status": "running" if service else "stopped",
                        "healthy": bool(service),
                    }
                except Exception as e:
                    health_data["services"][service_name] = {
                        "status": "error",
                        "healthy": False,
                        "error": str(e),
                    }

        # Format and display output
        output = format_health_output(health_data, fmt)
        print(output)

    except Exception as e:
        logger.exception("Health check failed")
        if fmt == OutputFormat.JSON:
            import json

            print(json.dumps({"error": str(e), "healthy": False}))
        else:
            print(f"Health check failed: {e}")
        sys.exit(1)


@app.command
async def shutdown(
    config: Annotated[
        Path | None, cyclopts.Parameter("--config", "-c", help="Path to configuration file")
    ] = None,
    force: Annotated[
        bool, cyclopts.Parameter("--force", help="Force shutdown without graceful cleanup")
    ] = False,
) -> None:
    """Shutdown CodeWeaver server and clean up resources."""
    try:
        if ServerManager.is_running():
            print("Shutting down CodeWeaver server...")
            await ServerManager.shutdown()
            print("Server shutdown complete.")
        else:
            print("No running server instance found.")

    except Exception as e:
        logger.exception("Shutdown failed")
        print(f"Shutdown failed: {e}")
        if not force:
            sys.exit(1)


def run_cli(args: list | None = None) -> None:
    """
    Run the CLI application.

    Args:
        args: Command line arguments (defaults to sys.argv)
    """
    try:
        # Parse and run commands
        app.parse_args(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        sys.exit(1)
    except Exception as e:
        logger.exception("CLI execution failed")
        print(f"Error: {e}")
        sys.exit(1)


def run_async_cli(args: list | None = None) -> None:
    """
    Run the CLI application with async support.

    Args:
        args: Command line arguments (defaults to sys.argv)
    """
    try:
        # Create and run async event loop for CLI
        if sys.platform == "win32":
            # Windows-specific event loop policy
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        # Run the CLI application
        app.parse_args(args)

    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        sys.exit(1)
    except Exception as e:
        logger.exception("CLI execution failed")
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        # Clean up resources
        with contextlib.suppress(Exception):
            asyncio.run(ServerManager.shutdown())


if __name__ == "__main__":
    run_async_cli()
