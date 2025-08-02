# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
CodeWeaver CLI package.

Provides command-line interface for CodeWeaver MCP server using cyclopts,
including MCP client operations, service management, auto-indexing, health
monitoring, and configuration management.
"""

from codeweaver.cli.app import run_async_cli, run_cli
from codeweaver.cli.types import CLIError, MCPTarget, OutputFormat, ServiceName
from codeweaver.cli.utils import ClientManager, ConfigHelper, ServerManager


__all__ = [
    "CLIError",
    "ClientManager",
    "ConfigHelper",
    "MCPTarget",
    # Types and enums
    "OutputFormat",
    # Utility classes
    "ServerManager",
    "ServiceName",
    "run_async_cli",
    # Main CLI functions
    "run_cli",
]
