# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
CLI-specific types and enums for CodeWeaver CLI.
"""

from typing import Literal


# Output format options
OutputFormat = Literal["table", "json", "yaml", "text"]

# Service names that can be managed via CLI
ServiceName = Literal[
    "auto_indexing",
    "chunking",
    "filtering",
    "monitoring",
    "telemetry",
    "intent_bridge",
    "intent_orchestrator",
]

# MCP client target applications
MCPTarget = Literal["claude-desktop", "claude-code", "cursor", "mcp-json"]

# Configuration profiles
ConfigProfile = Literal[
    "codeweaver_original", "minimal", "performance", "development", "production"
]


class CLIError(Exception):
    """Base exception for CLI operations with user-friendly messages."""

    def __init__(self, message: str, exit_code: int = 1):
        """Initialize CLI error with message and exit code."""
        super().__init__(message)
        self.exit_code = exit_code


class ServiceNotAvailableError(CLIError):
    """Raised when a required service is not available."""

    def __init__(self, service_name: str):
        """Initialize with service name."""
        super().__init__(f"Service '{service_name}' is not available", exit_code=2)
        self.service_name = service_name


class ServerNotRunningError(CLIError):
    """Raised when server operations require a running server."""

    def __init__(self, operation: str):
        """Initialize with operation description."""
        super().__init__(f"Server must be running for operation: {operation}", exit_code=3)
        self.operation = operation


class ConfigurationError(CLIError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, config_path: str | None = None):
        """Initialize with configuration error details."""
        if config_path:
            message = f"Configuration error in {config_path}: {message}"
        super().__init__(message, exit_code=4)
        self.config_path = config_path
