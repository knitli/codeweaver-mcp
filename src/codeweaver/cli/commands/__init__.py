# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
CLI command modules for CodeWeaver.

Contains individual command implementations organized by functionality:
- client_commands: MCP client wrapper operations
- services_commands: Service management (start/stop services)
- index_commands: Auto-indexing service management
- stats_commands: Health monitoring and statistics
- config_commands: Configuration generation and MCP integration
"""

# Import command modules for easy access
from . import client_commands, config_commands, index_commands, services_commands, stats_commands


__all__ = [
    "client_commands",
    "config_commands",
    "index_commands",
    "services_commands",
    "stats_commands",
]
