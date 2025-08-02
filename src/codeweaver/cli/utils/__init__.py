# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
CLI utilities for CodeWeaver CLI commands.
"""

from codeweaver.cli.utils.client_manager import (
    ClientManager,
    execute_mcp_operation,
    get_mcp_client,
    test_mcp_connection,
)
from codeweaver.cli.utils.config_helper import (
    ConfigHelper,
    generate_config_file,
    insert_mcp_server_config,
    list_mcp_client_configs,
)
from codeweaver.cli.utils.helpers import raise_cli_error
from codeweaver.cli.utils.server_manager import ServerManager, ensure_server_for_operation


__all__ = [
    "ClientManager",
    "ConfigHelper",
    "ServerManager",
    "ensure_server_for_operation",
    "execute_mcp_operation",
    "generate_config_file",
    "get_mcp_client",
    "insert_mcp_server_config",
    "list_mcp_client_configs",
    "raise_cli_error",
    "test_mcp_connection",
]
