# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
FastMCP client manager for CLI operations.

Manages FastMCP client connections for MCP client wrapper commands,
providing connection pooling and lifecycle management.
"""

import asyncio

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, ClassVar

from codeweaver.cli.types import CLIError
from codeweaver.cli.utils.client_logger import get_logger


logger = asyncio.run(get_logger())


Operation = Callable[[Any, Any], Awaitable[Any]]


class ClientManager:
    """Manages FastMCP client connections for CLI operations."""

    _clients: ClassVar[dict[str, Client]] = {}
    _active_contexts: ClassVar[dict[str, object]] = {}

    @classmethod
    async def get_client(cls, server_path: str, **kwargs: dict[str, Any]) -> Client:
        """
        Get or create FastMCP client for server path.

        Args:
            server_path: Path to MCP server file or URL
            **kwargs: Additional client configuration options

        Returns:
            FastMCP Client instance

        Raises:
            CLIError: If client creation fails
        """
        cache_key = cls._make_cache_key(server_path, kwargs)
        if cache_key not in cls._clients:
            try:
                cls._clients[cache_key] = cls._create_client(server_path, **kwargs)
                logger.debug("Created new FastMCP client for %s", server_path)
            except Exception as e:
                logger.exception("Failed to create FastMCP client for %s", server_path)
                raise CLIError(f"Failed to create MCP client: {e}") from e
        return cls._clients[cache_key]

    @classmethod
    def _create_client(cls, server_path: str, **kwargs) -> Client:
        """Create FastMCP client with proper configuration."""
        if server_path.startswith(("http://", "https://")):
            return Client(server_path, **kwargs)
        if Path(server_path).exists():
            return Client(server_path, **kwargs)
        raise ValueError(f"Invalid server path: {server_path}")

    @classmethod
    def _make_cache_key(cls, server_path: str, kwargs: dict[str, Any]) -> str:
        """Create cache key for client based on path and options."""
        return f"{server_path}:{hash(frozenset(kwargs.items()))}"

    @classmethod
    async def execute_with_client(
        cls, server_path: str, operation: Operation, **client_kwargs: dict[str, Any]
    ) -> Any:
        """
        Execute operation with managed client lifecycle.

        Args:
            server_path: Path to MCP server
            operation: Async function that takes a client parameter
            **client_kwargs: Additional client configuration

        Returns:
            Result from operation

        Raises:
            CLIError: If operation fails
        """
        client = await cls.get_client(server_path, **client_kwargs)
        try:
            async with client:
                return await operation(client)
        except Exception as e:
            logger.exception("Client operation failed for %s", server_path)
            raise CLIError(f"MCP client operation failed: {e}") from e

    @classmethod
    async def test_connection(cls, server_path: str, connection_timeout: float = 30.0) -> dict:
        """
        Test connection to MCP server.

        Args:
            server_path: Path to MCP server
            connection_timeout: Connection timeout in seconds

        Returns:
            Connection test results
        """

        async def _test_operation(client):
            try:
                tools = await client.list_tools()
            except Exception as e:
                return {"connected": False, "error": str(e), "tools_count": 0, "tools": []}
            else:
                return {
                    "connected": True,
                    "tools_count": len(tools),
                    "tools": [tool.name for tool in tools],
                    "server_info": getattr(client, "server_info", None),
                }

        try:
            return await cls.execute_with_client(
                server_path, _test_operation, connection_timeout=connection_timeout
            )
        except Exception as e:
            return {"connected": False, "error": str(e), "tools_count": 0, "tools": []}

    @classmethod
    async def list_tools(cls, server_path: str) -> list:
        """
        List tools available from MCP server.

        Args:
            server_path: Path to MCP server

        Returns:
            List of tool information
        """

        async def _list_tools_operation(client):
            tools = await client.list_tools()
            return [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in tools
            ]

        return await cls.execute_with_client(server_path, _list_tools_operation)

    @classmethod
    async def call_tool(cls, server_path: str, tool_name: str, arguments: dict) -> dict:
        """
        Call a specific tool on the MCP server.

        Args:
            server_path: Path to MCP server
            tool_name: Name of tool to call
            arguments: Tool arguments

        Returns:
            Tool execution result
        """

        async def _call_tool_operation(client):
            result = await client.call_tool(tool_name, arguments)
            return {
                "success": True,
                "result": result.data if hasattr(result, "data") else result,
                "structured_content": getattr(result, "structured_content", None),
            }

        try:
            return await cls.execute_with_client(server_path, _call_tool_operation)
        except Exception as e:
            return {"success": False, "error": str(e), "result": None}

    @classmethod
    async def list_resources(cls, server_path: str) -> list:
        """
        List resources available from MCP server.

        Args:
            server_path: Path to MCP server

        Returns:
            List of resource information
        """

        async def _list_resources_operation(client):
            resources = await client.list_resources()
            return [
                {
                    "uri": resource.uri,
                    "name": resource.name,
                    "description": resource.description,
                    "mime_type": getattr(resource, "mime_type", None),
                }
                for resource in resources
            ]

        return await cls.execute_with_client(server_path, _list_resources_operation)

    @classmethod
    async def close_client(cls, server_path: str) -> None:
        """
        Close specific client connection.

        Args:
            server_path: Path to MCP server
        """
        keys_to_remove = [key for key in cls._clients if key.startswith(server_path)]
        for key in keys_to_remove:
            if client := cls._clients.pop(key, None):
                try:
                    if hasattr(client, "__aexit__"):
                        await client.__aexit__(None, None, None)
                    logger.debug("Closed FastMCP client for %s", server_path)
                except Exception as e:
                    logger.warning("Error closing client for %s: %s", server_path, e)

    @classmethod
    async def close_all(cls) -> None:
        """Close all client connections and clean up resources."""
        for server_path in list(cls._clients.keys()):
            await cls.close_client(server_path.split(":")[0])
        cls._clients.clear()
        cls._active_contexts.clear()
        logger.info("All FastMCP clients closed")


async def get_mcp_client(server_path: str, **kwargs) -> Client:
    """Get FastMCP client for CLI operations."""
    return await ClientManager.get_client(server_path, **kwargs)


async def test_mcp_connection(server_path: str, connection_timeout: float = 30.0) -> dict:  # noqa: PT028
    """Test MCP server connection."""
    return await ClientManager.test_connection(server_path, connection_timeout)


async def execute_mcp_operation(
    server_path: str, operation: Operation, **client_kwargs: dict[str, Any]
) -> Any:
    """Execute operation with managed MCP client."""
    return await ClientManager.execute_with_client(server_path, operation, **client_kwargs)
