# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Server manager for CLI operations.

Manages CodeWeaver server lifecycle for CLI commands, providing
server instances for service management and other operations.
"""

import logging

from typing import TYPE_CHECKING

from codeweaver.cli.types import CLIError, ServerNotRunningError
from codeweaver.config import get_config_manager
from codeweaver.server import CodeWeaverServer, create_server


if TYPE_CHECKING:
    from codeweaver.services import BasicServiceProvider, ServicesManager

logger = logging.getLogger(__name__)


class ServerManager:
    """Manages server lifecycle for CLI operations."""

    _instance: CodeWeaverServer | None = None
    _config_path: str | None = None

    @classmethod
    async def get_server(
        cls, config_path: str | None = None, *, force_new: bool = False
    ) -> CodeWeaverServer:
        """
        Get or create server instance.

        Args:
            config_path: Optional path to configuration file
            force_new: Force creation of new server instance

        Returns:
            Initialized CodeWeaver server instance

        Raises:
            CLIError: If server initialization fails
        """
        if cls._instance is None or force_new or config_path != cls._config_path:
            await cls._create_server(config_path)
        return cls._instance

    @classmethod
    async def _create_server(cls, config_path: str | None = None) -> None:
        """Create and initialize new server instance."""
        try:
            if cls._instance is not None:
                await cls.shutdown()
            config_manager = get_config_manager(config_path)
            config = config_manager.get_config()
            cls._instance = create_server(config)
            await cls._instance.initialize()
            cls._config_path = config_path
            logger.info("Server instance created and initialized for CLI operations")
        except Exception as e:
            logger.exception("Failed to create server instance for CLI")
            raise CLIError(f"Failed to initialize server: {e}") from e

    @classmethod
    async def ensure_server(
        cls, operation: str, config_path: str | None = None
    ) -> CodeWeaverServer:
        """
        Ensure server is available for the given operation.

        Args:
            operation: Description of operation requiring server
            config_path: Optional path to configuration file

        Returns:
            Running server instance

        Raises:
            ServerNotRunningError: If server cannot be started
        """
        try:
            return await cls.get_server(config_path)
        except Exception as e:
            logger.exception("Cannot ensure server for operation '%s'", operation)
            raise ServerNotRunningError(operation) from e

    @classmethod
    async def get_services_manager(cls, config_path: str | None = None) -> "ServicesManager":
        """
        Get services manager from server instance.

        Args:
            config_path: Optional path to configuration file

        Returns:
            ServicesManager instance

        Raises:
            CLIError: If services manager is not available
        """
        server = await cls.ensure_server("get services manager", config_path)
        if server.services_manager is None:
            raise CLIError("Services manager is not available")
        return server.services_manager

    @classmethod
    async def get_service(
        cls, service_name: str, config_path: str | None = None
    ) -> "type[BasicServiceProvider]":
        """
        Get specific service from services manager.

        Args:
            service_name: Name of service to retrieve
            config_path: Optional path to configuration file

        Returns:
            Service instance

        Raises:
            CLIError: If service is not available
        """
        from codeweaver.cli.utils import raise_cli_error

        services_manager = await cls.get_services_manager(config_path)
        try:
            service = await services_manager.get_service(service_name)
            if service is None:
                raise_cli_error("Service '%s' is not available", service_name)
        except Exception as e:
            logger.exception("Failed to get service '%s'", service_name)
            raise_cli_error("Failed to get service '%s': %s", service_name, e)
        else:
            return service

    @classmethod
    async def shutdown(cls) -> None:
        """Shutdown server instance and clean up resources."""
        if cls._instance is not None:
            try:
                await cls._instance.shutdown()
                logger.info("Server instance shutdown complete")
            except Exception as e:
                logger.warning("Error during server shutdown: %s", e)
            finally:
                cls._instance = None
                cls._config_path = None

    @classmethod
    def is_running(cls) -> bool:
        """Check if server instance is currently running."""
        return cls._instance is not None and cls._instance._initialized

    @classmethod
    async def start_minimal_server(cls, config_path: str | None = None) -> CodeWeaverServer:
        """
        Start minimal server instance for CLI operations.

        This creates a lightweight server instance suitable for CLI operations
        without starting the full MCP server listener.

        Args:
            config_path: Optional path to configuration file

        Returns:
            Minimal server instance
        """
        try:
            await cls._create_server(config_path)
        except Exception as e:
            logger.exception("Failed to start minimal server")
            raise CLIError(f"Failed to start minimal server: {e}") from e
        else:
            return cls._instance


async def get_server(config_path: str | None = None) -> CodeWeaverServer:
    """Get server instance for CLI operations."""
    return await ServerManager.get_server(config_path)


async def get_services_manager(config_path: str | None = None) -> "ServicesManager":
    """Get services manager for CLI operations."""
    return await ServerManager.get_services_manager(config_path)


async def get_service(
    service_name: str, config_path: str | None = None
) -> "type[BasicServiceProvider]":
    """Get specific service for CLI operations."""
    return await ServerManager.get_service(service_name, config_path)


async def ensure_server_for_operation(
    operation: str, config_path: str | None = None
) -> CodeWeaverServer:
    """Ensure server is available for the given operation."""
    return await ServerManager.ensure_server(operation, config_path)
