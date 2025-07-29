# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Middleware bridge for coordinating FastMCP middleware with service layer."""

import logging

from typing import Any

from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.middleware.middleware import CallNext

from codeweaver.services.manager import ServicesManager
from codeweaver.types import ChunkingService, FilteringService, ServiceType


class ServiceBridge(Middleware):
    """Bridge between FastMCP middleware and service layer."""

    def __init__(self, services_manager: ServicesManager):
        """Initialize the service bridge with the services manager."""
        self._services_manager = services_manager
        self._logger = logging.getLogger("codeweaver.services.middleware_bridge")

    async def on_call_tool(self, context: MiddlewareContext, call_next: CallNext) -> Any:
        """Inject services into context for tools that need them."""
        try:
            # Check if this tool needs service injection
            if self._needs_service_injection(context):
                await self._inject_services(context)
                self._logger.debug("Injected services for tool: %s", context.message.name)

            # Continue with normal tool execution
            return await call_next(context)

        except Exception:
            self._logger.exception("Service bridge error for tool %s")

            # Continue execution even if service injection fails
            return await call_next(context)

    def _needs_service_injection(self, context: MiddlewareContext) -> bool:
        """Check if this tool call needs service injection."""
        if not hasattr(context.message, "name"):
            return False

        # Tools that need service injection
        service_tools = {
            "index_codebase": [ServiceType.CHUNKING, ServiceType.FILTERING],
            "search_code": [ServiceType.FILTERING],
            "ast_grep_search": [ServiceType.FILTERING],
        }

        return context.message.name in service_tools

    async def _inject_services(self, context: MiddlewareContext) -> None:
        """Inject appropriate services into the context."""
        tool_name = context.message.name

        # Define service injection mappings
        service_mappings = {
            "index_codebase": {
                "chunking_service": ServiceType.CHUNKING,
                "filtering_service": ServiceType.FILTERING,
            },
            "search_code": {"filtering_service": ServiceType.FILTERING},
            "ast_grep_search": {"filtering_service": ServiceType.FILTERING},
        }

        # Inject services for this tool
        if tool_name in service_mappings:
            for context_key, service_type in service_mappings[tool_name].items():
                try:
                    if service := self._services_manager.get_service(service_type):
                        context.fastmcp_context.set_state_value(context_key, service)
                        self._logger.debug("Injected %s service into context", service_type.value)
                    else:
                        self._logger.warning("Service %s not available", service_type.value)

                except Exception as e:
                    self._logger.warning("Failed to inject %s service: %s", service_type.value, e)


class ServiceCoordinator:
    """Coordinates between different service layers."""

    def __init__(self, services_manager: ServicesManager):
        """Initialize the service coordinator with the services manager."""
        self._services_manager = services_manager
        self._logger = logging.getLogger("codeweaver.services.coordinator")

    async def get_chunking_service(self) -> ChunkingService | None:
        """Get the active chunking service."""
        try:
            return self._services_manager.get_chunking_service()
        except Exception as e:
            self._logger.warning("Failed to get chunking service: %s", e)
            return None

    async def get_filtering_service(self) -> FilteringService | None:
        """Get the active filtering service."""
        try:
            return self._services_manager.get_filtering_service()
        except Exception as e:
            self._logger.warning("Failed to get filtering service: %s", e)
            return None

    async def coordinate_indexing(self, base_path: str, **kwargs) -> dict[str, Any]:
        """Coordinate chunking and filtering services for indexing."""
        results = {
            "chunking_available": False,
            "filtering_available": False,
            "services_healthy": False,
        }

        try:
            # Get services
            chunking_service = await self.get_chunking_service()
            filtering_service = await self.get_filtering_service()

            results["chunking_available"] = chunking_service is not None
            results["filtering_available"] = filtering_service is not None

            # Check health
            if chunking_service and filtering_service:
                chunking_health = await chunking_service.health_check()
                filtering_health = await filtering_service.health_check()

                results["services_healthy"] = chunking_health.status.value in [
                    "healthy",
                    "degraded",
                ] and filtering_health.status.value in ["healthy", "degraded"]

                # Add service capabilities info
                results["chunking_capabilities"] = chunking_service.get_supported_languages()
                results["filtering_patterns"] = filtering_service.get_active_patterns()

        except Exception as e:
            self._logger.exception("Service coordination failed")
            results["error"] = str(e)
            return results

        else:
            return results
