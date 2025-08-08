# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Intent service bridge for FastMCP middleware integration."""

import logging

from typing import Any

from codeweaver.cw_types import IntentResult, ServiceIntegrationError, ServiceType
from codeweaver.cw_types.services.config import ServiceConfig
from codeweaver.services.providers.base_provider import BaseServiceProvider


class IntentServiceBridge(BaseServiceProvider):
    """
    Bridge between FastMCP middleware and intent layer services.

    This bridge provides clean integration between the FastMCP middleware
    system and the intent layer, enabling dependency injection and
    service orchestration for intent processing.

    Key responsibilities:
    - Inject intent services into FastMCP context
    - Coordinate between IntentOrchestrator and AutoIndexingService
    - Provide middleware hooks for intent processing
    - Handle service lifecycle in middleware context
    - Bridge service health monitoring with FastMCP

    The bridge follows the existing ServiceBridge pattern used throughout
    CodeWeaver's service layer architecture.
    """

    def __init__(self, services_manager):
        """Initialize intent service bridge."""
        config = ServiceConfig(provider="intent_service_bridge")
        super().__init__(ServiceType.INTENT, config)
        self.services_manager = services_manager
        self.logger = logging.getLogger(__name__)
        self.intent_orchestrator = None
        self.auto_indexing_service = None
        self.bridge_initialized = False

    def _raise_intent_error(self, error: Exception, message: str) -> None:
        """Raise an IntentParsingError with the given message."""
        raise error(message)

    async def _initialize_provider(self) -> None:
        """Initialize the intent service bridge."""
        try:
            self.logger.info("Initializing Intent service bridge")
            self.intent_orchestrator = await self._get_intent_orchestrator()
            self.auto_indexing_service = await self._get_auto_indexing_service()
            if not self.intent_orchestrator:
                self._raise_intent_error(
                    ServiceIntegrationError, "IntentOrchestrator service not available"
                )
            if not self.auto_indexing_service:
                self.logger.warning(
                    "AutoIndexingService not available - background indexing disabled"
                )
            self.bridge_initialized = True
            self.logger.info("Intent service bridge initialized successfully")
        except Exception as e:
            self.logger.exception("Failed to initialize intent service bridge")
            raise ServiceIntegrationError(f"Intent bridge initialization failed: {e}") from e

    async def _shutdown_provider(self) -> None:
        """Shutdown the intent service bridge."""
        self.logger.info("Shutting down Intent service bridge")
        self.intent_orchestrator = None
        self.auto_indexing_service = None
        self.bridge_initialized = False

    async def _check_health(self) -> bool:
        """Check intent service bridge health."""
        if not self.bridge_initialized:
            return False
        try:
            if self.intent_orchestrator:
                orchestrator_health = await self.intent_orchestrator.health_check()
                if not orchestrator_health.healthy:
                    self.logger.warning(
                        "IntentOrchestrator unhealthy: %s", orchestrator_health.message
                    )
                    return False
            if self.auto_indexing_service:
                indexing_health = await self.auto_indexing_service.health_check()
                if not indexing_health.healthy:
                    self.logger.warning(
                        "AutoIndexingService unhealthy: %s", indexing_health.message
                    )
        except Exception:
            self.logger.exception("Intent service bridge health check failed")
            return False
        else:
            self.logger.info("Intent service bridge is healthy")
            return True

    async def get_context(self, user_input: str, context: dict[str, Any]) -> IntentResult:
        # sourcery skip: use-fstring-for-concatenation
        """
        Process intent through the orchestrator.

        This is the main entry point for intent processing from MCP tools.

        Args:
            user_input: Natural language input from user
            context: FastMCP context with service dependencies

        Returns:
            Intent processing result

        Raises:
            ServiceIntegrationError: If intent processing fails
        """
        if not self.bridge_initialized or not self.intent_orchestrator:
            raise ServiceIntegrationError("Intent service bridge not properly initialized")
        try:
            self.logger.info(
                "Processing intent through bridge: %s",
                user_input[:100] + "..." if len(user_input) > 100 else user_input,
            )
            result = await self.intent_orchestrator.get_context(user_input, context)
            self.logger.info(
                "Intent processed successfully: %s (strategy: %s)",
                result.success,
                result.strategy_used,
            )
        except Exception as e:
            self.logger.exception("Intent processing failed through bridge")
            raise ServiceIntegrationError(f"Intent processing failed: {e}") from e
        else:
            return result

    async def get_context_capabilities(self) -> dict[str, Any]:
        """
        Get intent layer capabilities.

        Returns information about available intent types, strategies,
        and system capabilities for MCP tool introspection.

        Returns:
            Capabilities information
        """
        if not self.bridge_initialized:
            return {"available": False, "error": "Intent service bridge not initialized"}
        try:
            capabilities = {
                "available": True,
                "intent_types": ["SEARCH", "UNDERSTAND", "ANALYZE"],
                "complexity_levels": ["SIMPLE", "MODERATE", "COMPLEX"],
                "background_indexing": bool(self.auto_indexing_service),
                "bridge_health": await self._check_health(),
            }
            if self.intent_orchestrator:
                orchestrator_caps = await self.intent_orchestrator.get_capabilities()
                capabilities |= orchestrator_caps
        except Exception as e:
            self.logger.exception("Failed to get intent capabilities")
            return {"available": False, "error": f"Capabilities query failed: {e}"}
        else:
            return capabilities

    async def inject_services_into_context(self, context: dict[str, Any]) -> None:
        """
        Inject intent services into FastMCP context.

        This method is called by middleware to make intent services
        available to MCP tools through the context.

        Args:
            context: FastMCP context to enhance with services
        """
        if not self.bridge_initialized:
            self.logger.warning("Attempting to inject services from uninitialized bridge")
            return
        try:
            if self.intent_orchestrator:
                context["intent_orchestrator"] = self.intent_orchestrator
            if self.auto_indexing_service:
                context["auto_indexing_service"] = self.auto_indexing_service
            context["intent_bridge"] = self
            context["intent_capabilities"] = await self.get_context_capabilities()
            self.logger.debug("Intent services injected into FastMCP context")
        except Exception:
            self.logger.exception("Failed to inject intent services into context")

    async def trigger_background_indexing(self, base_path: str) -> bool:
        """
        Trigger background indexing through AutoIndexingService.

        Args:
            base_path: Path to index

        Returns:
            True if indexing was triggered successfully
        """
        if not self.auto_indexing_service:
            self.logger.info("Background indexing not available")
            return False
        try:
            await self.auto_indexing_service.trigger_indexing(base_path)
            self.logger.info("Background indexing triggered for: %s", base_path)
        except Exception:
            self.logger.exception("Failed to trigger background indexing")
            return False
        else:
            return True

    async def get_bridge_status(self) -> dict[str, Any]:
        """
        Get comprehensive bridge status for monitoring.

        Returns:
            Bridge status information
        """
        try:
            status = {
                "bridge_initialized": self.bridge_initialized,
                "intent_orchestrator_available": bool(self.intent_orchestrator),
                "auto_indexing_available": bool(self.auto_indexing_service),
                "bridge_healthy": await self._check_health(),
            }
            if self.intent_orchestrator:
                orchestrator_health = await self.intent_orchestrator.health_check()
                status["intent_orchestrator_health"] = {
                    "healthy": orchestrator_health.healthy,
                    "message": orchestrator_health.message,
                }
            if self.auto_indexing_service:
                indexing_health = await self.auto_indexing_service.health_check()
                status["auto_indexing_health"] = {
                    "healthy": indexing_health.healthy,
                    "message": indexing_health.message,
                }
        except Exception as e:
            self.logger.exception("Failed to get bridge status")
            return {
                "error": f"Status query failed: {e}",
                "bridge_initialized": self.bridge_initialized,
            }
        else:
            return status

    async def _get_intent_orchestrator(self):
        """Get IntentOrchestrator service from services manager."""
        try:
            if not self.services_manager:
                self.logger.warning("ServicesManager not available")
                return None
            orchestrator = await self.services_manager.get_service("intent_orchestrator")
            if orchestrator:
                self.logger.debug("IntentOrchestrator service retrieved successfully")
            else:
                self.logger.info("IntentOrchestrator service not registered with ServicesManager")
        except Exception as e:
            self.logger.warning("Failed to get IntentOrchestrator service: %s", e)
            return None
        else:
            return orchestrator

    async def _get_auto_indexing_service(self):
        """Get AutoIndexingService from services manager."""
        try:
            if not self.services_manager:
                self.logger.warning("ServicesManager not available")
                return None
            indexing_service = await self.services_manager.get_service("auto_indexing")
            if indexing_service:
                self.logger.debug("AutoIndexingService retrieved successfully")
            else:
                self.logger.info("AutoIndexingService not registered with ServicesManager")
        except Exception as e:
            self.logger.warning("Failed to get AutoIndexingService: %s", e)
            return None
        else:
            return indexing_service
