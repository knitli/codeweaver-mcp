# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Central coordinator for CodeWeaver's extensibility system.

Manages the lifecycle and coordination of all extensible components including
backends, providers, data sources, using the new unified factory architecture.
"""

import asyncio
import logging

from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Any

from codeweaver.backends.base import VectorBackend
from codeweaver.config import CodeWeaverConfig
from codeweaver.factories.codeweaver_factory import CodeWeaverFactory
from codeweaver.providers.base import EmbeddingProvider, RerankProvider
from codeweaver.rate_limiter import RateLimiter
from codeweaver.sources.base import DataSource


logger = logging.getLogger(__name__)


@dataclass
class ExtensibilityConfig:
    """Configuration for extensibility features."""

    # Plugin discovery settings
    enable_plugin_discovery: bool = True
    plugin_directories: list[str] | None = None
    auto_discover_plugins: bool = True

    # Factory settings
    enable_dependency_injection: bool = True
    validate_configurations: bool = True

    # Lifecycle management
    enable_graceful_shutdown: bool = True
    shutdown_timeout: float = 30.0

    # Performance settings
    lazy_initialization: bool = True
    component_caching: bool = True


@dataclass
class ComponentInstances:
    """Container for instantiated components."""

    backend: VectorBackend | None = None
    CW_EMBEDDING_PROVIDER: EmbeddingProvider | None = None
    reranking_provider: RerankProvider | None = None
    data_sources: list[DataSource] | None = None
    rate_limiter: RateLimiter | None = None


class ExtensibilityManager:
    """
    Central coordinator for CodeWeaver's extensibility system.

    Manages the complete lifecycle of extensible components using the new
    unified factory architecture with plugin discovery and validation.
    """

    def __init__(
        self, config: CodeWeaverConfig, extensibility_config: ExtensibilityConfig | None = None
    ):
        """Initialize the extensibility manager.

        Args:
            config: Main CodeWeaver configuration
            extensibility_config: Optional extensibility-specific configuration
        """
        self.config = config
        self.extensibility_config = extensibility_config or ExtensibilityConfig()

        # Initialize new factory system
        self._factory: CodeWeaverFactory | None = None

        # Component instances
        self._components = ComponentInstances()
        self._initialized = False
        self._shutdown_handlers: list[callable] = []

    async def initialize(self) -> None:
        """Initialize the extensibility system."""
        if self._initialized:
            logger.warning("ExtensibilityManager already initialized")
            return

        logger.info("Initializing extensibility system")

        try:
            # Initialize new unified factory
            self._factory = CodeWeaverFactory(
                config=self.config,
                enable_plugins=self.extensibility_config.enable_plugin_discovery,
                enable_dependency_injection=self.extensibility_config.enable_dependency_injection,
                plugin_directories=self.extensibility_config.plugin_directories,
                auto_discover_plugins=self.extensibility_config.auto_discover_plugins,
            )
            logger.info("CodeWeaver factory initialized")

            # Validate configuration if required
            if self.extensibility_config.validate_configurations:
                validation_result = self._factory.validate_configuration(self.config)
                if not validation_result.is_valid:
                    logger.warning("Configuration validation issues: %s", validation_result.errors)
                if validation_result.warnings:
                    logger.info("Configuration warnings: %s", validation_result.warnings)

            # Initialize core components if not in lazy mode
            if not self.extensibility_config.lazy_initialization:
                await self._initialize_core_components()

            self._initialized = True
            logger.info("Extensibility system initialization complete")

        except Exception:
            logger.exception("Failed to initialize extensibility system")
            raise

    async def _initialize_core_components(self) -> None:
        """Initialize core components (backend, providers, rate limiter)."""
        try:
            # Initialize rate limiter
            if hasattr(self.config, "rate_limiting"):
                self._components.rate_limiter = RateLimiter(self.config.rate_limiting)

            # Initialize backend
            self._components.backend = await self._create_backend()

            # Initialize embedding provider
            self._components.CW_EMBEDDING_PROVIDER = await self._create_CW_EMBEDDING_PROVIDER()

            # Initialize reranking provider
            self._components.reranking_provider = await self._create_reranking_provider()

            # Initialize data sources
            self._components.data_sources = await self._create_data_sources()

            logger.info("Core components initialized")

        except Exception:
            logger.exception("Failed to initialize core components")
            raise

    async def _create_backend(self) -> VectorBackend:
        """Create and configure the vector backend."""
        if not self._factory:
            raise RuntimeError("Factory not initialized")

        backend = self._factory.create_backend(self.config.backend)

        # Register shutdown handler
        if hasattr(backend, "close"):
            self._shutdown_handlers.append(backend.close)

        return backend

    async def _create_CW_EMBEDDING_PROVIDER(self) -> EmbeddingProvider:
        """Create and configure the embedding provider."""
        if not self._factory:
            raise RuntimeError("Factory not initialized")

        provider = self._factory.create_provider(self.config.embedding)

        # Register shutdown handler
        if hasattr(provider, "close"):
            self._shutdown_handlers.append(provider.close)

        return provider

    async def _create_reranking_provider(self) -> RerankProvider | None:
        """Create and configure the reranking provider."""
        # For now, reranking providers are handled by the existing provider factory
        # This could be extended in the future to use the new factory system
        return None

    async def _create_data_sources(self) -> list[DataSource]:
        """Create and configure data sources."""
        if not self._factory:
            raise RuntimeError("Factory not initialized")

        if not hasattr(self.config, "data_sources") or not self.config.data_sources:
            # No data sources configured
            return []

        sources = []
        for source_config in self.config.data_sources.sources:
            source = self._factory.create_source(source_config)
            sources.append(source)

            # Register shutdown handler
            if hasattr(source, "cleanup"):
                self._shutdown_handlers.append(source.cleanup)

        return sources

    async def get_backend(self) -> VectorBackend:
        """Get the vector backend instance, creating it if necessary."""
        if not self._initialized:
            await self.initialize()

        if self._components.backend is None:
            self._components.backend = await self._create_backend()

        return self._components.backend

    async def get_CW_EMBEDDING_PROVIDER(self) -> EmbeddingProvider:
        """Get the embedding provider instance, creating it if necessary."""
        if not self._initialized:
            await self.initialize()

        if self._components.CW_EMBEDDING_PROVIDER is None:
            self._components.CW_EMBEDDING_PROVIDER = await self._create_CW_EMBEDDING_PROVIDER()

        return self._components.CW_EMBEDDING_PROVIDER

    async def get_reranking_provider(self) -> RerankProvider | None:
        """Get the reranking provider instance, creating it if necessary."""
        if not self._initialized:
            await self.initialize()

        if self._components.reranking_provider is None:
            self._components.reranking_provider = await self._create_reranking_provider()

        return self._components.reranking_provider

    async def get_data_sources(self) -> list[DataSource]:
        """Get the data source instances, creating them if necessary."""
        if not self._initialized:
            await self.initialize()

        if self._components.data_sources is None:
            self._components.data_sources = await self._create_data_sources()

        return self._components.data_sources

    def get_rate_limiter(self) -> RateLimiter | None:
        """Get the rate limiter instance."""
        if self._components.rate_limiter is None and hasattr(self.config, "rate_limiting"):
            self._components.rate_limiter = RateLimiter(self.config.rate_limiting)

        return self._components.rate_limiter

    def get_factory(self) -> CodeWeaverFactory:
        """Get the factory instance.

        Returns:
            CodeWeaver factory for creating additional components

        Raises:
            RuntimeError: If the extensibility system is not initialized
        """
        if not self._factory:
            raise RuntimeError("Extensibility system not initialized. Call initialize() first.")

        return self._factory

    def get_component_info(self) -> dict[str, Any]:
        """Get comprehensive information about all available components.

        Returns:
            Dictionary with detailed component information
        """
        if not self._factory:
            return {"error": "Extensibility system not initialized"}

        info = self._factory.get_available_components()

        # Add extensibility system status
        info.update({
            "extensibility_manager": {
                "initialized": self._initialized,
                "lazy_initialization": self.extensibility_config.lazy_initialization,
                "component_caching": self.extensibility_config.component_caching,
                "plugin_discovery_enabled": self.extensibility_config.enable_plugin_discovery,
                "dependency_injection_enabled": self.extensibility_config.enable_dependency_injection,
            },
            "components_instantiated": {
                "backend": self._components.backend is not None,
                "CW_EMBEDDING_PROVIDER": self._components.CW_EMBEDDING_PROVIDER is not None,
                "reranking_provider": self._components.reranking_provider is not None,
                "data_sources": self._components.data_sources is not None,
                "rate_limiter": self._components.rate_limiter is not None,
            },
        })

        return info

    def validate_configuration(self) -> dict[str, Any]:
        """Validate the current configuration across all components.

        Returns:
            Validation results with any issues found
        """
        if not self._factory:
            return {"error": "Extensibility system not initialized"}

        validation_result = self._factory.validate_configuration(self.config)

        return {
            "valid": validation_result.is_valid,
            "errors": validation_result.errors,
            "warnings": validation_result.warnings,
        }

    @asynccontextmanager
    async def managed_lifecycle(self) -> AbstractAsyncContextManager["ExtensibilityManager"]:
        """Context manager for automatic lifecycle management.

        Yields:
            Initialized ExtensibilityManager instance
        """
        try:
            await self.initialize()
            yield self
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Gracefully shutdown all components and cleanup resources."""
        if not self._initialized:
            logger.debug("ExtensibilityManager not initialized, skipping shutdown")
            return

        logger.info("Starting graceful shutdown of extensibility system")

        # Execute shutdown handlers in reverse order
        for handler in reversed(self._shutdown_handlers):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler()
                else:
                    handler()
            except Exception as e:
                logger.warning("Error during component shutdown: %s", e)

        # Factory cleanup is handled automatically

        # Clear component instances
        self._components = ComponentInstances()
        self._shutdown_handlers.clear()
        self._initialized = False

        logger.info("Extensibility system shutdown complete")
