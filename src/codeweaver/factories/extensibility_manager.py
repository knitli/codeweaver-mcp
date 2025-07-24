# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Central coordinator for CodeWeaver's extensibility system.

Manages the lifecycle and coordination of all extensible components including
backends, providers, data sources, with plugin discovery and dependency injection.
"""

import logging

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncContextManager

from codeweaver.backends.base import VectorBackend
from codeweaver.config import CodeWeaverConfig
from codeweaver.factories.dependency_injection import DependencyContainer
from codeweaver.factories.plugin_discovery import PluginDiscovery
from codeweaver.factories.unified_factory import UnifiedFactory
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
    auto_load_plugins: bool = True

    # Dependency injection settings
    enable_dependency_injection: bool = True
    singleton_backends: bool = True
    singleton_providers: bool = True

    # Lifecycle management
    enable_graceful_shutdown: bool = True
    shutdown_timeout: float = 30.0

    # Performance settings
    lazy_initialization: bool = True
    component_caching: bool = True

    # Compatibility settings
    enable_legacy_fallbacks: bool = True
    migration_mode: bool = False


@dataclass
class ComponentInstances:
    """Container for instantiated components."""

    backend: VectorBackend | None = None
    embedding_provider: EmbeddingProvider | None = None
    reranking_provider: RerankProvider | None = None
    data_sources: list[DataSource] | None = None
    rate_limiter: RateLimiter | None = None


class ExtensibilityManager:
    """
    Central coordinator for CodeWeaver's extensibility system.

    Manages the complete lifecycle of extensible components with plugin discovery,
    dependency injection, configuration validation, and graceful shutdown.
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

        # Initialize core systems
        self._dependency_container: DependencyContainer | None = None
        self._plugin_discovery: PluginDiscovery | None = None
        self._unified_factory: UnifiedFactory | None = None

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
            # Initialize dependency injection container
            if self.extensibility_config.enable_dependency_injection:
                self._dependency_container = DependencyContainer(
                    singleton_backends=self.extensibility_config.singleton_backends,
                    singleton_providers=self.extensibility_config.singleton_providers,
                )
                logger.info("Dependency injection container initialized")

            # Initialize plugin discovery
            if self.extensibility_config.enable_plugin_discovery:
                self._plugin_discovery = PluginDiscovery(
                    plugin_directories=self.extensibility_config.plugin_directories,
                    auto_load=self.extensibility_config.auto_load_plugins,
                )
                await self._plugin_discovery.discover_plugins()
                logger.info("Plugin discovery system initialized")

            # Initialize unified factory with discovered components
            self._unified_factory = UnifiedFactory(
                dependency_container=self._dependency_container,
                plugin_discovery=self._plugin_discovery,
            )
            logger.info("Unified factory initialized")

            # Initialize core components if not in lazy mode
            if not self.extensibility_config.lazy_initialization:
                await self._initialize_core_components()

            self._initialized = True
            logger.info("Extensibility system initialization complete")

        except Exception as e:
            logger.exception("Failed to initialize extensibility system: %s", e)
            raise

    async def _initialize_core_components(self) -> None:
        """Initialize core components (backend, providers, rate limiter)."""
        try:
            # Initialize rate limiter
            self._components.rate_limiter = RateLimiter(self.config.rate_limiting)

            # Initialize backend
            self._components.backend = await self._create_backend()

            # Initialize embedding provider
            self._components.embedding_provider = await self._create_embedding_provider()

            # Initialize reranking provider
            self._components.reranking_provider = await self._create_reranking_provider()

            # Initialize data sources
            self._components.data_sources = await self._create_data_sources()

            logger.info("Core components initialized")

        except Exception as e:
            logger.exception("Failed to initialize core components: %s", e)
            raise

    async def _create_backend(self) -> VectorBackend:
        """Create and configure the vector backend."""
        if not self._unified_factory:
            raise RuntimeError("Unified factory not initialized")

        backend = self._unified_factory.backends.create_backend(self.config.backend)

        # Register shutdown handler
        if hasattr(backend, "close"):
            self._shutdown_handlers.append(backend.close)

        return backend

    async def _create_embedding_provider(self) -> EmbeddingProvider:
        """Create and configure the embedding provider."""
        if not self._unified_factory:
            raise RuntimeError("Unified factory not initialized")

        provider = self._unified_factory.providers.create_embedding_provider(
            config=self.config.embedding, rate_limiter=self._components.rate_limiter
        )

        # Register shutdown handler
        if hasattr(provider, "close"):
            self._shutdown_handlers.append(provider.close)

        return provider

    async def _create_reranking_provider(self) -> RerankProvider | None:
        """Create and configure the reranking provider."""
        if not self._unified_factory:
            raise RuntimeError("Unified factory not initialized")

        provider = self._unified_factory.providers.get_default_reranking_provider(
            embedding_provider_name=self.config.embedding.provider,
            api_key=self.config.embedding.api_key,
            rate_limiter=self._components.rate_limiter,
        )

        # Register shutdown handler
        if provider and hasattr(provider, "close"):
            self._shutdown_handlers.append(provider.close)

        return provider

    async def _create_data_sources(self) -> list[DataSource]:
        """Create and configure data sources."""
        if not self._unified_factory:
            raise RuntimeError("Unified factory not initialized")

        if not hasattr(self.config, "data_sources") or not self.config.data_sources:
            # Fallback to file system source for backward compatibility
            return []

        sources = self._unified_factory.sources.create_multiple_sources(
            self.config.data_sources.sources
        )

        # Register shutdown handlers
        for source in sources:
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

    async def get_embedding_provider(self) -> EmbeddingProvider:
        """Get the embedding provider instance, creating it if necessary."""
        if not self._initialized:
            await self.initialize()

        if self._components.embedding_provider is None:
            self._components.embedding_provider = await self._create_embedding_provider()

        return self._components.embedding_provider

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

    def get_rate_limiter(self) -> RateLimiter:
        """Get the rate limiter instance."""
        if self._components.rate_limiter is None:
            self._components.rate_limiter = RateLimiter(self.config.rate_limiting)

        return self._components.rate_limiter

    def get_unified_factory(self) -> UnifiedFactory:
        """Get the unified factory instance.

        Returns:
            Unified factory for creating additional components

        Raises:
            RuntimeError: If the extensibility system is not initialized
        """
        if not self._unified_factory:
            raise RuntimeError("Extensibility system not initialized. Call initialize() first.")

        return self._unified_factory

    def get_component_info(self) -> dict[str, Any]:
        """Get comprehensive information about all available components.

        Returns:
            Dictionary with detailed component information
        """
        if not self._unified_factory:
            return {"error": "Extensibility system not initialized"}

        info = self._unified_factory.get_component_info()

        # Add extensibility system status
        info.update({
            "extensibility_manager": {
                "initialized": self._initialized,
                "dependency_injection_enabled": self._dependency_container is not None,
                "plugin_discovery_enabled": self._plugin_discovery is not None,
                "lazy_initialization": self.extensibility_config.lazy_initialization,
                "component_caching": self.extensibility_config.component_caching,
            },
            "components_instantiated": {
                "backend": self._components.backend is not None,
                "embedding_provider": self._components.embedding_provider is not None,
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
        if not self._unified_factory:
            return {"error": "Extensibility system not initialized"}

        # Convert config to dict for validation
        config_dict = {
            "backend": self.config.backend.__dict__
            if hasattr(self.config.backend, "__dict__")
            else {},
            "embedding": self.config.embedding.__dict__
            if hasattr(self.config.embedding, "__dict__")
            else {},
        }

        # Add data sources if available
        if hasattr(self.config, "data_sources") and self.config.data_sources:
            config_dict["data_sources"] = self.config.data_sources.__dict__

        return self._unified_factory.validate_configuration(config_dict)

    @asynccontextmanager
    async def managed_lifecycle(self) -> AsyncContextManager["ExtensibilityManager"]:
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

        # Cleanup plugin discovery
        if self._plugin_discovery:
            await self._plugin_discovery.cleanup()

        # Cleanup dependency container
        if self._dependency_container:
            await self._dependency_container.cleanup()

        # Clear component instances
        self._components = ComponentInstances()
        self._shutdown_handlers.clear()
        self._initialized = False

        logger.info("Extensibility system shutdown complete")


# Import asyncio at the end to avoid circular imports
import asyncio
