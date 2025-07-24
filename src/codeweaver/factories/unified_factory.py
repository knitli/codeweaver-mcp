# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Unified factory classes for creating backends, providers, and data sources.

Coordinates between existing factory implementations and the new extensibility system
to provide a consistent interface for component creation.
"""

import logging

from typing import Any

from codeweaver.backends.base import VectorBackend
from codeweaver.backends.config import BackendConfig
from codeweaver.backends.factory import BackendFactory as BackendFactoryImpl
from codeweaver.config import EmbeddingConfig
from codeweaver.providers.base import EmbeddingProvider, RerankProvider
from codeweaver.providers.factory import ProviderFactory as ProviderFactoryImpl
from codeweaver.rate_limiter import RateLimiter
from codeweaver.sources.base import DataSource, SourceConfig
from codeweaver.sources.factory import SourceFactory as SourceFactoryImpl


logger = logging.getLogger(__name__)


class BackendFactory:
    """
    Unified factory for vector database backends.

    Integrates with existing backend factory implementation while providing
    extensibility features like plugin discovery and dependency injection.
    """

    def __init__(self, dependency_container=None):
        """Initialize the backend factory.

        Args:
            dependency_container: Optional dependency injection container
        """
        self._container = dependency_container
        self._backend_factory = BackendFactoryImpl()

    def create_backend(self, config: BackendConfig, **kwargs: Any) -> VectorBackend:
        """Create a vector database backend.

        Args:
            config: Backend configuration
            **kwargs: Additional configuration options

        Returns:
            Configured vector backend instance
        """
        # Inject dependencies if container is available
        if self._container:
            dependencies = self._container.resolve_dependencies("backend", config.provider)
            kwargs |= dependencies

        return self._backend_factory.create_backend(config)

    def create_from_url(self, url: str, **kwargs: Any) -> VectorBackend:
        """Create a backend from a connection URL.

        Args:
            url: Connection URL
            **kwargs: Additional configuration options

        Returns:
            Configured vector backend instance
        """
        return self._backend_factory.create_from_url(url, **kwargs)

    def list_supported_providers(self) -> dict[str, dict[str, bool]]:
        """List all supported providers and their capabilities.

        Returns:
            Dictionary mapping provider names to their capabilities
        """
        return self._backend_factory.list_supported_providers()

    def register_backend(
        self, provider: str, backend_class: type[VectorBackend], *, supports_hybrid: bool = False
    ) -> None:
        """Register a new backend provider.

        Args:
            provider: Provider name
            backend_class: Backend implementation class
            supports_hybrid: Whether the backend supports hybrid search
        """
        self._backend_factory.register_backend(provider, backend_class, supports_hybrid)


class ProviderFactory:
    """
    Unified factory for embedding and reranking providers.

    Integrates with existing provider factory implementation while providing
    extensibility features and unified provider management.
    """

    def __init__(self, dependency_container=None):
        """Initialize the provider factory.

        Args:
            dependency_container: Optional dependency injection container
        """
        self._container = dependency_container
        self._provider_factory = ProviderFactoryImpl()

    def create_embedding_provider(
        self, config: EmbeddingConfig, rate_limiter: RateLimiter | None = None, **kwargs: Any
    ) -> EmbeddingProvider:
        """Create an embedding provider.

        Args:
            config: Embedding configuration
            rate_limiter: Optional rate limiter
            **kwargs: Additional configuration options

        Returns:
            Configured embedding provider instance
        """
        # Inject dependencies if container is available
        if self._container:
            dependencies = self._container.resolve_dependencies("embedding", config.provider)
            kwargs |= dependencies

        return self._provider_factory.create_embedding_provider(config, rate_limiter)

    def create_reranking_provider(
        self,
        provider_name: str,
        api_key: str | None = None,
        model: str | None = None,
        rate_limiter: RateLimiter | None = None,
        **kwargs: Any,
    ) -> RerankProvider:
        """Create a reranking provider.

        Args:
            provider_name: Name of the reranking provider
            api_key: API key for the provider
            model: Model name to use
            rate_limiter: Optional rate limiter
            **kwargs: Additional provider-specific configuration

        Returns:
            Configured reranking provider instance
        """
        # Inject dependencies if container is available
        if self._container:
            dependencies = self._container.resolve_dependencies("reranking", provider_name)
            kwargs |= dependencies

        return self._provider_factory.create_reranking_provider(
            provider_name, api_key, model, rate_limiter, **kwargs
        )

    def get_default_reranking_provider(
        self,
        embedding_provider_name: str,
        api_key: str | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> RerankProvider | None:
        """Get the default reranking provider for an embedding provider.

        Args:
            embedding_provider_name: Name of the embedding provider
            api_key: API key to use
            rate_limiter: Optional rate limiter

        Returns:
            Default reranking provider or None if none available
        """
        return self._provider_factory.get_default_reranking_provider(
            embedding_provider_name, api_key, rate_limiter
        )

    def list_available_embedding_providers(self) -> dict[str, Any]:
        """List all available embedding providers.

        Returns:
            Dictionary of available embedding providers and their info
        """
        return self._provider_factory.registry.get_available_embedding_providers()

    def list_available_reranking_providers(self) -> dict[str, Any]:
        """List all available reranking providers.

        Returns:
            Dictionary of available reranking providers and their info
        """
        return self._provider_factory.registry.get_available_reranking_providers()


class SourceFactory:
    """
    Unified factory for data sources.

    Integrates with existing source factory implementation while providing
    extensibility features and multi-source coordination.
    """

    def __init__(self, dependency_container=None):
        """Initialize the source factory.

        Args:
            dependency_container: Optional dependency injection container
        """
        self._container = dependency_container
        self._source_factory = SourceFactoryImpl()

    def create_source(self, source_type: str, config: SourceConfig, **kwargs: Any) -> DataSource:
        """Create a data source instance.

        Args:
            source_type: Type of data source to create
            config: Configuration for the data source
            **kwargs: Additional configuration options

        Returns:
            Configured data source instance
        """
        # Inject dependencies if container is available
        if self._container:
            dependencies = self._container.resolve_dependencies("source", source_type)
            config.update(dependencies)

        return self._source_factory.create_source(source_type, config)

    def create_multiple_sources(
        self, source_configs: list[dict[str, Any]], **kwargs: Any
    ) -> list[DataSource]:
        """Create multiple data source instances.

        Args:
            source_configs: List of source configuration dictionaries
            **kwargs: Additional configuration options

        Returns:
            List of configured data source instances
        """
        return self._source_factory.create_multiple_sources(source_configs)

    async def validate_source_config(self, source_type: str, config: SourceConfig) -> bool:
        """Validate a source configuration.

        Args:
            source_type: Type of data source
            config: Configuration to validate

        Returns:
            True if configuration is valid, False otherwise
        """
        return await self._source_factory.validate_source_config(source_type, config)

    def list_available_sources(self) -> dict[str, dict[str, Any]]:
        """Get information about all available data source types.

        Returns:
            Dictionary mapping source types to their information
        """
        return self._source_factory.list_available_sources()


class UnifiedFactory:
    """
    Unified factory coordinating all component creation.

    Provides a single entry point for creating backends, providers, and data sources
    with integrated dependency injection and plugin discovery.
    """

    def __init__(self, dependency_container=None, plugin_discovery=None):
        """Initialize the unified factory.

        Args:
            dependency_container: Optional dependency injection container
            plugin_discovery: Optional plugin discovery system
        """
        self._container = dependency_container
        self._plugin_discovery = plugin_discovery

        # Initialize component factories
        self.backends = BackendFactory(dependency_container)
        self.providers = ProviderFactory(dependency_container)
        self.sources = SourceFactory(dependency_container)

        # Initialize plugin discovery if available
        if plugin_discovery:
            self._discover_plugins()

    def _discover_plugins(self) -> None:
        """Discover and register available plugins."""
        if not self._plugin_discovery:
            return

        try:
            self._discover_all_plugins()
        except Exception as e:
            logger.warning("Failed to discover plugins: %s", e)

    def _discover_all_plugins(self) -> None:
        """Discover and register all available plugins."""
        # Discover all plugin types
        backend_plugins = self._plugin_discovery.discover_backend_plugins()
        provider_plugins = self._plugin_discovery.discover_provider_plugins()
        source_plugins = self._plugin_discovery.discover_source_plugins()

        # Register each plugin type
        self._register_backend_plugins(backend_plugins)
        self._register_provider_plugins(provider_plugins)
        self._register_source_plugins(source_plugins)

    def _register_backend_plugins(self, plugins) -> None:
        """Register backend plugins."""
        if not plugins:
            logger.info("No backend plugins discovered")
            return

        logger.info("Discovered %d backend plugins", len(plugins))
        for plugin in plugins:
            self.backends.register_backend(
                plugin.name, plugin.implementation, plugin.supports_hybrid
            )

    def _register_provider_plugins(self, plugins) -> None:
        """Register provider plugins."""
        if not plugins:
            logger.info("No provider plugins discovered")
            return

        logger.info("Discovered %d provider plugins", len(plugins))
        for plugin in plugins:
            self._register_single_provider_plugin(plugin)

    def _register_single_provider_plugin(self, plugin) -> None:
        """Register a single provider plugin."""
        # Register with underlying provider registry
        if plugin.capabilities.supports_embedding:
            self.providers._provider_factory.registry.register_embedding_provider(  # noqa: SLF001
                plugin.name, plugin.implementation, plugin.provider_info
            )
        if plugin.capabilities.supports_reranking:
            self.providers._provider_factory.registry.register_reranking_provider(  # noqa: SLF001
                plugin.name, plugin.implementation, plugin.provider_info
            )

    def _register_source_plugins(self, plugins) -> None:
        """Register source plugins."""
        if not plugins:
            logger.info("No source plugins discovered")
            return

        logger.info("Discovered %d source plugins", len(plugins))
        for plugin in plugins:
            self.sources._source_factory.get_source_registry().register(  # noqa: SLF001
                plugin.name, plugin.implementation
            )

    def get_component_info(self) -> dict[str, Any]:
        """Get information about all available components.

        Returns:
            Dictionary with information about backends, providers, and sources
        """
        return {
            "backends": self.backends.list_supported_providers(),
            "embedding_providers": self.providers.list_available_embedding_providers(),
            "reranking_providers": self.providers.list_available_reranking_providers(),
            "data_sources": self.sources.list_available_sources(),
            "plugin_discovery_enabled": self._plugin_discovery is not None,
            "dependency_injection_enabled": self._container is not None,
        }

    def validate_configuration(self, config: dict[str, Any]) -> dict[str, Any]:
        """Validate a complete configuration across all components.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Validation results with any issues found
        """
        validation_results = {"valid": True, "issues": [], "warnings": []}

        # Validate backend configuration
        if "backend" in config:
            backend_config = config["backend"]
            try:
                BackendConfig(**backend_config)
            except Exception as e:
                validation_results["valid"] = False
                validation_results["issues"].append(f"Backend config invalid: {e}")

        # Validate embedding configuration
        if "embedding" in config:
            embedding_config = config["embedding"]
            provider = embedding_config.get("provider")
            if provider not in self.providers.list_available_embedding_providers():
                validation_results["valid"] = False
                validation_results["issues"].append(f"Unknown embedding provider: {provider}")

        # Validate data source configurations
        if "data_sources" in config and "sources" in config["data_sources"]:
            available_sources = self.sources.list_available_sources()
            for source_config in config["data_sources"]["sources"]:
                source_type = source_config.get("type")
                if source_type not in available_sources:
                    validation_results["valid"] = False
                    validation_results["issues"].append(f"Unknown data source type: {source_type}")
                elif not available_sources[source_type]["implemented"]:
                    validation_results["warnings"].append(
                        f"Data source '{source_type}' is not fully implemented"
                    )

        return validation_results
