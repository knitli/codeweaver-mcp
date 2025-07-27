# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Simplified CodeWeaver Factory with focused responsibilities.

Provides unified component creation, dependency injection, and lifecycle management
using the consolidated registry and base patterns. Eliminates god object anti-patterns
by focusing on core factory responsibilities.
"""

import logging
from typing import TYPE_CHECKING, Any

from codeweaver._types import (
    BaseComponentConfig,
    ComponentCreationError,
    ComponentNotFoundError,
    ComponentType,
    ConfigurationError,
    ServiceType,
    ValidationResult,
)
from codeweaver.backends.base import VectorBackend
from codeweaver.backends.config import BackendConfig
from codeweaver.factories.base import FactoryContext, create_factory_context
from codeweaver.factories.registry import ComponentRegistry, get_global_registry
from codeweaver.providers.base import EmbeddingProvider
from codeweaver.providers.factory import ProviderFactory as ExistingProviderFactory
from codeweaver.services.manager import ServicesManager
from codeweaver.sources.base import DataSource, SourceConfig

if TYPE_CHECKING:
    from codeweaver.config import CodeWeaverConfig
    from codeweaver.server import CodeWeaverServer

logger = logging.getLogger(__name__)


class CodeWeaverFactory:
    """Simplified factory with focused responsibilities."""
    
    def __init__(
        self,
        config: "CodeWeaverConfig | None" = None,
        registry: ComponentRegistry | None = None,
        services_manager: ServicesManager | None = None
    ):
        """Initialize the CodeWeaver factory.
        
        Args:
            config: Optional factory configuration
            registry: Optional component registry (uses global if None)
            services_manager: Optional services manager
        """
        # Import at runtime to avoid circular import
        from codeweaver.config import CodeWeaverConfig
        
        self._config = config or CodeWeaverConfig()
        self.registry = registry or get_global_registry()
        self.services_manager = services_manager or ServicesManager()
        
        # Use existing provider factory for now (maintains compatibility)
        self._provider_factory = ExistingProviderFactory()
        
        # Initialize built-in components
        self._initialize_builtin_components()
    
    def _initialize_builtin_components(self) -> None:
        """Initialize built-in components."""
        try:
            # Initialize built-in backends
            self._initialize_builtin_backends()
            
            # Initialize built-in sources
            self._initialize_builtin_sources()
            
            # Initialize built-in services
            self._initialize_builtin_services()
            
            logger.info("Initialized built-in components")
            
        except Exception as e:
            logger.error(f"Failed to initialize built-in components: {e}")
            raise ComponentCreationError(f"Factory initialization failed: {e}") from e
    
    def _initialize_builtin_backends(self) -> None:
        """Initialize built-in backend components."""
        # Import and register Qdrant backend
        try:
            from codeweaver.backends.qdrant import QdrantBackend
            from codeweaver._types import BackendCapabilities
            from codeweaver.factories.registry import BackendInfo
            
            backend_info = BackendInfo(
                name="qdrant",
                backend_type="qdrant",
                description="Qdrant vector database backend",
                connection_requirements={"url": "Qdrant server URL"},
                optional_parameters={"api_key": "Optional API key for authentication"}
            )
            
            capabilities = BackendCapabilities(
                supports_filtering=True,
                supports_metadata=True,
                supports_batch_operations=True,
                max_vector_dimension=65536
            )
            
            self.registry.register_backend(
                name="qdrant",
                backend_class=QdrantBackend,
                capabilities=capabilities,
                backend_info=backend_info
            )
            
        except ImportError:
            logger.warning("Qdrant backend not available - missing dependencies")
        except Exception as e:
            logger.error(f"Failed to register Qdrant backend: {e}")
    
    def _initialize_builtin_sources(self) -> None:
        """Initialize built-in source components."""
        # Import and register filesystem source
        try:
            from codeweaver.sources.filesystem import FileSystemSource
            from codeweaver._types import SourceCapabilities
            from codeweaver.factories.registry import SourceInfo
            
            source_info = SourceInfo(
                name="filesystem",
                source_type="filesystem",
                description="Local filesystem data source",
                supported_schemes=["file"],
                configuration_schema={"root_path": "string"}
            )
            
            capabilities = SourceCapabilities(
                supports_streaming=True,
                supports_filtering=True,
                supports_metadata=True,
                supports_incremental_updates=False
            )
            
            self.registry.register_source(
                name="filesystem",
                source_class=FileSystemSource,
                capabilities=capabilities,
                source_info=source_info
            )
            
        except ImportError:
            logger.warning("Filesystem source not available")
        except Exception as e:
            logger.error(f"Failed to register filesystem source: {e}")
    
    def _initialize_builtin_services(self) -> None:
        """Initialize built-in service components."""
        # Register built-in service providers
        try:
            from codeweaver.services.providers.chunking import ChunkingService
            from codeweaver.services.providers.file_filtering import FileFilteringService
            from codeweaver.services.providers.rate_limiting import RateLimitingService
            from codeweaver.services.providers.caching import CachingService
            from codeweaver._types.config import ServiceType
            from codeweaver._types.service_data import ServiceCapabilities
            
            # Register chunking service
            self.registry.register_service_provider(
                service_type=ServiceType.CHUNKING,
                provider_name="default",
                provider_class=ChunkingService,
                capabilities=ServiceCapabilities(
                    supports_streaming=True,
                    supports_batching=True,
                    max_batch_size=100
                )
            )
            
            # Register file filtering service
            self.registry.register_service_provider(
                service_type=ServiceType.FILE_FILTERING,
                provider_name="default",
                provider_class=FileFilteringService,
                capabilities=ServiceCapabilities(
                    supports_streaming=True,
                    supports_batching=False
                )
            )
            
            # Register rate limiting service
            self.registry.register_service_provider(
                service_type=ServiceType.RATE_LIMITING,
                provider_name="default",
                provider_class=RateLimitingService,
                capabilities=ServiceCapabilities(
                    supports_streaming=False,
                    supports_batching=True,
                    max_batch_size=1000,
                    supports_async=True
                )
            )
            
            # Register caching service
            self.registry.register_service_provider(
                service_type=ServiceType.CACHING,
                provider_name="default",
                provider_class=CachingService,
                capabilities=ServiceCapabilities(
                    supports_streaming=False,
                    supports_batching=True,
                    max_batch_size=1000,
                    supports_async=True
                )
            )
            
        except ImportError:
            logger.warning("Some service providers not available")
        except Exception as e:
            logger.error(f"Failed to register service providers: {e}")
    
    # Core Factory Methods
    
    async def create_backend(self, config: BackendConfig | dict[str, Any]) -> VectorBackend:
        """Create a backend instance.
        
        Args:
            config: Backend configuration
            
        Returns:
            Configured VectorBackend instance
            
        Raises:
            ComponentNotFoundError: If backend type is not registered
            ComponentCreationError: If backend creation fails
        """
        if isinstance(config, dict):
            backend_type = config.get("type")
            if not backend_type:
                raise ConfigurationError("Backend configuration must specify 'type'")
        else:
            backend_type = getattr(config, "type", None)
            if not backend_type:
                raise ConfigurationError("Backend configuration must specify 'type'")
        
        # Get backend registration
        registration = self.registry.get_backend(backend_type)
        if not registration:
            raise ComponentNotFoundError(f"Backend type '{backend_type}' not registered")
        
        # Create context with services
        context = create_factory_context(await self._get_services_context())
        
        try:
            # Create backend instance
            backend_class = registration.component_class
            if isinstance(config, dict):
                backend_config = BackendConfig(**config)
            else:
                backend_config = config
            
            backend = backend_class(backend_config)
            
            # Initialize backend if needed
            if hasattr(backend, 'initialize'):
                await backend.initialize()
            
            logger.info(f"Created backend: {backend_type}")
            return backend
            
        except Exception as e:
            logger.error(f"Failed to create backend {backend_type}: {e}")
            raise ComponentCreationError(f"Backend creation failed: {e}") from e
    
    async def create_provider(self, config: dict[str, Any] | BaseComponentConfig) -> EmbeddingProvider:
        """Create a provider instance.
        
        Args:
            config: Provider configuration
            
        Returns:
            Configured EmbeddingProvider instance
        """
        # Delegate to existing provider factory for now
        return await self._provider_factory.create_provider(config)
    
    async def create_source(self, config: SourceConfig | dict[str, Any]) -> DataSource:
        """Create a source instance.
        
        Args:
            config: Source configuration
            
        Returns:
            Configured DataSource instance
            
        Raises:
            ComponentNotFoundError: If source type is not registered
            ComponentCreationError: If source creation fails
        """
        if isinstance(config, dict):
            source_type = config.get("type")
            if not source_type:
                raise ConfigurationError("Source configuration must specify 'type'")
            source_config = SourceConfig(**config)
        else:
            source_type = getattr(config, "type", None)
            if not source_type:
                raise ConfigurationError("Source configuration must specify 'type'")
            source_config = config
        
        # Get source registration
        registration = self.registry.get_source(source_type)
        if not registration:
            raise ComponentNotFoundError(f"Source type '{source_type}' not registered")
        
        # Create context with services
        context = create_factory_context(await self._get_services_context())
        
        try:
            # Create source instance
            source_class = registration.component_class
            source = source_class(source_config)
            
            # Initialize source if needed
            if hasattr(source, 'initialize'):
                await source.initialize(context.to_dict())
            
            logger.info(f"Created source: {source_type}")
            return source
            
        except Exception as e:
            logger.error(f"Failed to create source {source_type}: {e}")
            raise ComponentCreationError(f"Source creation failed: {e}") from e
    
    async def create_service(self, service_type: ServiceType, provider_name: str = "default") -> Any:
        """Create a service instance.
        
        Args:
            service_type: Type of service to create
            provider_name: Name of the provider to use
            
        Returns:
            Configured service instance
        """
        try:
            provider_class = self.registry.get_service_provider_class(service_type, provider_name)
            
            # Create service instance
            service = provider_class()
            
            # Initialize service if needed
            if hasattr(service, 'initialize'):
                await service.initialize()
            
            logger.info(f"Created service: {service_type.value}.{provider_name}")
            return service
            
        except Exception as e:
            logger.error(f"Failed to create service {service_type.value}.{provider_name}: {e}")
            raise ComponentCreationError(f"Service creation failed: {e}") from e
    
    async def create_server(
        self, config: "CodeWeaverConfig | None" = None, *, validate_config: bool = True
    ) -> "CodeWeaverServer":
        """Create a complete CodeWeaver server instance.
        
        Args:
            config: Server configuration (uses factory config if None)
            validate_config: Whether to validate configuration before creation
            
        Returns:
            Configured CodeWeaverServer instance
            
        Raises:
            ConfigurationError: If configuration is invalid
            ComponentCreationError: If server creation fails
        """
        effective_config = config or self._config
        
        if validate_config:
            validation_result = self.validate_configuration(effective_config)
            if not validation_result.is_valid:
                raise ConfigurationError(
                    f"Configuration validation failed: {validation_result.errors}"
                )
        
        try:
            # Import server class
            from codeweaver.server import CodeWeaverServer
            
            # Create server with factory
            server = CodeWeaverServer(config=effective_config, factory=self)
            
            logger.info("Created CodeWeaver server")
            return server
            
        except Exception as e:
            logger.error(f"Failed to create server: {e}")
            raise ComponentCreationError(f"Server creation failed: {e}") from e
    
    # Validation Methods
    
    def validate_configuration(self, config: "CodeWeaverConfig") -> ValidationResult:
        """Validate a complete configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []
        
        try:
            # Validate backend configuration
            if hasattr(config, 'backend') and config.backend:
                backend_type = getattr(config.backend, 'type', None)
                if backend_type and not self.registry.is_backend_registered(backend_type):
                    errors.append(f"Backend type '{backend_type}' is not registered")
            
            # Validate source configurations
            if hasattr(config, 'sources') and config.sources:
                for source_config in config.sources:
                    source_type = getattr(source_config, 'type', None)
                    if source_type and not self.registry.is_source_registered(source_type):
                        errors.append(f"Source type '{source_type}' is not registered")
            
            # Additional validation can be added here
            
        except Exception as e:
            errors.append(f"Configuration validation error: {e}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    # Information Methods
    
    def get_available_components(self) -> dict[str, dict[str, Any]]:
        """Get information about all available components.
        
        Returns:
            Dictionary with component information organized by type
        """
        return {
            "backends": self.registry.list_backends(),
            "sources": self.registry.list_sources(),
            "services": {
                service_type.value: self.registry.list_service_providers(service_type)
                for service_type in ServiceType
            },
            "providers": self._provider_factory.get_available_embedding_providers()
        }
    
    def get_component_count(self) -> dict[str, int]:
        """Get count of registered components by type."""
        counts = self.registry.get_component_count()
        counts["providers"] = len(self._provider_factory.get_available_embedding_providers())
        return counts
    
    # Service Integration
    
    async def _get_services_context(self) -> dict[str, Any]:
        """Get services context for component creation."""
        services = {}
        
        try:
            # Create service instances directly from registry
            from codeweaver._types.config import ServiceType
            
            # Add rate limiting service
            try:
                rate_limiting_service = await self.create_service(ServiceType.RATE_LIMITING)
                if rate_limiting_service:
                    services['rate_limiting_service'] = rate_limiting_service
            except Exception as e:
                logger.warning(f"Failed to create rate limiting service: {e}")
            
            # Add caching service
            try:
                caching_service = await self.create_service(ServiceType.CACHING)
                if caching_service:
                    services['caching_service'] = caching_service
            except Exception as e:
                logger.warning(f"Failed to create caching service: {e}")
            
            # Add chunking service
            try:
                chunking_service = await self.create_service(ServiceType.CHUNKING)
                if chunking_service:
                    services['chunking_service'] = chunking_service
            except Exception as e:
                logger.warning(f"Failed to create chunking service: {e}")
            
            # Add filtering service
            try:
                filtering_service = await self.create_service(ServiceType.FILE_FILTERING)
                if filtering_service:
                    services['filtering_service'] = filtering_service
            except Exception as e:
                logger.warning(f"Failed to create filtering service: {e}")
            
            # Add services manager if available
            if self.services_manager:
                services['services_manager'] = self.services_manager
        
        except Exception as e:
            logger.warning(f"Failed to get services context: {e}")
        
        return services
    
    # Cleanup
    
    async def shutdown(self) -> None:
        """Shutdown the factory and clean up resources."""
        try:
            if self.services_manager:
                await self.services_manager.shutdown()
            
            logger.info("Factory shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during factory shutdown: {e}")


# Global factory instance
_global_factory: CodeWeaverFactory | None = None


def get_global_factory() -> CodeWeaverFactory:
    """Get the global factory instance."""
    global _global_factory
    if _global_factory is None:
        _global_factory = CodeWeaverFactory()
    return _global_factory


def reset_global_factory() -> None:
    """Reset the global factory (mainly for testing)."""
    global _global_factory
    _global_factory = None
