# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Main CodeWeaver Factory orchestrator.

Provides unified creation, configuration, and lifecycle management for
backends, providers, and sources with plugin extensibility and dependency injection.
"""

import logging

from typing import TYPE_CHECKING, Any

from codeweaver._types import BaseComponentConfig, ComponentType, ServiceType, ValidationResult
from codeweaver.backends.base import VectorBackend
from codeweaver.backends.config import BackendConfig


if TYPE_CHECKING:
    from codeweaver.config import CodeWeaverConfig
    from codeweaver.factories.service_registry import ServiceRegistry
from codeweaver._types import (
    ComponentCreationError,
    ComponentNotFoundError,
    ConfigurationError,
    PluginInfo,
)
from codeweaver.factories.backend_registry import BackendRegistry
from codeweaver.factories.error_handling import ErrorHandler, GracefulDegradationManager
from codeweaver.factories.initialization import FactoryInitializer
from codeweaver.factories.plugin_protocols import PluginDiscoveryEngine

# ServiceRegistry imported lazily to avoid circular imports
from codeweaver.factories.source_registry import SourceRegistry
from codeweaver.providers.base import EmbeddingProvider
from codeweaver.providers.factory import ProviderFactory as ExistingProviderFactory


if TYPE_CHECKING:
    from codeweaver.server import CodeWeaverServer
from codeweaver.sources.base import DataSource, SourceConfig


logger = logging.getLogger(__name__)


class CapabilityManager:
    """Unified capability queries across all components."""

    def __init__(
        self,
        backend_registry: BackendRegistry,
        provider_registry: Any,  # Existing provider registry
        source_registry: SourceRegistry,
        service_registry: "ServiceRegistry",
    ):
        """Initialize the capability manager with registries."""
        self._backend_registry = backend_registry
        self._provider_registry = provider_registry
        self._source_registry = source_registry
        self._service_registry = service_registry

    def get_component_capabilities(self, component_type: ComponentType, component_name: str) -> Any:
        """Universal capability query pattern."""
        registry = self._get_registry(component_type)
        return registry.get_capabilities(component_name)

    def supports_capability(
        self, component_type: ComponentType, component_name: str, capability: str
    ) -> bool:
        """Check if component supports specific capability."""
        try:
            capabilities = self.get_component_capabilities(component_type, component_name)
            return hasattr(capabilities, capability) and getattr(capabilities, capability, False)
        except Exception:
            return False

    def _get_registry(self, component_type: ComponentType) -> Any:
        """Get registry for component type."""
        registries = {
            ComponentType.BACKEND: self._backend_registry,
            ComponentType.PROVIDER: self._provider_registry,
            ComponentType.SOURCE: self._source_registry,
            ComponentType.SERVICE: self._service_registry,
        }
        return registries[component_type]


class ConfigurationValidator:
    """Multi-stage configuration validation."""

    def __init__(
        self,
        backend_registry: BackendRegistry,
        provider_registry: Any,
        source_registry: SourceRegistry,
        service_registry: "ServiceRegistry",
    ):
        """Initialize the configuration validator with registries."""
        self._backend_registry = backend_registry
        self._provider_registry = provider_registry
        self._source_registry = source_registry
        self._service_registry = service_registry

    def validate_configuration(self, config: "CodeWeaverConfig") -> ValidationResult:
        """Comprehensive configuration validation."""
        errors = []
        warnings = []

        validation_pipeline = [
            self._validate_syntax,
            self._validate_component_availability,
            self._validate_capabilities,
            self._validate_dependencies,
        ]

        for validator in validation_pipeline:
            try:
                result = validator(config)
                if not result.is_valid:
                    errors.extend(result.errors)
                    warnings.extend(result.warnings)
                    # Continue validation to collect all issues
            except Exception as e:
                errors.append(f"Validation failed: {e}")

        return ValidationResult(is_valid=not errors, errors=errors, warnings=warnings)

    def _validate_syntax(self, config: "CodeWeaverConfig") -> ValidationResult:
        """Basic syntax and structure validation."""
        errors = []
        warnings = []

        # Check required fields
        if not hasattr(config, "backend") or not config.backend:
            errors.append("Missing backend configuration")

        if not hasattr(config, "providers") or not config.providers.embedding:
            errors.append("Missing embedding configuration")

        return ValidationResult(is_valid=not errors, errors=errors, warnings=warnings)

    def _validate_component_availability(self, config: "CodeWeaverConfig") -> ValidationResult:
        """Validate component existence."""
        errors = []
        warnings = []

        # Check backend availability
        if (
            hasattr(config, "backend")
            and config.backend
            and not self._backend_registry.has_component(config.backend.provider)
        ):
            errors.append(f"Backend '{config.backend.provider}' not available")

        # Check embedding provider availability
        if hasattr(config, "providers") and config.providers.embedding:
            available_providers = self._provider_registry.get_available_embedding_providers()
            if config.providers.embedding.provider not in available_providers:
                errors.append(
                    f"Embedding provider '{config.providers.embedding.provider}' not available"
                )

        return ValidationResult(is_valid=not errors, errors=errors, warnings=warnings)

    def _validate_capabilities(self, config: "CodeWeaverConfig") -> ValidationResult:
        """Validate capability compatibility."""
        return self._validate_capabilities_and_dependencies()

    def _validate_dependencies(self, config: "CodeWeaverConfig") -> ValidationResult:
        """Validate inter-component dependencies."""
        return self._validate_capabilities_and_dependencies()

    # TODO Rename this here and in `_validate_capabilities` and `_validate_dependencies`
    def _validate_capabilities_and_dependencies(self):
        errors = []
        warnings = []
        return ValidationResult(is_valid=not errors, errors=errors, warnings=warnings)


class CodeWeaverFactory:
    """
    Main factory orchestrator for all CodeWeaver components.

    Provides unified creation, configuration, and lifecycle management
    for backends, providers, and sources with plugin extensibility.
    """

    def __init__(
        self,
        config: "CodeWeaverConfig | None" = None,
        *,
        enable_plugins: bool = True,
        enable_dependency_injection: bool = True,
        plugin_directories: list[str] | None = None,
        auto_discover_plugins: bool = True,
    ):  # sourcery skip: remove-empty-nested-block, remove-redundant-if
        """Initialize the CodeWeaver factory.

        Args:
            config: Optional factory configuration
            enable_plugins: Whether to enable plugin discovery and loading
            enable_dependency_injection: Whether to enable dependency injection
            plugin_directories: Custom plugin directories to scan
            auto_discover_plugins: Whether to automatically discover plugins on initialization
        """
        # Import at runtime to avoid circular import
        from codeweaver.config import CodeWeaverConfig

        self._config = config or CodeWeaverConfig()

        # Initialize registries
        self._backend_registry = BackendRegistry()
        self._provider_registry = (
            ExistingProviderFactory().registry
        )  # Use existing A+ implementation
        self._source_registry = SourceRegistry()

        # Import ServiceRegistry lazily to avoid circular imports
        from codeweaver.factories.service_registry import ServiceRegistry

        self._service_registry = ServiceRegistry()

        # Initialize supporting systems
        self._capability_manager = CapabilityManager(
            backend_registry=self._backend_registry,
            provider_registry=self._provider_registry,
            source_registry=self._source_registry,
            service_registry=self._service_registry,
        )

        self._configuration_validator = ConfigurationValidator(
            backend_registry=self._backend_registry,
            provider_registry=self._provider_registry,
            source_registry=self._source_registry,
            service_registry=self._service_registry,
        )

        self._plugin_manager = None
        if enable_plugins:
            self._plugin_manager = PluginDiscoveryEngine(
                plugin_directories=plugin_directories,
                enable_directory_scan=True,
                enable_entry_points=True,
                enable_module_scan=True,
            )

            if auto_discover_plugins:
                self._discover_and_register_plugins()

        self._dependency_resolver = None
        # TODO: Implement dependency resolver if needed
        if enable_dependency_injection:
            # Dependency resolver would be implemented here
            pass

        self._error_handler = ErrorHandler()
        self._degradation_manager = GracefulDegradationManager()
        self._initializer = FactoryInitializer()

        # Initialize default components
        self._initialize_builtin_components()

        # Initialize service registry with built-in providers
        self._initialize_builtin_services()

    # Component Creation Methods

    def create_server(
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
            ComponentNotFoundError: If required components are not available
            ComponentCreationError: If component creation fails
        """
        effective_config = config or self._config

        if validate_config:
            validation_result = self.validate_configuration(effective_config)
            if not validation_result.is_valid:
                raise ConfigurationError(
                    f"Configuration validation failed: {validation_result.errors}"
                )

        # Create components with error handling
        try:
            self.create_backend(effective_config.backend)
            self.create_provider(effective_config.providers.embedding)

            if hasattr(effective_config, "data_sources") and effective_config.data_sources:
                # Create sources if configured
                sources = []
                for source_config in effective_config.data_sources.sources:
                    source = self.create_source(source_config)
                    sources.append(source)

            # Import at runtime to avoid circular import
            from codeweaver.server import CodeWeaverServer

            return CodeWeaverServer(config=effective_config)

        except Exception as e:
            logger.exception("Failed to create CodeWeaver server")
            raise ComponentCreationError(f"Server creation failed: {e}") from e

    def create_backend(self, config: BackendConfig) -> VectorBackend:
        """Create a vector backend instance.

        Args:
            config: Backend configuration

        Returns:
            Configured VectorBackend instance
        """
        return self._create_component("backend", config, self._backend_registry)

    def create_provider(self, config: Any) -> EmbeddingProvider:
        """Create an embedding provider instance.

        Args:
            config: Provider configuration

        Returns:
            Configured EmbeddingProvider instance
        """
        # Use existing provider factory for now
        provider_factory = ExistingProviderFactory()
        return provider_factory.create_embedding_provider(config)

    def create_source(self, config: SourceConfig | dict[str, Any]) -> DataSource:
        """Create a data source instance.

        Args:
            config: Source configuration

        Returns:
            Configured DataSource instance
        """
        if isinstance(config, dict):
            source_type = config.get("type")
            source_config = SourceConfig(config)
        else:
            source_type = getattr(config, "type", None)
            source_config = config

        if not source_type:
            raise ConfigurationError("Source configuration must specify 'type'")

        return self._source_registry.create_source(source_type, source_config)

    def create_service(self, service_type: ServiceType, config: Any) -> Any:
        """Create a service instance.

        Args:
            service_type: Type of service to create
            config: Service configuration

        Returns:
            Configured service instance
        """
        return self._service_registry.create_service(service_type, config)

    # Configuration and Validation Methods

    def validate_configuration(self, config: "CodeWeaverConfig") -> ValidationResult:
        """Validate a complete configuration.

        Args:
            config: Configuration to validate

        Returns:
            ValidationResult with validation details
        """
        return self._configuration_validator.validate_configuration(config)

    def validate_component_config(
        self, component_type: ComponentType, config: BaseComponentConfig
    ) -> ValidationResult:
        """Validate configuration for a specific component.

        Args:
            component_type: Type of component to validate
            config: Component configuration

        Returns:
            ValidationResult with validation details
        """
        registry = self._get_registry(component_type)
        return registry.validate_component(config.provider)

    # Information and Discovery Methods

    def get_available_components(self) -> dict[str, dict[str, Any]]:
        """Get information about all available components.

        Returns:
            Dictionary with component information organized by type
        """
        return {
            "backends": self._backend_registry.list_available_components(),
            "providers": self._provider_registry.get_available_embedding_providers(),
            "sources": self._source_registry.list_available_components(),
            "services": self._service_registry.list_providers(),
        }

    def get_component_capabilities(self, component_type: ComponentType, component_name: str) -> Any:
        """Get capabilities for a specific component.

        Args:
            component_type: Type of component
            component_name: Name of component

        Returns:
            Component capabilities
        """
        return self._capability_manager.get_component_capabilities(component_type, component_name)

    def supports_capability(
        self, component_type: ComponentType, component_name: str, capability: str
    ) -> bool:
        """Check if a component supports a specific capability.

        Args:
            component_type: Type of component
            component_name: Name of component
            capability: Capability to check

        Returns:
            True if component supports the capability
        """
        return self._capability_manager.supports_capability(
            component_type, component_name, capability
        )

    # Plugin Management Methods

    def discover_plugins(self) -> dict[ComponentType, list[str]]:
        """Discover available plugins.

        Returns:
            Dictionary of discovered plugins by component type
        """
        if not self._plugin_manager:
            return {
                ComponentType.BACKEND: [],
                ComponentType.PROVIDER: [],
                ComponentType.SOURCE: [],
                ComponentType.SERVICE: [],
            }

        discovered = self._plugin_manager.discover_all_plugins()
        return {
            component_type: [plugin.name for plugin in plugins]
            for component_type, plugins in discovered.items()
        }

    def register_plugin(self, plugin_info: PluginInfo) -> bool:
        """Register a custom plugin.

        Args:
            plugin_info: Plugin information and implementation

        Returns:
            True if registration successful
        """
        try:
            if plugin_info.component_type == ComponentType.BACKEND:
                return self._register_backend_plugin(plugin_info)
            if plugin_info.component_type == ComponentType.PROVIDER:
                return self._register_provider_plugin(plugin_info)
            if plugin_info.component_type == ComponentType.SOURCE:
                return self._register_source_plugin(plugin_info)
            if plugin_info.component_type == ComponentType.SERVICE:
                return self._register_service_plugin(plugin_info)
            logger.error("Unknown component type: %s", plugin_info.component_type)

        except Exception:
            logger.exception("Plugin registration failed for %s", plugin_info.name)

        else:
            return False

    # Private Implementation Methods

    def _create_component(self, component_type: str, config: Any, registry: Any) -> Any:
        """Unified component creation pattern."""
        provider_name = getattr(config, "provider", None)
        if not provider_name:
            raise ConfigurationError(f"Configuration must specify provider for {component_type}")

        if not registry.has_component(provider_name):
            raise ComponentNotFoundError(f"{component_type} '{provider_name}' not found")

        # Validate configuration
        validation_result = registry.validate_component(provider_name)
        if not validation_result.is_valid:
            raise ConfigurationError(
                f"Invalid {component_type} configuration: {validation_result.errors}"
            )

        # Resolve dependencies
        dependencies = {}
        if self._dependency_resolver:
            dependencies = self._dependency_resolver.resolve_dependencies(
                component_type, provider_name
            )

        # Create component
        try:
            if component_type == "backend":
                return registry.create_backend(config)
            if component_type == "source":
                return registry.create_source(provider_name, config)
            component_class = registry.get_component_class(provider_name)

        except Exception as e:
            logger.exception("Component creation failed")
            raise ComponentCreationError(
                f"Failed to create {component_type} '{provider_name}': {e}"
            ) from e

        else:
            return component_class(config=config, **dependencies)

    def _get_registry(self, component_type: ComponentType) -> Any:
        """Get registry for component type."""
        registries = {
            ComponentType.BACKEND: self._backend_registry,
            ComponentType.PROVIDER: self._provider_registry,
            ComponentType.SOURCE: self._source_registry,
            ComponentType.SERVICE: self._service_registry,
        }
        return registries[component_type]

    def _initialize_builtin_components(self) -> None:
        """Initialize built-in component registrations."""
        # Initialize backend registrations
        self._backend_registry.initialize_builtin_backends()

        # Provider registrations are already handled by existing A+ system

        # Initialize source registrations
        self._source_registry.initialize_builtin_sources()

    def _initialize_builtin_services(self) -> None:
        """Initialize built-in service provider registrations."""
        from codeweaver._types.service_data import ServiceCapabilities
        from codeweaver.services.providers.chunking import ChunkingService
        from codeweaver.services.providers.file_filtering import FilteringService

        # Register chunking service providers
        self._service_registry.register_provider(
            ServiceType.CHUNKING,
            "fastmcp_chunking",
            ChunkingService,
            ServiceCapabilities(
                supports_streaming=True,
                supports_batch=True,
                supports_async=True,
                max_concurrency=10,
                memory_usage="medium",
                performance_profile="standard",
            ),
        )

        # Register filtering service providers
        self._service_registry.register_provider(
            ServiceType.FILTERING,
            "fastmcp_filtering",
            FilteringService,
            ServiceCapabilities(
                supports_streaming=True,
                supports_batch=True,
                supports_async=True,
                max_concurrency=10,
                memory_usage="low",
                performance_profile="standard",
            ),
        )

        logger.info("Built-in service providers registered")

    def _discover_and_register_plugins(self) -> None:
        """Discover and register available plugins."""
        if not self._plugin_manager:
            return

        try:
            discovered_plugins = self._plugin_manager.discover_all_plugins()

            for component_type, plugins in discovered_plugins.items():
                for plugin_info in plugins:
                    if self.register_plugin(plugin_info):
                        logger.info(
                            "Registered plugin: %s (%s)", plugin_info.name, component_type.value
                        )
                    else:
                        logger.warning("Failed to register plugin: %s", plugin_info.name)

        except Exception as e:
            logger.warning("Plugin discovery failed: %s", e)

    def _register_backend_plugin(self, plugin_info: PluginInfo) -> bool:
        """Register a backend plugin."""
        try:
            backend_class = plugin_info.plugin_class.get_backend_class()
            result = self._backend_registry.register_component(
                name=plugin_info.name,
                component_class=backend_class,
                capabilities=plugin_info.capabilities,
                component_info=plugin_info.component_info,
            )
        except Exception:
            logger.exception("Backend plugin registration failed")
            return False

        else:
            return result.success

    def _register_provider_plugin(self, plugin_info: PluginInfo) -> bool:
        """Register a provider plugin."""
        # Provider plugin registration would go here
        # For now, return False as it's not implemented
        logger.warning("Provider plugin registration not yet implemented")
        return False

    def _register_source_plugin(self, plugin_info: PluginInfo) -> bool:
        """Register a source plugin."""
        try:
            source_class = plugin_info.plugin_class.get_source_class()
            result = self._source_registry.register_component(
                name=plugin_info.name,
                component_class=source_class,
                capabilities=plugin_info.capabilities,
                component_info=plugin_info.component_info,
            )
        except Exception:
            logger.exception("Source plugin registration failed")
            return False
        else:
            return result.success

    def _register_service_plugin(self, plugin_info: PluginInfo) -> bool:
        """Register a service plugin."""
        try:
            service_class = plugin_info.plugin_class.get_service_class()
            # Determine service type from plugin info
            service_type = getattr(plugin_info, "service_type", None)
            if not service_type:
                logger.error("Service plugin %s missing service_type", plugin_info.name)
                return False

            self._service_registry.register_provider(
                service_type=service_type,
                provider_name=plugin_info.name,
                provider_class=service_class,
                capabilities=plugin_info.capabilities,
            )

        except Exception:
            logger.exception("Service plugin registration failed")
            return False

        else:
            logger.info("Registered service plugin: %s (%s)", plugin_info.name, service_type.value)
            return True
