<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# CodeWeaver Factories API Specification

**Document Version**: 1.0
**Date**: 2025-01-25
**Status**: API Design Specification

## Executive Summary

This document provides the complete API specification for CodeWeaver's unified factory system, including detailed interface definitions, configuration models, error handling patterns, and implementation guidelines. All APIs follow the A+ providers module pattern with comprehensive type safety and plugin extensibility.

---

## 🏗️ Core API Interfaces

### 1. Universal Component Protocol

```python
from typing import Any, Protocol, runtime_checkable
from abc import abstractmethod
from pydantic import BaseModel

@runtime_checkable
class ComponentRegistry(Protocol):
    """Universal protocol for all component registries."""

    @abstractmethod
    def register_component(
        self,
        name: str,
        component_class: type,
        capabilities: "BaseCapabilities",
        component_info: "BaseComponentInfo",
        *,
        validate: bool = True,
        check_availability: bool = True
    ) -> "RegistrationResult":
        """Register a component with the registry."""
        ...

    @abstractmethod
    def get_component_class(self, name: str) -> type:
        """Get the component class for a registered component."""
        ...

    @abstractmethod
    def get_capabilities(self, name: str) -> "BaseCapabilities":
        """Get capabilities for a registered component."""
        ...

    @abstractmethod
    def get_component_info(self, name: str) -> "BaseComponentInfo":
        """Get detailed information about a component."""
        ...

    @abstractmethod
    def list_available_components(self) -> dict[str, "BaseComponentInfo"]:
        """List all available components and their information."""
        ...

    @abstractmethod
    def has_component(self, name: str) -> bool:
        """Check if a component is registered."""
        ...

    @abstractmethod
    def validate_component(self, name: str) -> "ValidationResult":
        """Validate a registered component."""
        ...
```

### 2. Base Configuration Models

```python
from typing import Annotated, Any, ClassVar, Generic, TypeVar
from pydantic import BaseModel, ConfigDict, Field, field_validator
from enum import Enum

class ComponentType(Enum):
    """Types of components in the system."""
    BACKEND = "backend"
    PROVIDER = "provider"
    SOURCE = "source"

class BaseCapabilities(BaseModel):
    """Base capabilities model for all components."""
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        frozen=False
    )

    supported_capabilities: list[str] = Field(
        default_factory=list,
        description="List of capabilities this component supports"
    )

    def supports(self, capability: str) -> bool:
        """Check if this component supports a specific capability."""
        return capability in self.supported_capabilities

    def supports_all(self, capabilities: list[str]) -> bool:
        """Check if this component supports all specified capabilities."""
        return all(cap in self.supported_capabilities for cap in capabilities)

    def supports_any(self, capabilities: list[str]) -> bool:
        """Check if this component supports any of the specified capabilities."""
        return any(cap in self.supported_capabilities for cap in capabilities)

class BaseComponentInfo(BaseModel):
    """Base information model for all components."""
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True
    )

    name: Annotated[str, Field(description="Component name")]
    display_name: Annotated[str, Field(description="Human-readable display name")]
    description: Annotated[str, Field(description="Component description")]
    component_type: Annotated[ComponentType, Field(description="Type of component")]
    version: Annotated[str, Field(default="1.0.0", description="Component version")]
    author: Annotated[str | None, Field(default=None, description="Component author")]
    license: Annotated[str | None, Field(default=None, description="Component license")]
    documentation_url: Annotated[str | None, Field(default=None, description="Documentation URL")]
    source_url: Annotated[str | None, Field(default=None, description="Source code URL")]
    implemented: Annotated[bool, Field(default=True, description="Whether component is fully implemented")]

class BaseComponentConfig(BaseModel):
    """Base configuration for all components."""
    model_config = ConfigDict(
        extra="allow",  # Allow component-specific extensions
        validate_assignment=True
    )

    # Core identification
    component_type: Annotated[ComponentType, Field(description="Type of component")]
    provider: Annotated[str, Field(description="Provider/implementation name")]
    name: Annotated[str | None, Field(default=None, description="Optional component instance name")]

    # Control settings
    enabled: Annotated[bool, Field(default=True, description="Whether component is enabled")]

    # Plugin support
    custom_class: Annotated[str | None, Field(default=None, description="Custom implementation class path")]
    custom_module: Annotated[str | None, Field(default=None, description="Custom implementation module")]

    # Validation settings
    validate_on_creation: Annotated[bool, Field(default=True, description="Validate component on creation")]
    fail_fast: Annotated[bool, Field(default=True, description="Fail immediately on validation errors")]
```

### 3. Factory Result Models

```python
from pydantic.dataclasses import dataclass
from typing import Any

@dataclass
class RegistrationResult:
    """Result of component registration."""
    success: bool
    component_name: str | None = None
    errors: list[str] = None
    warnings: list[str] = None
    metadata: dict[str, Any] = None

    @property
    def has_errors(self) -> bool:
        return self.errors is not None and len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        return self.warnings is not None and len(self.warnings) > 0

@dataclass
class ValidationResult:
    """Result of component validation."""
    is_valid: bool
    errors: list[str] = None
    warnings: list[str] = None
    metadata: dict[str, Any] = None

    @property
    def has_errors(self) -> bool:
        return self.errors is not None and len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        return self.warnings is not None and len(self.warnings) > 0

@dataclass
class CreationResult:
    """Result of component creation."""
    success: bool
    component: Any | None = None
    errors: list[str] = None
    warnings: list[str] = None
    creation_time_ms: float | None = None

    @property
    def has_errors(self) -> bool:
        return self.errors is not None and len(self.errors) > 0
```

### 4. Component Registration Pattern

```python
from typing import Generic, TypeVar
from pydantic.dataclasses import dataclass, field

T = TypeVar('T')

@dataclass
class ComponentRegistration(Generic[T]):
    """Universal component registration following providers pattern."""
    component_class: type[T]
    capabilities: BaseCapabilities
    component_info: BaseComponentInfo
    is_available: bool = True
    unavailable_reason: str | None = None
    metadata: dict[str, Any] = Fielddefault_factory=dict)
    registration_time: float = Fielddefault_factory=lambda: time.time())

    @property
    def is_usable(self) -> bool:
        """Check if component is usable (available and implemented)."""
        return self.is_available and self.component_info.implemented
```

---

## 🔧 Specific Component APIs

### 1. Backend Factory API

```python
from codeweaver.backends.base import VectorBackend
from codeweaver.backends.config import BackendConfig

class BackendCapabilities(BaseCapabilities):
    """Backend-specific capabilities."""

    # Vector search capabilities
    supports_vector_search: bool = True
    supports_hybrid_search: bool = False
    supports_metadata_filtering: bool = True
    supports_namespace_isolation: bool = False

    # Performance capabilities
    supports_batch_operations: bool = True
    supports_concurrent_access: bool = True
    max_batch_size: int | None = None
    max_vector_dimensions: int | None = None

    # Storage capabilities
    supports_persistence: bool = True
    supports_backup_restore: bool = False
    supports_replication: bool = False

class BackendInfo(BaseComponentInfo):
    """Backend-specific information."""
    component_type: ComponentType = ComponentType.BACKEND

    # Backend-specific fields
    backend_type: str = Field(description="Type of vector backend (e.g., 'qdrant', 'pinecone')")
    connection_requirements: dict[str, str] = Field(
        default_factory=dict,
        description="Required connection parameters"
    )
    optional_parameters: dict[str, str] = Field(
        default_factory=dict,
        description="Optional configuration parameters"
    )

class BackendRegistry(ComponentRegistry):
    """Registry for vector backends following providers A+ pattern."""

    _backends: ClassVar[dict[str, ComponentRegistration[VectorBackend]]] = {}
    _initialized: ClassVar[bool] = False

    @classmethod
    def register_backend(
        cls,
        name: str,
        backend_class: type[VectorBackend],
        capabilities: BackendCapabilities,
        backend_info: BackendInfo,
        *,
        validate: bool = True,
        check_availability: bool = True
    ) -> RegistrationResult:
        """Register a vector backend."""

        if validate:
            validation_result = cls._validate_backend_class(backend_class)
            if not validation_result.is_valid:
                return RegistrationResult(
                    success=False,
                    component_name=name,
                    errors=validation_result.errors
                )

        if check_availability:
            is_available, reason = cls._check_backend_availability(backend_class)
        else:
            is_available, reason = True, None

        registration = ComponentRegistration(
            component_class=backend_class,
            capabilities=capabilities,
            component_info=backend_info,
            is_available=is_available,
            unavailable_reason=reason
        )

        cls._backends[name] = registration

        if is_available:
            logger.info("Registered backend: %s", name)
        else:
            logger.warning("Registered backend %s (unavailable: %s)", name, reason)

        return RegistrationResult(success=True, component_name=name)

    @classmethod
    def create_backend(cls, config: BackendConfig) -> VectorBackend:
        """Create a backend instance."""
        if config.provider not in cls._backends:
            raise ComponentNotFoundError(f"Backend '{config.provider}' not registered")

        registration = cls._backends[config.provider]
        if not registration.is_usable:
            raise ComponentUnavailableError(
                f"Backend '{config.provider}' is not usable: {registration.unavailable_reason}"
            )

        return registration.component_class(config)

    @classmethod
    def get_capabilities(cls, name: str) -> BackendCapabilities:
        """Get backend capabilities."""
        if name not in cls._backends:
            raise ComponentNotFoundError(f"Backend '{name}' not registered")
        return cls._backends[name].capabilities
```

### 2. Source Factory API

```python
from codeweaver.sources.base import DataSource
from codeweaver.sources.config import SourceConfig

class SourceCapabilities(BaseCapabilities):
    """Source-specific capabilities."""

    # Content discovery
    supports_content_discovery: bool = True
    supports_recursive_discovery: bool = True
    supports_pattern_filtering: bool = True
    supports_metadata_extraction: bool = False

    # Content reading
    supports_content_reading: bool = True
    supports_streaming_read: bool = False
    supports_incremental_updates: bool = False
    max_file_size_mb: int | None = None

    # Data processing
    supports_chunking: bool = False
    supports_preprocessing: bool = False
    supported_file_types: list[str] = Field(default_factory=list)

class SourceInfo(BaseComponentInfo):
    """Source-specific information."""
    component_type: ComponentType = ComponentType.SOURCE

    # Source-specific fields
    source_type: str = Field(description="Type of data source (e.g., 'filesystem', 'database')")
    supported_schemes: list[str] = Field(
        default_factory=list,
        description="Supported URI schemes (e.g., ['file', 'http'])"
    )
    configuration_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON schema for source configuration"
    )

class SourceRegistry(ComponentRegistry):
    """Registry for data sources following providers A+ pattern."""

    _sources: ClassVar[dict[str, ComponentRegistration[DataSource]]] = {}
    _initialized: ClassVar[bool] = False

    @classmethod
    def register_source(
        cls,
        name: str,
        source_class: type[DataSource],
        capabilities: SourceCapabilities,
        source_info: SourceInfo,
        *,
        validate: bool = True,
        check_availability: bool = True
    ) -> RegistrationResult:
        """Register a data source."""

        if validate:
            validation_result = cls._validate_source_class(source_class)
            if not validation_result.is_valid:
                return RegistrationResult(
                    success=False,
                    component_name=name,
                    errors=validation_result.errors
                )

        if check_availability:
            is_available, reason = cls._check_source_availability(source_class)
        else:
            is_available, reason = True, None

        registration = ComponentRegistration(
            component_class=source_class,
            capabilities=capabilities,
            component_info=source_info,
            is_available=is_available,
            unavailable_reason=reason
        )

        cls._sources[name] = registration

        if is_available:
            logger.info("Registered source: %s", name)
        else:
            logger.warning("Registered source %s (unavailable: %s)", name, reason)

        return RegistrationResult(success=True, component_name=name)
```

---

## 🔌 Plugin System APIs

### 1. Plugin Interface Protocol

```python
@runtime_checkable
class PluginInterface(Protocol):
    """Universal interface that all plugins must implement."""

    @classmethod
    def get_plugin_name(cls) -> str:
        """Get the unique name for this plugin."""
        ...

    @classmethod
    def get_component_type(cls) -> ComponentType:
        """Get the type of component this plugin provides."""
        ...

    @classmethod
    def get_capabilities(cls) -> BaseCapabilities:
        """Get the capabilities this plugin provides."""
        ...

    @classmethod
    def get_component_info(cls) -> BaseComponentInfo:
        """Get detailed information about this plugin."""
        ...

    @classmethod
    def validate_config(cls, config: BaseComponentConfig) -> ValidationResult:
        """Validate configuration for this plugin."""
        ...

    @classmethod
    def get_dependencies(cls) -> list[str]:
        """Get list of required dependencies for this plugin."""
        ...

class BackendPlugin(PluginInterface):
    """Specific interface for backend plugins."""

    @classmethod
    def get_component_type(cls) -> ComponentType:
        return ComponentType.BACKEND

    @classmethod
    def get_backend_class(cls) -> type[VectorBackend]:
        """Get the backend implementation class."""
        ...

class ProviderPlugin(PluginInterface):
    """Specific interface for provider plugins."""

    @classmethod
    def get_component_type(cls) -> ComponentType:
        return ComponentType.PROVIDER

    @classmethod
    def get_provider_class(cls) -> type[EmbeddingProvider | RerankProvider]:
        """Get the provider implementation class."""
        ...

class SourcePlugin(PluginInterface):
    """Specific interface for source plugins."""

    @classmethod
    def get_component_type(cls) -> ComponentType:
        return ComponentType.SOURCE

    @classmethod
    def get_source_class(cls) -> type[DataSource]:
        """Get the source implementation class."""
        ...
```

### 2. Plugin Discovery API

```python
from pydantic.dataclasses import dataclass
from pathlib import Path

@dataclass
class PluginInfo:
    """Information about a discovered plugin."""
    name: str
    component_type: ComponentType
    plugin_class: type
    capabilities: BaseCapabilities
    component_info: BaseComponentInfo
    entry_point: str | None = None
    file_path: str | None = None
    module_name: str | None = None
    metadata: dict[str, Any] = Fielddefault_factory=dict)

class PluginDiscovery:
    """Plugin discovery system with multiple discovery methods."""

    def __init__(
        self,
        plugin_directories: list[str] | None = None,
        enable_entry_points: bool = True,
        enable_directory_scan: bool = True,
        enable_module_scan: bool = True
    ):
        self.plugin_directories = plugin_directories or []
        self.enable_entry_points = enable_entry_points
        self.enable_directory_scan = enable_directory_scan
        self.enable_module_scan = enable_module_scan

    def discover_all_plugins(self) -> dict[ComponentType, list[PluginInfo]]:
        """Discover all available plugins."""
        discovered = {
            ComponentType.BACKEND: [],
            ComponentType.PROVIDER: [],
            ComponentType.SOURCE: []
        }

        if self.enable_entry_points:
            entry_point_plugins = self._discover_entry_point_plugins()
            self._merge_plugin_discoveries(discovered, entry_point_plugins)

        if self.enable_directory_scan:
            directory_plugins = self._discover_directory_plugins()
            self._merge_plugin_discoveries(discovered, directory_plugins)

        if self.enable_module_scan:
            module_plugins = self._discover_module_plugins()
            self._merge_plugin_discoveries(discovered, module_plugins)

        return discovered

    def discover_plugins_for_type(self, component_type: ComponentType) -> list[PluginInfo]:
        """Discover plugins for a specific component type."""
        all_plugins = self.discover_all_plugins()
        return all_plugins.get(component_type, [])

    def validate_plugin(self, plugin_info: PluginInfo) -> ValidationResult:
        """Validate a discovered plugin."""
        try:
            # Check plugin interface compliance
            if not isinstance(plugin_info.plugin_class, type):
                return ValidationResult(
                    is_valid=False,
                    errors=["Plugin class must be a type"]
                )

            # Check protocol compliance
            if not hasattr(plugin_info.plugin_class, 'get_plugin_name'):
                return ValidationResult(
                    is_valid=False,
                    errors=["Plugin must implement PluginInterface protocol"]
                )

            # Validate plugin-specific requirements
            config_validation = plugin_info.plugin_class.validate_config(
                BaseComponentConfig(
                    component_type=plugin_info.component_type,
                    provider=plugin_info.name
                )
            )

            if not config_validation.is_valid:
                return config_validation

            return ValidationResult(is_valid=True)

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Plugin validation failed: {e}"]
            )
```

### 3. Plugin Manager API

```python
class PluginManager:
    """Central plugin management system."""

    def __init__(
        self,
        plugin_discovery: PluginDiscovery | None = None,
        auto_discover: bool = True,
        auto_register: bool = True
    ):
        self._discovery = plugin_discovery or PluginDiscovery()
        self._discovered_plugins: dict[ComponentType, list[PluginInfo]] = {}
        self._registered_plugins: dict[str, PluginInfo] = {}

        if auto_discover:
            self.discover_plugins()

        if auto_register:
            self.register_discovered_plugins()

    def discover_plugins(self) -> dict[ComponentType, list[PluginInfo]]:
        """Discover all available plugins."""
        self._discovered_plugins = self._discovery.discover_all_plugins()
        return self._discovered_plugins

    def register_discovered_plugins(self) -> dict[str, RegistrationResult]:
        """Register all discovered plugins with their respective registries."""
        registration_results = {}

        for component_type, plugins in self._discovered_plugins.items():
            for plugin_info in plugins:
                result = self.register_plugin(plugin_info)
                registration_results[plugin_info.name] = result

        return registration_results

    def register_plugin(self, plugin_info: PluginInfo) -> RegistrationResult:
        """Register a single plugin."""
        try:
            # Validate plugin
            validation_result = self._discovery.validate_plugin(plugin_info)
            if not validation_result.is_valid:
                return RegistrationResult(
                    success=False,
                    component_name=plugin_info.name,
                    errors=validation_result.errors
                )

            # Register with appropriate registry
            if plugin_info.component_type == ComponentType.BACKEND:
                return self._register_backend_plugin(plugin_info)
            elif plugin_info.component_type == ComponentType.PROVIDER:
                return self._register_provider_plugin(plugin_info)
            elif plugin_info.component_type == ComponentType.SOURCE:
                return self._register_source_plugin(plugin_info)
            else:
                return RegistrationResult(
                    success=False,
                    component_name=plugin_info.name,
                    errors=[f"Unknown component type: {plugin_info.component_type}"]
                )

        except Exception as e:
            return RegistrationResult(
                success=False,
                component_name=plugin_info.name,
                errors=[f"Plugin registration failed: {e}"]
            )

    def list_discovered_plugins(self) -> dict[ComponentType, list[str]]:
        """List all discovered plugins by type."""
        return {
            component_type: [plugin.name for plugin in plugins]
            for component_type, plugins in self._discovered_plugins.items()
        }

    def list_registered_plugins(self) -> list[str]:
        """List all successfully registered plugins."""
        return list(self._registered_plugins.keys())

    def get_plugin_info(self, plugin_name: str) -> PluginInfo | None:
        """Get information about a specific plugin."""
        return self._registered_plugins.get(plugin_name)
```

---

## 🎛️ Main Factory API

### 1. CodeWeaver Factory

```python
from typing import overload

class CodeWeaverFactory:
    """
    Main factory orchestrator for all CodeWeaver components.

    Provides unified creation, configuration, and lifecycle management
    for backends, providers, and sources with plugin extensibility.
    """

    def __init__(
        self,
        config: CodeWeaverConfig | None = None,
        enable_plugins: bool = True,
        enable_dependency_injection: bool = True,
        plugin_directories: list[str] | None = None,
        auto_discover_plugins: bool = True
    ):
        """Initialize the CodeWeaver factory.

        Args:
            config: Optional factory configuration
            enable_plugins: Whether to enable plugin discovery and loading
            enable_dependency_injection: Whether to enable dependency injection
            plugin_directories: Custom plugin directories to scan
            auto_discover_plugins: Whether to automatically discover plugins on initialization
        """
        self._config = config or CodeWeaverConfig()

        # Initialize registries
        self._backend_registry = BackendRegistry()
        self._provider_registry = ProviderRegistry()  # Existing A+ implementation
        self._source_registry = SourceRegistry()

        # Initialize supporting systems
        self._capability_manager = CapabilityManager(
            backend_registry=self._backend_registry,
            provider_registry=self._provider_registry,
            source_registry=self._source_registry
        )

        self._plugin_manager = None
        if enable_plugins:
            self._plugin_manager = PluginManager(
                plugin_discovery=PluginDiscovery(
                    plugin_directories=plugin_directories
                ),
                auto_discover=auto_discover_plugins,
                auto_register=auto_discover_plugins
            )

        self._dependency_resolver = None
        if enable_dependency_injection:
            self._dependency_resolver = DependencyResolver()

        # Initialize default components
        self._initialize_builtin_components()

    # Component Creation Methods

    def create_server(
        self,
        config: CodeWeaverConfig | None = None,
        validate_config: bool = True
    ) -> "CodeEmbeddingsServer":
        """Create a complete CodeWeaver server instance.

        Args:
            config: Server configuration (uses factory config if None)
            validate_config: Whether to validate configuration before creation

        Returns:
            Configured CodeEmbeddingsServer instance

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
            backend = self.create_backend(effective_config.backend)
            provider = self.create_provider(effective_config.providers)
            sources = [
                self.create_source(source_config)
                for source_config in effective_config.sources
            ]

            return CodeEmbeddingsServer(
                backend=backend,
                provider=provider,
                sources=sources
            )

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

    def create_provider(self, config: ProviderConfig) -> EmbeddingProvider:
        """Create an embedding provider instance.

        Args:
            config: Provider configuration

        Returns:
            Configured EmbeddingProvider instance
        """
        return self._create_component("provider", config, self._provider_registry)

    def create_source(self, config: SourceConfig) -> DataSource:
        """Create a data source instance.

        Args:
            config: Source configuration

        Returns:
            Configured DataSource instance
        """
        return self._create_component("source", config, self._source_registry)

    # Configuration and Validation Methods

    def validate_configuration(self, config: CodeWeaverConfig) -> ValidationResult:
        """Validate a complete configuration.

        Args:
            config: Configuration to validate

        Returns:
            ValidationResult with validation details
        """
        validator = ConfigurationValidator(
            backend_registry=self._backend_registry,
            provider_registry=self._provider_registry,
            source_registry=self._source_registry
        )
        return validator.validate_configuration(config)

    def validate_component_config(
        self,
        component_type: ComponentType,
        config: BaseComponentConfig
    ) -> ValidationResult:
        """Validate configuration for a specific component.

        Args:
            component_type: Type of component to validate
            config: Component configuration

        Returns:
            ValidationResult with validation details
        """
        registry = self._get_registry(component_type)
        return registry.validate_component_config(config)

    # Information and Discovery Methods

    def get_available_components(self) -> dict[str, dict[str, BaseComponentInfo]]:
        """Get information about all available components.

        Returns:
            Dictionary with component information organized by type
        """
        return {
            "backends": self._backend_registry.list_available_components(),
            "providers": self._provider_registry.list_available_components(),
            "sources": self._source_registry.list_available_components()
        }

    def get_component_capabilities(
        self,
        component_type: ComponentType,
        component_name: str
    ) -> BaseCapabilities:
        """Get capabilities for a specific component.

        Args:
            component_type: Type of component
            component_name: Name of component

        Returns:
            Component capabilities
        """
        return self._capability_manager.get_component_capabilities(
            component_type, component_name
        )

    def supports_capability(
        self,
        component_type: ComponentType,
        component_name: str,
        capability: str
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
            raise PluginError("Plugin system not enabled")

        discovered = self._plugin_manager.discover_plugins()
        return {
            component_type: [plugin.name for plugin in plugins]
            for component_type, plugins in discovered.items()
        }

    def register_plugin(self, plugin_info: PluginInfo) -> RegistrationResult:
        """Register a custom plugin.

        Args:
            plugin_info: Plugin information and implementation

        Returns:
            Registration result
        """
        if not self._plugin_manager:
            raise PluginError("Plugin system not enabled")

        return self._plugin_manager.register_plugin(plugin_info)

    def list_registered_plugins(self) -> list[str]:
        """List all registered plugins.

        Returns:
            List of registered plugin names
        """
        if not self._plugin_manager:
            return []

        return self._plugin_manager.list_registered_plugins()

    # Private Implementation Methods

    def _create_component(
        self,
        component_type: str,
        config: BaseComponentConfig,
        registry: ComponentRegistry
    ) -> Any:
        """Unified component creation pattern."""
        if not registry.has_component(config.provider):
            raise ComponentNotFoundError(
                f"{component_type} '{config.provider}' not found"
            )

        # Validate configuration
        validation_result = registry.validate_component_config(config)
        if not validation_result.is_valid:
            raise ConfigurationError(
                f"Invalid {component_type} configuration: {validation_result.errors}"
            )

        # Resolve dependencies
        dependencies = {}
        if self._dependency_resolver:
            dependencies = self._dependency_resolver.resolve_dependencies(
                component_type, config.provider
            )

        # Create component
        try:
            component_class = registry.get_component_class(config.provider)
            return component_class(config=config, **dependencies)
        except Exception as e:
            logger.exception("Component creation failed")
            raise ComponentCreationError(
                f"Failed to create {component_type} '{config.provider}': {e}"
            ) from e

    def _get_registry(self, component_type: ComponentType) -> ComponentRegistry:
        """Get registry for component type."""
        registries = {
            ComponentType.BACKEND: self._backend_registry,
            ComponentType.PROVIDER: self._provider_registry,
            ComponentType.SOURCE: self._source_registry
        }
        return registries[component_type]

    def _initialize_builtin_components(self) -> None:
        """Initialize built-in component registrations."""
        # Backend registrations
        self._register_builtin_backends()

        # Provider registrations (already handled by existing A+ system)

        # Source registrations
        self._register_builtin_sources()
```

---

## 🚨 Error Handling API

### Exception Hierarchy

```python
class CodeWeaverFactoryError(Exception):
    """Base exception for factory-related errors."""
    pass

class ConfigurationError(CodeWeaverFactoryError):
    """Configuration validation or processing error."""
    pass

class ComponentNotFoundError(CodeWeaverFactoryError):
    """Requested component not found in registry."""
    pass

class ComponentUnavailableError(CodeWeaverFactoryError):
    """Component found but not available for use."""
    pass

class ComponentCreationError(CodeWeaverFactoryError):
    """Error during component creation."""
    pass

class PluginError(CodeWeaverFactoryError):
    """Plugin-related error."""
    pass

class RegistrationError(CodeWeaverFactoryError):
    """Component registration error."""
    pass

class ValidationError(CodeWeaverFactoryError):
    """Validation failure error."""
    pass
```

---

## 📋 Usage Examples

### Basic Factory Usage

```python
# Initialize factory with default configuration
factory = CodeWeaverFactory()

# Create server with configuration
config = CodeWeaverConfig(
    backend=BackendConfig(provider="qdrant", url="http://localhost:6333"),
    providers=ProviderConfig(provider="voyage-ai", api_key="your-key"),
    sources=[SourceConfig(type="filesystem", path="/path/to/code")]
)

server = factory.create_server(config)
```

### Plugin Development Example

```python
class CustomBackendPlugin(BackendPlugin):
    """Example custom backend plugin."""

    @classmethod
    def get_plugin_name(cls) -> str:
        return "custom-vector-db"

    @classmethod
    def get_capabilities(cls) -> BackendCapabilities:
        return BackendCapabilities(
            supports_vector_search=True,
            supports_hybrid_search=True,
            supports_metadata_filtering=True
        )

    @classmethod
    def get_component_info(cls) -> BackendInfo:
        return BackendInfo(
            name="custom-vector-db",
            display_name="Custom Vector Database",
            description="Custom vector database implementation",
            backend_type="custom"
        )

    @classmethod
    def get_backend_class(cls) -> type[VectorBackend]:
        return CustomVectorBackend
```

This comprehensive API specification provides all the interfaces, models, and patterns needed to implement the unified CodeWeaver factory system with full plugin extensibility and consistent patterns across all component types.
