# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Unified component registry consolidating backend, source, and service registries.

Provides a single registry interface for all component types while maintaining
the specific functionality and patterns from the original separate registries.
"""

import logging

from datetime import UTC, datetime
from typing import Annotated, Any, ClassVar, TypeVar

from pydantic import Field

from codeweaver._types import (
    BackendCapabilities,
    BaseComponentInfo,
    ComponentRegistration,
    ComponentType,
    RegistrationResult,
    SourceCapabilities,
    ValidationResult,
)
from codeweaver._types.config import ServiceType
from codeweaver._types.service_data import (
    ProviderStatus,
    ServiceCapabilities,
    ServiceInstanceInfo,
    ServiceProviderInfo,
    ServiceRegistryHealth,
)
from codeweaver._types.service_exceptions import (
    DuplicateProviderError,
    ProviderNotFoundError,
    ProviderRegistrationError,
    ServiceNotFoundError,
)
from codeweaver._types.services import ServiceProvider
from codeweaver.backends.base import VectorBackend
from codeweaver.services.providers.base_provider import BaseServiceProvider
from codeweaver.sources.base import DataSource


logger = logging.getLogger(__name__)

T = TypeVar('T')


# Component-specific info classes
class BackendInfo(BaseComponentInfo):
    """Backend-specific information."""

    component_type: ComponentType = ComponentType.BACKEND

    backend_type: Annotated[
        str, Field(description="Type of vector backend (e.g., 'qdrant', 'pinecone')")
    ]
    connection_requirements: Annotated[
        dict[str, str], Field(default_factory=dict, description="Required connection parameters")
    ]
    optional_parameters: Annotated[
        dict[str, str], Field(default_factory=dict, description="Optional configuration parameters")
    ]


class SourceInfo(BaseComponentInfo):
    """Source-specific information."""

    component_type: ComponentType = ComponentType.SOURCE

    source_type: Annotated[
        str, Field(description="Type of data source (e.g., 'filesystem', 'database')")
    ]
    supported_schemes: Annotated[
        list[str],
        Field(default_factory=list, description="Supported URI schemes (e.g., ['file', 'http'])"),
    ]
    configuration_schema: Annotated[
        dict[str, Any],
        Field(default_factory=dict, description="JSON schema for source configuration"),
    ]


class ComponentRegistry:
    """Unified registry for all component types (backends, sources, services)."""

    # Class-level storage for different component types
    _backends: ClassVar[dict[str, ComponentRegistration[VectorBackend]]] = {}
    _sources: ClassVar[dict[str, ComponentRegistration[DataSource]]] = {}
    _services: ClassVar[dict[ServiceType, dict[str, type[ServiceProvider]]]] = {}

    # Service-specific storage (from original ServiceRegistry)
    _service_provider_info: ClassVar[dict[ServiceType, dict[str, ServiceProviderInfo]]] = {}
    _service_instances: ClassVar[dict[ServiceType, ServiceProvider | None]] = {}
    _service_instance_info: ClassVar[dict[ServiceType, ServiceInstanceInfo | None]] = {}

    _initialized: ClassVar[bool] = False
    _created_at: ClassVar[datetime] = datetime.now(UTC)

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize the component registry."""
        self._logger = logger or logging.getLogger(__name__)
        if not self._initialized:
            self._initialize_registries()

    @classmethod
    def _initialize_registries(cls) -> None:
        """Initialize all registry storage."""
        if not cls._initialized:
            cls._backends = {}
            cls._sources = {}
            cls._services = {}
            cls._service_provider_info = {}
            cls._service_instances = {}
            cls._service_instance_info = {}
            cls._initialized = True
            cls._created_at = datetime.now(UTC)

    # Backend Registry Methods
    def register_backend(
        self,
        name: str,
        backend_class: type[VectorBackend],
        capabilities: BackendCapabilities,
        backend_info: BackendInfo,
        *,
        validate: bool = True,
        check_availability: bool = True,
    ) -> RegistrationResult:
        """Register a vector backend."""
        if validate:
            validation_result = self._validate_backend_class(backend_class)
            if not validation_result.is_valid:
                return RegistrationResult(
                    success=False, component_name=name, errors=validation_result.errors
                )

        if check_availability:
            is_available, reason = self._check_backend_availability(backend_class)
            if not is_available:
                return RegistrationResult(
                    success=False,
                    component_name=name,
                    errors=[f"Backend not available: {reason}"]
                )

        # Create registration
        registration = ComponentRegistration(
            component_class=backend_class,
            capabilities=capabilities,
            component_info=backend_info,
            is_available=True,
            availability_reason=None
        )

        self._backends[name] = registration
        self._logger.info("Registered backend: %s", name)

        return RegistrationResult(success=True, component_name=name)

    def get_backend(self, name: str) -> ComponentRegistration[VectorBackend] | None:
        """Get a registered backend."""
        return self._backends.get(name)

    def list_backends(self) -> list[str]:
        """List all registered backend names."""
        return list(self._backends.keys())

    def is_backend_registered(self, name: str) -> bool:
        """Check if a backend is registered."""
        return name in self._backends

    # Source Registry Methods
    def register_source(
        self,
        name: str,
        source_class: type[DataSource],
        capabilities: SourceCapabilities,
        source_info: SourceInfo,
        *,
        validate: bool = True,
        check_availability: bool = True,
    ) -> RegistrationResult:
        """Register a data source."""
        if validate:
            validation_result = self._validate_source_class(source_class)
            if not validation_result.is_valid:
                return RegistrationResult(
                    success=False, component_name=name, errors=validation_result.errors
                )

        if check_availability:
            is_available, reason = self._check_source_availability(source_class)
            if not is_available:
                return RegistrationResult(
                    success=False,
                    component_name=name,
                    errors=[f"Source not available: {reason}"]
                )

        # Create registration
        registration = ComponentRegistration(
            component_class=source_class,
            capabilities=capabilities,
            component_info=source_info,
            is_available=True,
            availability_reason=None
        )

        self._sources[name] = registration
        self._logger.info(f"Registered source: {name}")

        return RegistrationResult(success=True, component_name=name)

    def get_source(self, name: str) -> ComponentRegistration[DataSource] | None:
        """Get a registered source."""
        return self._sources.get(name)

    def list_sources(self) -> list[str]:
        """List all registered source names."""
        return list(self._sources.keys())

    def is_source_registered(self, name: str) -> bool:
        """Check if a source is registered."""
        return name in self._sources

    # Service Registry Methods
    def register_service_provider(
        self,
        service_type: ServiceType,
        provider_name: str,
        provider_class: type[ServiceProvider],
        capabilities: ServiceCapabilities | None = None,
        configuration_schema: dict[str, Any] | None = None,
    ) -> None:
        """Register a service provider class."""
        try:
            # Initialize service type registry if needed
            if service_type not in self._services:
                self._services[service_type] = {}
                self._service_provider_info[service_type] = {}

            # Check for duplicates
            if provider_name in self._services[service_type]:
                raise DuplicateProviderError(service_type.value, provider_name)

            # Validate provider class
            if not issubclass(provider_class, BaseServiceProvider):
                raise ProviderRegistrationError(
                    f"Provider {provider_name} must inherit from BaseServiceProvider"
                )

            # Register provider
            self._services[service_type][provider_name] = provider_class

            # Store provider info
            provider_info = ServiceProviderInfo(
                name=provider_name,
                service_type=service_type,
                provider_class=provider_class.__name__,
                capabilities=capabilities or ServiceCapabilities(),
                configuration_schema=configuration_schema or {},
                status=ProviderStatus.REGISTERED,
                registered_at=datetime.now(UTC)
            )
            self._service_provider_info[service_type][provider_name] = provider_info

            self._logger.info(f"Registered service provider: {service_type.value}.{provider_name}")

        except Exception as e:
            raise ProviderRegistrationError(f"Failed to register provider {provider_name}: {e}") from e

    def get_service_provider_class(
        self, service_type: ServiceType, provider_name: str
    ) -> type[ServiceProvider]:
        """Get a registered service provider class."""
        if service_type not in self._services:
            raise ServiceNotFoundError(f"Service type {service_type.value} not found")

        if provider_name not in self._services[service_type]:
            raise ProviderNotFoundError(service_type.value, provider_name)

        return self._services[service_type][provider_name]

    def list_service_providers(self, service_type: ServiceType) -> list[str]:
        """List all registered providers for a service type."""
        return list(self._services.get(service_type, {}).keys())

    def is_service_provider_registered(self, service_type: ServiceType, provider_name: str) -> bool:
        """Check if a service provider is registered."""
        return (
            service_type in self._services and
            provider_name in self._services[service_type]
        )

    # Unified Registry Methods
    def get_component_count(self) -> dict[str, int]:
        """Get count of registered components by type."""
        return {
            "backends": len(self._backends),
            "sources": len(self._sources),
            "services": sum(len(providers) for providers in self._services.values())
        }

    def clear_registry(self, component_type: ComponentType | None = None) -> None:
        """Clear registry (mainly for testing)."""
        if component_type is None:
            # Clear all
            self._backends.clear()
            self._sources.clear()
            self._services.clear()
            self._service_provider_info.clear()
            self._service_instances.clear()
            self._service_instance_info.clear()
            self._logger.debug("Cleared all registries")
        elif component_type == ComponentType.BACKEND:
            self._backends.clear()
            self._logger.debug("Cleared backend registry")
        elif component_type == ComponentType.SOURCE:
            self._sources.clear()
            self._logger.debug("Cleared source registry")
        elif component_type == ComponentType.SERVICE:
            self._services.clear()
            self._service_provider_info.clear()
            self._service_instances.clear()
            self._service_instance_info.clear()
            self._logger.debug("Cleared service registry")

    def get_registry_health(self) -> ServiceRegistryHealth:
        """Get overall registry health status."""
        total_providers = sum(len(providers) for providers in self._services.values())
        healthy_providers = 0

        for providers in self._service_provider_info.values():
            for provider_info in providers.values():
                if provider_info.status == ProviderStatus.HEALTHY:
                    healthy_providers += 1

        return ServiceRegistryHealth(
            total_providers=total_providers,
            healthy_providers=healthy_providers,
            service_types=list(self._services.keys()),
            last_check=datetime.now(UTC),
            uptime_seconds=(datetime.now(UTC) - self._created_at).total_seconds()
        )

    # Validation methods
    def _validate_backend_class(self, backend_class: type[VectorBackend]) -> ValidationResult:
        """Validate backend class implementation."""
        errors = []
        warnings = []

        # Check if it's a subclass of VectorBackend
        if not issubclass(backend_class, VectorBackend):
            errors.append("Backend class must inherit from VectorBackend")

        # Check for required methods
        required_methods = ['store_vectors', 'search_vectors', 'delete_vectors']
        for method in required_methods:
            if not hasattr(backend_class, method):
                errors.append(f"Backend class missing required method: {method}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def _validate_source_class(self, source_class: type[DataSource]) -> ValidationResult:
        """Validate source class implementation."""
        errors = []
        warnings = []

        # Check if it's a subclass of DataSource
        if not issubclass(source_class, DataSource):
            errors.append("Source class must inherit from DataSource")

        # Check for required methods
        required_methods = ['discover_content', 'get_content']
        errors.extend(
            f"Source class missing required method: {method}"
            for method in required_methods
            if not hasattr(source_class, method)
        )
        return ValidationResult(is_valid=not errors, errors=errors, warnings=warnings)

    def _check_backend_availability(self, backend_class: type[VectorBackend]) -> tuple[bool, str | None]:
        """Check if backend is available."""
        try:
            # Try to call check_availability if it exists
            if hasattr(backend_class, 'check_availability'):
                return backend_class.check_availability()
            # Default to available if no check method
            return True, None
        except Exception as e:
            return False, str(e)

    def _check_source_availability(self, source_class: type[DataSource]) -> tuple[bool, str | None]:
        """Check if source is available."""
        try:
            # Try to call check_availability if it exists
            if hasattr(source_class, 'check_availability'):
                return source_class.check_availability()
            # Default to available if no check method
            return True, None
        except Exception as e:
            return False, str(e)


# Global registry instance
_global_registry: ComponentRegistry | None = None


def get_global_registry() -> ComponentRegistry:
    """Get the global component registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ComponentRegistry()
    return _global_registry


def reset_global_registry() -> None:
    """Reset the global registry (mainly for testing)."""
    global _global_registry
    _global_registry = None
