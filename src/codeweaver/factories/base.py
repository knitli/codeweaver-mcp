# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Base factory patterns and protocols for the CodeWeaver factory system.

Provides foundational patterns, interfaces, and protocols that serve as the
foundation for the consolidated factory architecture. This module establishes
the core patterns used by registry and factory implementations.
"""

import logging

from abc import ABC, abstractmethod
from typing import Any, Protocol, TypeVar, runtime_checkable

from codeweaver.types import (
    BaseComponentConfig,
    BaseComponentInfo,
    ComponentType,
    PluginInfo,
    ValidationResult,
)


logger = logging.getLogger(__name__)

# Type variables for generic factory patterns
T = TypeVar("T")
ConfigT = TypeVar("ConfigT", bound=BaseComponentConfig)
InfoT = TypeVar("InfoT", bound=BaseComponentInfo)


@runtime_checkable
class ComponentFactory(Protocol[T, ConfigT]):
    """Protocol for component factory implementations."""

    @abstractmethod
    async def create_component(self, config: ConfigT, context: dict[str, Any]) -> T:
        """Create a component instance with the given configuration and context."""
        ...

    @abstractmethod
    def validate_config(self, config: ConfigT) -> ValidationResult:
        """Validate component configuration."""
        ...

    @abstractmethod
    def get_component_info(self) -> InfoT:
        """Get information about this component factory."""
        ...


@runtime_checkable
class RegistryProtocol(Protocol[T]):
    """Protocol for component registry implementations."""

    @abstractmethod
    def register_component(self, name: str, factory: ComponentFactory[T, Any]) -> None:
        """Register a component factory."""
        ...

    @abstractmethod
    def get_component_factory(self, name: str) -> ComponentFactory[T, Any] | None:
        """Get a registered component factory by name."""
        ...

    @abstractmethod
    def list_components(self) -> list[str]:
        """List all registered component names."""
        ...

    @abstractmethod
    def is_registered(self, name: str) -> bool:
        """Check if a component is registered."""
        ...


class BaseComponentFactory(ABC):
    """Base class for component factories with common functionality."""

    def __init__(self, component_type: ComponentType):
        """Initialize the factory with a component type."""
        self.component_type = component_type
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def create_component(self, config: BaseComponentConfig, context: dict[str, Any]) -> Any:
        """Create a component instance."""
        ...

    def validate_config(self, config: BaseComponentConfig) -> ValidationResult:
        """Default configuration validation."""
        try:
            # Basic validation - subclasses should override for specific validation
            if not hasattr(config, "model_validate"):
                return ValidationResult(
                    is_valid=False, errors=["Configuration must be a Pydantic model"], warnings=[]
                )

            # Validate the model
            config.model_validate(config.model_dump())
            return ValidationResult(is_valid=True, errors=[], warnings=[])

        except Exception as e:
            return ValidationResult(
                is_valid=False, errors=[f"Configuration validation failed: {e}"], warnings=[]
            )

    def get_component_info(self) -> BaseComponentInfo:
        """Get basic component information."""
        return BaseComponentInfo(
            name=self.__class__.__name__,
            component_type=self.component_type,
            description=self.__doc__ or "No description available",
        )


class BaseRegistry(ABC):
    """Base class for component registries with common functionality."""

    def __init__(self, component_type: ComponentType):
        """Initialize the registry with a component type."""
        self.component_type = component_type
        self._factories: dict[str, ComponentFactory] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def register_component(self, name: str, factory: ComponentFactory) -> None:
        """Register a component factory."""
        if name in self._factories:
            self.logger.warning(
                "Overriding existing %s factory: %s", self.component_type.value, name
            )

        self._factories[name] = factory
        self.logger.debug("Registered %s factory: %s", self.component_type.value, name)

    @abstractmethod
    def get_component_factory(self, name: str) -> ComponentFactory | None:
        """Get a registered component factory by name."""
        return self._factories.get(name)

    @abstractmethod
    def list_components(self) -> list[str]:
        """List all registered component names."""
        return list(self._factories.keys())

    @abstractmethod
    def is_registered(self, name: str) -> bool:
        """Check if a component is registered."""
        return name in self._factories

    @abstractmethod
    def get_component_count(self) -> int:
        """Get the number of registered components."""
        return len(self._factories)

    @abstractmethod
    def clear_registry(self) -> None:
        """Clear all registered components (mainly for testing)."""
        self._factories.clear()
        self.logger.debug("Cleared %s registry", self.component_type.value)


class FactoryContext:
    """Context object for factory operations with service injection."""

    def __init__(self, services: dict[str, Any] | None = None):
        """Initialize the factory context with optional services."""
        self._services = services or {}
        self._metadata: dict[str, Any] = {}

    def get_service(self, service_name: str) -> Any | None:
        """Get a service by name."""
        return self._services.get(service_name)

    def has_service(self, service_name: str) -> bool:
        """Check if a service is available."""
        return service_name in self._services

    def add_service(self, service_name: str, service: Any) -> None:
        """Add a service to the context."""
        self._services[service_name] = service

    def get_metadata(self, key: str) -> Any | None:
        """Get metadata by key."""
        return self._metadata.get(key)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata."""
        self._metadata[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary for backward compatibility."""
        return {**self._services, "_metadata": self._metadata}


class ComponentRegistration:
    """Registration information for a component."""

    def __init__(
        self,
        name: str,
        factory: ComponentFactory,
        component_type: ComponentType,
        plugin_info: PluginInfo | None = None,
    ):
        """Initialize a component registration."""
        self.name = name
        self.factory = factory
        self.component_type = component_type
        self.plugin_info = plugin_info
        self.registration_time = logging.time.time()

    def __repr__(self) -> str:
        """String representation of the component registration."""
        return f"ComponentRegistration(name={self.name}, type={self.component_type.value})"


# Factory pattern utilities
def create_factory_context(services: dict[str, Any] | None = None) -> FactoryContext:
    """Create a factory context with optional services."""
    return FactoryContext(services)


def validate_component_factory(factory: Any) -> bool:
    """Validate that an object implements the ComponentFactory protocol."""
    return isinstance(factory, ComponentFactory)


def validate_registry(registry: Any) -> bool:
    """Validate that an object implements the RegistryProtocol."""
    return isinstance(registry, RegistryProtocol)
