# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Universal component types and protocols for the CodeWeaver factory system.

Provides unified interfaces, configuration models, and result types that
all components (backends, providers, sources) must follow for consistency.
"""

import time

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, NewType, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


# Type variable for generic component types
T = TypeVar("T")


class ComponentType(Enum):
    """Types of components in the system."""

    BACKEND = "backend"
    PROVIDER = "provider"
    SOURCE = "source"


# Dimension size for embeddings (must be positive integer)
DimensionSize = NewType("DimensionSize", int)


class BaseCapabilities(BaseModel):
    """Base class for all capability models.

    Provides common functionality and interface for backend, provider,
    and source capability models.
    """

    def is_compatible_with_requirements(self, **requirements: Any) -> bool:
        """Check if capabilities meet specific requirements.

        Args:
            **requirements: Key-value pairs of requirements to check

        Returns:
            True if all requirements are met, False otherwise
        """
        for req_name, req_value in requirements.items():
            if hasattr(self, req_name):
                capability_value = getattr(self, req_name)
                if (callable(capability_value) and not capability_value()) or (
                    not callable(capability_value) and capability_value != req_value
                ):
                    return False
            else:
                # Unknown requirement - be conservative and return False
                return False
        return True


class CapabilityQueryMixin(ABC):
    """Standard capability query interface for all modules.

    Provides a unified way to query capabilities across backends,
    providers, and sources modules.
    """

    @classmethod
    @abstractmethod
    def get_all_capabilities(cls) -> dict[str, BaseCapabilities]:
        """Get all capabilities for this module type (static).

        Returns:
            Dictionary mapping component names to their capabilities
        """
        ...

    @classmethod
    def get_capability(cls, name: str) -> BaseCapabilities | None:
        """Get capability for a specific provider/backend/source (static).

        Args:
            name: Name of the component to get capabilities for

        Returns:
            Capabilities object if found, None otherwise
        """
        all_caps = cls.get_all_capabilities()
        return all_caps.get(name)

    def get_instance_capabilities(self) -> BaseCapabilities:
        """Get capabilities for this specific instance (dynamic).

        Default implementation returns static capabilities based on
        the instance's name attribute.

        Returns:
            Capabilities object for this instance
        """
        # Default implementation - subclasses can override for dynamic capabilities
        if hasattr(self, "name"):
            return self.get_capability(self.name)
        if hasattr(self, "provider_name"):
            return self.get_capability(self.provider_name)
        if hasattr(self, "source_type"):
            return self.get_capability(self.source_type)
        raise NotImplementedError(
            "Instance must have 'name', 'provider_name', or 'source_type' attribute, "
            "or override get_instance_capabilities() method"
        )


class BaseComponentInfo(BaseModel):
    """Base information model for all components."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    name: Annotated[str, Field(description="Component name")]
    display_name: Annotated[str, Field(description="Human-readable display name")]
    description: Annotated[str, Field(description="Component description")]
    component_type: Annotated[ComponentType, Field(description="Type of component")]
    version: Annotated[str, Field(default="1.0.0", description="Component version")]
    author: Annotated[str | None, Field(default=None, description="Component author")]
    license: Annotated[str | None, Field(default=None, description="Component license")]
    documentation_url: Annotated[str | None, Field(default=None, description="Documentation URL")]
    source_url: Annotated[str | None, Field(default=None, description="Source code URL")]
    implemented: Annotated[
        bool, Field(default=True, description="Whether component is fully implemented")
    ]


@dataclass
class RegistrationResult:
    """Result of component registration."""

    success: bool
    component_name: str | None = None
    errors: list[str] = None
    warnings: list[str] = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        """Initialize the registration result."""
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors in the registration result."""
        return self.errors is not None and len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings in the registration result."""
        return self.warnings is not None and len(self.warnings) > 0


@dataclass
class ValidationResult:
    """Result of component validation."""

    is_valid: bool
    errors: list[str] = None
    warnings: list[str] = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        """Initialize the validation result."""
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors in the validation result."""
        return self.errors is not None and len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings in the validation result."""
        return self.warnings is not None and len(self.warnings) > 0


@dataclass
class CreationResult:
    """Result of component creation."""

    success: bool
    component: Any | None = None
    errors: list[str] = None
    warnings: list[str] = None
    creation_time_ms: float | None = None

    def __post_init__(self):
        """Initialize the creation result."""
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors in the creation result."""
        return self.errors is not None and len(self.errors) > 0


@dataclass
class ComponentRegistration[T]:
    """Universal component registration following providers pattern."""

    component_class: type[T]
    capabilities: BaseCapabilities
    component_info: BaseComponentInfo
    is_available: bool = True
    unavailable_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    registration_time: float = field(default_factory=lambda: time.time())

    @property
    def is_usable(self) -> bool:
        """Check if component is usable (available and implemented)."""
        return self.is_available and self.component_info.implemented


@runtime_checkable
class ComponentRegistry(Protocol):
    """Universal protocol for all component registries."""

    @abstractmethod
    def register_component(
        self,
        name: str,
        component_class: type,
        capabilities: BaseCapabilities,
        component_info: BaseComponentInfo,
        *,
        validate: bool = True,
        check_availability: bool = True,
    ) -> RegistrationResult:
        """Register a component with the registry."""
        ...

    @abstractmethod
    def get_component_class(self, name: str) -> type:
        """Get the component class for a registered component."""
        ...

    @abstractmethod
    def get_capabilities(self, name: str) -> BaseCapabilities:
        """Get capabilities for a registered component."""
        ...

    @abstractmethod
    def get_component_info(self, name: str) -> BaseComponentInfo:
        """Get detailed information about a component."""
        ...

    @abstractmethod
    def list_available_components(self) -> dict[str, BaseComponentInfo]:
        """List all available components and their information."""
        ...

    @abstractmethod
    def has_component(self, name: str) -> bool:
        """Check if a component is registered."""
        ...

    @abstractmethod
    def validate_component(self, name: str) -> ValidationResult:
        """Validate a registered component."""
        ...
