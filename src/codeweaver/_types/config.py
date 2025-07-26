# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Base types for CodeWeaver's configuration system."""

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from codeweaver._types.base_enum import BaseEnum


class ConfigFormat(BaseEnum):
    """Supported configuration formats."""

    TOML = "toml"
    JSON = "json"

    def __str__(self) -> str:
        """Return the string representation of the format."""
        return self.value


class ComponentType(BaseEnum):
    """Component types in the CodeWeaver system."""

    BACKEND = "backend"
    PROVIDER = "provider"
    SOURCE = "source"
    SERVICE = "service"  # NEW: For service providers
    MIDDLEWARE = "middleware"  # NEW: For middleware components
    FACTORY = "factory"
    PLUGIN = "plugin"


class ServiceType(BaseEnum):
    """Types of services in the system."""

    CHUNKING = "chunking"
    FILTERING = "filtering"
    VALIDATION = "validation"
    CACHE = "cache"
    MONITORING = "monitoring"
    METRICS = "metrics"

    @classmethod
    def get_core_services(cls) -> list["ServiceType"]:
        """Get core services required for basic operation."""
        return [cls.CHUNKING, cls.FILTERING]

    @classmethod
    def get_optional_services(cls) -> list["ServiceType"]:
        """Get optional services for enhanced functionality."""
        return [cls.VALIDATION, cls.CACHE, cls.MONITORING, cls.METRICS]


class ValidationLevel(BaseEnum):
    """Configuration validation levels."""

    STRICT = "strict"  # Fail on any validation error
    STANDARD = "standard"  # Warn on non-critical errors
    PERMISSIVE = "permissive"  # Allow most configurations


class ConfigurationError(Exception):
    """Base exception for configuration errors."""


class BaseComponentConfig(BaseModel):
    """
    Base configuration for all CodeWeaver components.

    Follows the factories architecture specification for unified
    component configuration patterns.
    """

    model_config = ConfigDict(
        extra="allow",  # Allow component-specific extensions
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=True,
        frozen=False,
    )

    # Core identification
    component_type: Annotated[ComponentType, Field(description="Type of component")]
    provider: Annotated[str, Field(description="Provider/implementation name")]
    name: Annotated[str | None, Field(default=None, description="Optional component instance name")]

    # Control settings
    enabled: Annotated[bool, Field(default=True, description="Whether component is enabled")]

    # Plugin support (matches factories specification)
    custom_class: Annotated[
        str | None, Field(default=None, description="Custom implementation class path")
    ]
    custom_module: Annotated[
        str | None, Field(default=None, description="Custom implementation module")
    ]

    # Validation settings
    validate_on_creation: Annotated[
        bool, Field(default=True, description="Validate component on creation")
    ]
    fail_fast: Annotated[
        bool, Field(default=True, description="Fail immediately on validation errors")
    ]

    # Metadata
    metadata: Annotated[
        dict[str, Any], Field(default_factory=dict, description="Component metadata")
    ]
