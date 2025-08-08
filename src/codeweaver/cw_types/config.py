# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Base types for CodeWeaver's configuration system."""

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from codeweaver.cw_types.base_enum import BaseEnum


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
    SERVICE = "service"
    # For middleware components (part of services layer)
    # But distinguished by also implementing `fastmcp.MiddleWare`
    MIDDLEWARE = "middleware"

    FACTORY = "factory"
    PLUGIN = "plugin"


class ServiceType(BaseEnum):
    """Types of services in the system."""

    # Core services
    AUTO_INDEXING = "auto_indexing"
    CHUNKING = "chunking"
    FILTERING = "filtering"
    # Telemetry is completely optional, but it's important for improving results and user experience, so we'll call it a core service.
    TELEMETRY = "telemetry"

    # Middleware services (FastMCP integration)
    LOGGING = "logging"
    TIMING = "timing"
    ERROR_HANDLING = "error_handling"
    RATE_LIMITING = "rate_limiting"

    # Intent layer services
    INTENT = "intent"
    IMPLICIT_LEARNING = "implicit_learning"
    CONTEXT_INTELLIGENCE = "context_intelligence"
    ZERO_SHOT_OPTIMIZATION = "zero_shot_optimization"

    # TODO: These services all have protocols and configurations, but aren't implemented in CodeWeaver yet. They are still available if someone wants to extend CodeWeaver with them.
    # We *do* have validation... because pydantic.
    VALIDATION = "validation"
    MONITORING = "monitoring"
    METRICS = "metrics"
    CACHE = "cache"

    @classmethod
    def get_core_services(cls) -> tuple["ServiceType"]:
        """Get core services required for basic operation."""
        return (cls.AUTO_INDEXING, cls.CHUNKING, cls.FILTERING, cls.TELEMETRY)

    @classmethod
    def get_middleware_services(cls) -> tuple["ServiceType"]:
        """Get middleware services for FastMCP integration."""
        return (cls.LOGGING, cls.TIMING, cls.ERROR_HANDLING, cls.RATE_LIMITING)

    @classmethod
    def get_available_but_unimplemented_services(cls) -> tuple["ServiceType"] | tuple[None]:
        """A list of services that are available *to implement* but not yet implemented in CodeWeaver. They have full protocols and configurations, but no implementations."""
        return (cls.VALIDATION, cls.CACHE, cls.MONITORING, cls.METRICS)

    @classmethod
    def get_optional_services(cls) -> tuple["ServiceType"] | tuple[None]:
        """Get optional services that can be enabled based on user needs."""
        return ()

    @classmethod
    def get_intent_services(cls) -> tuple["ServiceType"]:
        """Get intent layer services for natural language processing."""
        return (
            cls.INTENT,
            cls.IMPLICIT_LEARNING,
            cls.CONTEXT_INTELLIGENCE,
            cls.ZERO_SHOT_OPTIMIZATION,
        )

    @classmethod
    def get_all_services(cls) -> tuple["ServiceType"]:
        """Get all service types."""
        return (
            *cls.get_core_services(),
            *cls.get_middleware_services(),
            *cls.get_optional_services(),
            *cls.get_intent_services(),
            *cls.get_available_but_unimplemented_services(),
        )


class ValidationLevel(BaseEnum):
    """Validation levels for configuration and data validation."""

    STRICT = "strict"
    """STRICT: Fail on any error."""
    STANDARD = "standard"
    """STANDARD: Warn on non-critical errors"""
    RELAXED = "relaxed"
    """RELAXED: Allow most configurations and fail gracefully"""
    DISABLED = "disabled"
    """DISABLED: No validation; it's the wild west :gun: :cowboy_hat_face: 🔫"""


# ConfigurationError moved to root config.py to avoid circular imports


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
