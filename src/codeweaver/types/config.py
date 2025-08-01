# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Base types for CodeWeaver's configuration system."""

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from codeweaver.types.base_enum import BaseEnum


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
    CHUNKING = "chunking"
    FILTERING = "filtering"

    # Middleware services (FastMCP integration)
    LOGGING = "logging"
    TIMING = "timing"
    ERROR_HANDLING = "error_handling"
    RATE_LIMITING = "rate_limiting"

    # Optional services
    VALIDATION = "validation"
    CACHE = "cache"
    MONITORING = "monitoring"
    METRICS = "metrics"
    TELEMETRY = "telemetry"

    # Intent layer services
    INTENT = "intent"
    AUTO_INDEXING = "auto_indexing"

    # Phase 3 Intent layer services - LLM-focused optimization
    IMPLICIT_LEARNING = "implicit_learning"
    CONTEXT_INTELLIGENCE = "context_intelligence"
    ZERO_SHOT_OPTIMIZATION = "zero_shot_optimization"

    @classmethod
    def get_core_services(cls) -> tuple["ServiceType"]:
        """Get core services required for basic operation."""
        return (cls.CHUNKING, cls.FILTERING)

    @classmethod
    def get_middleware_services(cls) -> tuple["ServiceType"]:
        """Get middleware services for FastMCP integration."""
        return (cls.LOGGING, cls.TIMING, cls.ERROR_HANDLING, cls.RATE_LIMITING)

    @classmethod
    def get_optional_services(cls) -> tuple["ServiceType"]:
        """Get optional services for enhanced functionality."""
        return (cls.VALIDATION, cls.CACHE, cls.MONITORING, cls.METRICS, cls.TELEMETRY)

    @classmethod
    def get_intent_services(cls) -> tuple["ServiceType"]:
        """Get intent layer services for natural language processing."""
        return (
            cls.INTENT,
            cls.AUTO_INDEXING,
            cls.IMPLICIT_LEARNING,
            cls.CONTEXT_INTELLIGENCE,
            cls.ZERO_SHOT_OPTIMIZATION,
        )

    @classmethod
    def get_all_services(cls) -> tuple["ServiceType"]:
        """Get all available service types."""
        return (
            *cls.get_core_services(),
            *cls.get_middleware_services(),
            *cls.get_optional_services(),
            *cls.get_intent_services(),
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


# ConfigurationError moved to config.py to avoid circular imports


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
