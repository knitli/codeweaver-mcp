# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Source registry following providers A+ pattern.

Implements unified component registration for data sources with capability
tracking, availability checking, and validation.
"""

import asyncio
import logging

from typing import Annotated, Any, ClassVar

from pydantic import Field

from codeweaver._types import (
    BaseComponentInfo,
    ComponentNotFoundError,
    ComponentRegistration,
    ComponentType,
    ComponentUnavailableError,
    RegistrationResult,
    SourceCapabilities,
    ValidationResult,
)
from codeweaver.sources.base import DataSource, SourceConfig


logger = logging.getLogger(__name__)


class SourceInfo(BaseComponentInfo):
    """Source-specific information."""

    component_type: ComponentType = ComponentType.SOURCE

    # Source-specific fields
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


class SourceRegistry:
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
        check_availability: bool = True,
    ) -> RegistrationResult:
        """Register a data source."""
        if validate:
            validation_result = cls._validate_source_class(source_class)
            if not validation_result.is_valid:
                return RegistrationResult(
                    success=False, component_name=name, errors=validation_result.errors
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
            unavailable_reason=reason,
        )

        cls._sources[name] = registration

        if is_available:
            logger.info("Registered source: %s", name)
        else:
            logger.warning("Registered source %s (unavailable: %s)", name, reason)

        return RegistrationResult(success=True, component_name=name)

    @classmethod
    def register_component(
        cls,
        name: str,
        component_class: type,
        capabilities: SourceCapabilities,
        component_info: BaseComponentInfo,
        *,
        validate: bool = True,
        check_availability: bool = True,
    ) -> RegistrationResult:
        """Register a component (universal interface)."""
        if not issubclass(component_class, DataSource):
            return RegistrationResult(
                success=False,
                component_name=name,
                errors=["Component class must be a DataSource subclass"],
            )

        if not isinstance(component_info, SourceInfo):
            # Convert to SourceInfo if needed
            source_info = SourceInfo(
                name=component_info.name,
                display_name=component_info.display_name,
                description=component_info.description,
                component_type=ComponentType.SOURCE,
                version=component_info.version,
                author=component_info.author,
                license=component_info.license,
                documentation_url=component_info.documentation_url,
                source_url=component_info.source_url,
                implemented=component_info.implemented,
                source_type=name,  # Default source type to name
            )
        else:
            source_info = component_info

        return cls.register_source(
            name=name,
            source_class=component_class,
            capabilities=capabilities,
            source_info=source_info,
            validate=validate,
            check_availability=check_availability,
        )

    @classmethod
    def create_source(cls, source_type: str, config: SourceConfig) -> DataSource:
        """Create a source instance."""
        if source_type not in cls._sources:
            raise ComponentNotFoundError(f"Source '{source_type}' not registered")

        registration = cls._sources[source_type]
        if not registration.is_usable:
            raise ComponentUnavailableError(
                f"Source '{source_type}' is not usable: {registration.unavailable_reason}"
            )

        return registration.component_class(config)

    @classmethod
    def get_capabilities(cls, name: str) -> SourceCapabilities:
        """Get source capabilities."""
        if name not in cls._sources:
            raise ComponentNotFoundError(f"Source '{name}' not registered")
        return cls._sources[name].capabilities

    @classmethod
    def get_component_class(cls, name: str) -> type[DataSource]:
        """Get the component class for a registered source."""
        if name not in cls._sources:
            raise ComponentNotFoundError(f"Source '{name}' not registered")
        return cls._sources[name].component_class

    @classmethod
    def get_component_info(cls, name: str) -> SourceInfo:
        """Get detailed information about a source."""
        if name not in cls._sources:
            raise ComponentNotFoundError(f"Source '{name}' not registered")
        return cls._sources[name].component_info

    @classmethod
    def list_available_components(cls) -> dict[str, SourceInfo]:
        """List all available sources and their information."""
        return {
            name: registration.component_info
            for name, registration in cls._sources.items()
            if registration.is_usable
        }

    @classmethod
    def has_component(cls, name: str) -> bool:
        """Check if a source is registered."""
        return name in cls._sources

    @classmethod
    def validate_component(cls, name: str) -> ValidationResult:
        """Validate a registered source."""
        if name not in cls._sources:
            return ValidationResult(is_valid=False, errors=[f"Source '{name}' not registered"])

        registration = cls._sources[name]

        errors = []
        warnings = []

        if not registration.is_available:
            errors.append(f"Source not available: {registration.unavailable_reason}")

        if not registration.component_info.implemented:
            warnings.append("Source is not fully implemented")

        # Validate component class
        class_validation = cls._validate_source_class(registration.component_class)
        errors.extend(class_validation.errors)
        warnings.extend(class_validation.warnings)

        return ValidationResult(is_valid=not errors, errors=errors, warnings=warnings)

    @classmethod
    def list_available_sources(cls) -> dict[str, dict[str, Any]]:
        """Get available sources and their capabilities (legacy interface)."""
        result = {}
        for name, registration in cls._sources.items():
            capabilities = registration.capabilities
            info = registration.component_info
            result[name] = {
                "supports_discovery": capabilities.supports_content_discovery,
                "supports_reading": capabilities.supports_content_reading,
                "supports_watching": capabilities.supports_change_watching,
                "available": registration.is_available,
                "implemented": info.implemented,
                "description": info.description,
            }
        return result

    @classmethod
    def create_multiple_sources(cls, source_configs: list[dict[str, Any]]) -> list[DataSource]:
        """Create multiple data source instances."""
        sources = []
        for config_dict in source_configs:
            source_type = config_dict.get("type")
            if not source_type:
                raise ValueError("Source configuration must include 'type' field")

            config = SourceConfig(config_dict)
            source = cls.create_source(source_type, config)
            sources.append(source)

        return sources

    @classmethod
    async def validate_source_config(cls, source_type: str, config: SourceConfig) -> bool:
        """Validate a source configuration."""
        if source_type not in cls._sources:
            return False

        registration = cls._sources[source_type]
        if not registration.is_usable:
            return False

        try:
            # Basic validation - check if source can be created
            source_class = registration.component_class
            if hasattr(source_class, "validate_config"):
                if asyncio.iscoroutinefunction(source_class.validate_config):
                    return await source_class.validate_config(config)
                return source_class.validate_config(config)

            # If no validation method, assume valid
        except Exception:
            return False
        else:
            return True

    @classmethod
    def _validate_source_class(cls, source_class: type[DataSource]) -> ValidationResult:
        """Validate that a source class implements required methods."""
        errors = []
        warnings = []

        required_methods = ["discover_content", "read_content", "get_metadata"]

        errors.extend(
            f"Source class missing required method: {method}"
            for method in required_methods
            if not hasattr(source_class, method)
        )
        # Check if it's actually a DataSource subclass
        if not issubclass(source_class, DataSource):
            errors.append("Source class must inherit from DataSource")

        return ValidationResult(is_valid=not errors, errors=errors, warnings=warnings)

    @classmethod
    def _check_source_availability(cls, source_class: type[DataSource]) -> tuple[bool, str | None]:
        """Check if a source is available for use."""
        try:
            # Check if required dependencies are available
            if hasattr(source_class, "_check_dependencies") and (
                missing_deps := source_class._check_dependencies()
            ):
                return False, f"Missing dependencies: {', '.join(missing_deps)}"

            # TODO: Additional availability checks could go here

        except Exception as e:
            return False, f"Availability check failed: {e}"
        else:
            return True, None

    @classmethod
    def initialize_builtin_sources(cls) -> None:
        """Initialize built-in source registrations."""
        if cls._initialized:
            return

        try:
            # Register filesystem source
            from codeweaver.sources.filesystem import FileSystemSourceProvider

            filesystem_capabilities = SourceCapabilities(
                supports_content_discovery=True,
                supports_content_reading=True,
                supports_change_watching=True,
                supports_metadata_extraction=True,
                max_content_size_mb=1024,  # 1GB max file size
                required_dependencies=["watchdog"],
            )

            filesystem_info = SourceInfo(
                name="filesystem",
                display_name="Filesystem Source",
                description="Read content from local filesystem with directory scanning and file watching",
                component_type=ComponentType.SOURCE,
                source_type="filesystem",
                supported_schemes=["file"],
                configuration_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Root directory path"},
                        "patterns": {"type": "array", "items": {"type": "string"}},
                        "exclude_patterns": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["path"],
                },
            )

            cls.register_source(
                name="filesystem",
                source_class=FileSystemSourceProvider,
                capabilities=filesystem_capabilities,
                source_info=filesystem_info,
            )

        except ImportError:
            logger.warning("Filesystem source not available - missing dependencies")

        # Additional built-in sources would be registered here

        cls._initialized = True
