# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Backend registry following providers A+ pattern.

Implements unified component registration for vector database backends
with capability tracking, availability checking, and validation.
"""

import logging

from typing import Annotated, ClassVar

from pydantic import Field

from codeweaver._types import (
    BackendCapabilities,
    BaseComponentInfo,
    ComponentNotFoundError,
    ComponentRegistration,
    ComponentType,
    ComponentUnavailableError,
    RegistrationResult,
    ValidationResult,
)
from codeweaver.backends.base import VectorBackend
from codeweaver.backends.config import BackendConfig


logger = logging.getLogger(__name__)


class BackendInfo(BaseComponentInfo):
    """Backend-specific information."""

    component_type: ComponentType = ComponentType.BACKEND

    # Backend-specific fields
    backend_type: Annotated[
        str, Field(description="Type of vector backend (e.g., 'qdrant', 'pinecone')")
    ]
    connection_requirements: Annotated[
        dict[str, str], Field(default_factory=dict, description="Required connection parameters")
    ]
    optional_parameters: Annotated[
        dict[str, str], Field(default_factory=dict, description="Optional configuration parameters")
    ]


class BackendRegistry:
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
        check_availability: bool = True,
    ) -> RegistrationResult:
        """Register a vector backend."""
        if validate:
            validation_result = cls._validate_backend_class(backend_class)
            if not validation_result.is_valid:
                return RegistrationResult(
                    success=False, component_name=name, errors=validation_result.errors
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
            unavailable_reason=reason,
        )

        cls._backends[name] = registration

        if is_available:
            logger.info("Registered backend: %s", name)
        else:
            logger.warning("Registered backend %s (unavailable: %s)", name, reason)

        return RegistrationResult(success=True, component_name=name)

    @classmethod
    def register_component(
        cls,
        name: str,
        component_class: type,
        capabilities: BackendCapabilities,
        component_info: BaseComponentInfo,
        *,
        validate: bool = True,
        check_availability: bool = True,
    ) -> RegistrationResult:
        """Register a component (universal interface)."""
        if not issubclass(component_class, VectorBackend):
            return RegistrationResult(
                success=False,
                component_name=name,
                errors=["Component class must be a VectorBackend subclass"],
            )

        if not isinstance(component_info, BackendInfo):
            # Convert to BackendInfo if needed
            backend_info = BackendInfo(
                name=component_info.name,
                display_name=component_info.display_name,
                description=component_info.description,
                component_type=ComponentType.BACKEND,
                version=component_info.version,
                author=component_info.author,
                license=component_info.license,
                documentation_url=component_info.documentation_url,
                source_url=component_info.source_url,
                implemented=component_info.implemented,
                backend_type=name,  # Default backend type to name
            )
        else:
            backend_info = component_info

        return cls.register_backend(
            name=name,
            backend_class=component_class,
            capabilities=capabilities,
            backend_info=backend_info,
            validate=validate,
            check_availability=check_availability,
        )

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

    @classmethod
    def get_component_class(cls, name: str) -> type[VectorBackend]:
        """Get the component class for a registered backend."""
        if name not in cls._backends:
            raise ComponentNotFoundError(f"Backend '{name}' not registered")
        return cls._backends[name].component_class

    @classmethod
    def get_component_info(cls, name: str) -> BackendInfo:
        """Get detailed information about a backend."""
        if name not in cls._backends:
            raise ComponentNotFoundError(f"Backend '{name}' not registered")
        return cls._backends[name].component_info

    @classmethod
    def list_available_components(cls) -> dict[str, BackendInfo]:
        """List all available backends and their information."""
        return {
            name: registration.component_info
            for name, registration in cls._backends.items()
            if registration.is_usable
        }

    @classmethod
    def has_component(cls, name: str) -> bool:
        """Check if a backend is registered."""
        return name in cls._backends

    @classmethod
    def validate_component(cls, name: str) -> ValidationResult:
        """Validate a registered backend."""
        if name not in cls._backends:
            return ValidationResult(is_valid=False, errors=[f"Backend '{name}' not registered"])

        registration = cls._backends[name]

        errors = []
        warnings = []

        if not registration.is_available:
            errors.append(f"Backend not available: {registration.unavailable_reason}")

        if not registration.component_info.implemented:
            warnings.append("Backend is not fully implemented")

        # Validate component class
        class_validation = cls._validate_backend_class(registration.component_class)
        errors.extend(class_validation.errors)
        warnings.extend(class_validation.warnings)

        return ValidationResult(is_valid=not errors, errors=errors, warnings=warnings)

    @classmethod
    def get_supported_providers(cls) -> dict[str, dict[str, bool]]:
        """Get supported providers and their capabilities (legacy interface)."""
        result = {}
        for name, registration in cls._backends.items():
            capabilities = registration.capabilities
            result[name] = {
                "supports_hybrid": capabilities.supports_hybrid_search,
                "supports_filtering": capabilities.supports_filtering,
                "available": registration.is_available,
                "implemented": registration.component_info.implemented,
            }
        return result

    @classmethod
    def _validate_backend_class(cls, backend_class: type[VectorBackend]) -> ValidationResult:
        """Validate that a backend class implements required methods."""
        warnings = []

        required_methods = [
            "create_collection",
            "upsert_vectors",
            "search_vectors",
            "delete_vectors",
            "get_collection_info",
        ]

        errors = [
            f"Backend class missing required method: {method}"
            for method in required_methods
            if not hasattr(backend_class, method)
        ]
        # Check if it's actually a VectorBackend subclass
        if not issubclass(backend_class, VectorBackend):
            errors.append("Backend class must inherit from VectorBackend")

        return ValidationResult(is_valid=not errors, errors=errors, warnings=warnings)

    @classmethod
    def _check_backend_availability(
        cls, backend_class: type[VectorBackend]
    ) -> tuple[bool, str | None]:
        """Check if a backend is available for use."""
        try:
            # Check if required dependencies are available
            if hasattr(backend_class, "_check_dependencies") and (
                missing_deps := backend_class._check_dependencies()
            ):
                return False, f"Missing dependencies: {', '.join(missing_deps)}"

            # Additional availability checks could go here

        except Exception as e:
            return False, f"Availability check failed: {e}"

        else:
            return True, None

    @classmethod
    def initialize_builtin_backends(cls) -> None:
        """Initialize built-in backend registrations."""
        if cls._initialized:
            return

        # Import and register built-in backends
        from codeweaver._types import get_all_backend_capabilities

        try:
            # Register Qdrant backend
            from codeweaver.backends.qdrant import QdrantBackend

            if qdrant_capabilities := get_all_backend_capabilities().get("qdrant"):
                qdrant_info = BackendInfo(
                    name="qdrant",
                    display_name="Qdrant Vector Database",
                    description="High-performance vector similarity search engine with hybrid search support",
                    component_type=ComponentType.BACKEND,
                    backend_type="qdrant",
                    connection_requirements={"url": "Connection URL to Qdrant instance"},
                    optional_parameters={
                        "api_key": "API key for authentication",
                        "timeout": "Request timeout in seconds",
                    },
                )

                cls.register_backend(
                    name="qdrant",
                    backend_class=QdrantBackend,
                    capabilities=qdrant_capabilities,
                    backend_info=qdrant_info,
                )

        except ImportError:
            logger.warning("Qdrant backend not available - missing dependencies")

        # Additional built-in backends would be registered here

        cls._initialized = True
