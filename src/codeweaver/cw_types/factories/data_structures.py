# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Centralized data structures for CodeWeaver.

Contains reusable data structures that are used across multiple modules
in the CodeWeaver system, including configuration containers, component
tracking, registration information, and universal content representation.
"""

import time

from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, computed_field
from pydantic.dataclasses import dataclass

from codeweaver.cw_types.factories.core import BaseCapabilities, BaseComponentInfo, ComponentType
from codeweaver.cw_types.providers import EmbeddingProviderInfo, ProviderCapability
from codeweaver.cw_types.sources import ContentType


if TYPE_CHECKING:
    from codeweaver.backends.base import VectorBackend
    from codeweaver.data_sources.base import DataSource
    from codeweaver.providers.combined import CombinedProviderBase
    from codeweaver.providers.embedding import EmbeddingProvider, EmbeddingProviderBase
    from codeweaver.providers.rerank import RerankProvider, RerankProviderBase


@dataclass
class PluginInfo:
    """
    Information about a discovered plugin.

    Contains all metadata and references needed to work with a plugin,
    including its class, capabilities, and discovery context.
    """

    name: str
    component_type: ComponentType
    plugin_class: type
    capabilities: BaseCapabilities
    component_info: BaseComponentInfo
    entry_point: str | None = None
    file_path: str | None = None
    module_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class ExtensibilityConfig:
    """
    Configuration for extensibility features.

    Controls how the CodeWeaver factory system discovers, loads, and
    manages extensible components like plugins and custom implementations.
    """

    # Plugin discovery settings
    enable_plugin_discovery: bool = True
    plugin_directories: list[str] | None = None
    auto_discover_plugins: bool = True

    # Factory settings
    enable_dependency_injection: bool = True
    validate_configurations: bool = True

    # Lifecycle management
    enable_graceful_shutdown: bool = True
    shutdown_timeout: float = 30.0


@dataclass
class ComponentInstances:
    """
    Container for instantiated components.

    Holds references to all the major component instances created by
    the factory system, providing a centralized registry of active components.

    Note: Rate limiting is handled by FastMCP's RateLimitingMiddleware,
    not as a separate component.
    """

    backend: "VectorBackend | None" = None  # VectorBackend
    embedding_provider: "EmbeddingProvider | None" = None  # EmbeddingProvider
    reranking_provider: "RerankProvider | None" = None  # RerankProvider
    data_sources: "list[DataSource] | None" = None  # list[DataSource]


@dataclass
class ComponentLifecycle:
    """
    Component lifecycle tracking.

    Tracks the state and timing information for a component throughout
    its lifecycle from creation to destruction.
    """

    component_name: str
    component_type: ComponentType
    state: str = "uninitialized"  # ComponentState enum value
    created_at: float = Field(default_factory=time.time)
    initialized_at: float | None = None
    started_at: float | None = None
    stopped_at: float | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class InitializationContext:
    """
    Context for initialization process.

    Provides all the necessary context and dependencies for component
    initialization stages to execute successfully.
    """

    factory: Any
    config: Any
    registries: dict[str, Any]
    plugin_manager: Any | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    lifecycle_tracker: dict[str, ComponentLifecycle] = Field(default_factory=dict)


@dataclass
class InitializationResult:
    """
    Result of an initialization stage.

    Contains success status, timing information, and any errors or
    warnings generated during a specific initialization stage.
    """

    success: bool
    stage_name: str
    duration_ms: float
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class ProviderRegistration:
    """
    Registration information for a provider.

    Contains all information needed to register and manage a provider
    implementation, including its capabilities and availability status.
    """

    provider_class: "EmbeddingProviderBase | RerankProviderBase | CombinedProviderBase"
    # EmbeddingProviderBase | RerankProviderBase | CombinedProvider
    capabilities: list[ProviderCapability]
    provider_info: EmbeddingProviderInfo
    is_available: bool = True
    unavailable_reason: str | None = None
    registration_metadata: dict[str, Any] = Field(default_factory=dict)


class ContentItem(BaseModel):
    """
    Universal content item with Pydantic validation.

    Represents a piece of content from any data source with standardized
    metadata and validation. Supports source-specific extensions through
    the flexible metadata field.
    """

    model_config = ConfigDict(
        extra="allow",  # Allow source-specific metadata
        validate_assignment=True,
        use_enum_values=True,
    )

    path: Annotated[str, Field(description="Universal identifier for content")]
    content_type: Annotated[ContentType, Field(description="Type of content")]
    metadata: Annotated[
        dict[str, Any], Field(default_factory=dict, description="Source-specific metadata")
    ]
    last_modified: Annotated[
        datetime | None, Field(None, description="Last modification timestamp")
    ]
    size: Annotated[int | None, Field(None, ge=0, description="Content size in bytes")]
    language: Annotated[str | None, Field(None, description="Detected programming language")]
    source_id: Annotated[str | None, Field(None, description="Source identifier")]
    version: Annotated[str | None, Field(None, description="Version identifier")]
    checksum: Annotated[str | None, Field(None, description="Content checksum")]

    @computed_field
    @property
    def is_text(self) -> bool:
        """Check if this content item represents text content."""
        return self.content_type in {
            ContentType.CODE,
            ContentType.DOCUMENTATION,
            ContentType.CONFIG,
            ContentType.TEXT,
        }

    @computed_field
    @property
    def display_name(self) -> str:
        """Get a human-readable display name for this content item."""
        return self.path.split("/")[-1] if "/" in self.path else self.path
