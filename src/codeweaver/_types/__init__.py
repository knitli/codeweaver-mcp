# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Centralized types module for CodeWeaver.

This module provides a single import point for all commonly used types,
classes, and functions from the _types subpackage, eliminating circular
dependency issues and simplifying import statements.
"""

# Core types and protocols
# Re-export submodules for backwards compatibility
from codeweaver._types import (
    backends,
    capabilities,
    provider_capabilities,
    provider_enums,
    provider_registry,
    providers,
    source_capabilities,
    source_enums,
    source_providers,
)

# Backend types
from codeweaver._types.backends import (
    CollectionInfo,
    FilterCondition,
    SearchFilter,
    SearchResult,
    VectorPoint,
)

# Base enum for all enums
from codeweaver._types.base_enum import BaseEnum

# Backend capabilities
from codeweaver._types.capabilities import BackendCapabilities, get_all_backend_capabilities

# Configuration types
from codeweaver._types.config import (
    BaseComponentConfig,
    ComponentType,
    ConfigFormat,
    ConfigurationError,
    ServiceType,
    ValidationLevel,
)

# Content models
from codeweaver._types.content import CodeChunk, ContentSearchResult
from codeweaver._types.core import (
    BaseCapabilities,
    BaseComponentInfo,
    CapabilityQueryMixin,
    ComponentRegistration,
    ComponentRegistry,
    CreationResult,
    DimensionSize,
    RegistrationResult,
    T,
    ValidationResult,
)

# Data structures
from codeweaver._types.data_structures import (
    ComponentInstances,
    ComponentLifecycle,
    ContentItem,
    ExtensibilityConfig,
    InitializationContext,
    InitializationResult,
    PluginInfo,
    ProviderRegistration,
)

# Enums
from codeweaver._types.enums import (
    ChunkingStrategy,
    ComponentState,
    ErrorCategory,
    ErrorSeverity,
    Language,
    PerformanceMode,
    SearchComplexity,
)

# Exception types
from codeweaver._types.exceptions import (
    BackendAuthError,
    BackendCollectionNotFoundError,
    BackendConnectionError,
    BackendError,
    BackendUnsupportedOperationError,
    BackendVectorDimensionMismatchError,
    CodeWeaverError,
    CodeWeaverFactoryError,
    ComponentCreationError,
    ComponentNotFoundError,
    ComponentUnavailableError,
    PluginError,
    RegistrationError,
    ValidationError,
)

# Provider capabilities and registry
from codeweaver._types.provider_capabilities import ProviderCapabilities

# Provider enums and types
from codeweaver._types.provider_enums import (
    BackendProvider,
    CohereModels,
    CohereRerankModels,
    DistanceMetric,
    FilterOperator,
    HybridFusionStrategy,
    HybridStrategy,
    IndexType,
    ModelFamily,
    OpenAIModels,
    ProviderCapability,
    ProviderKind,
    ProviderType,
    RerankResult,
    SparseIndexType,
    StorageType,
    VoyageModels,
    VoyageRerankModels,
)
from codeweaver._types.provider_registry import (
    PROVIDER_REGISTRY,
    EmbeddingProviderInfo,
    ProviderRegistryEntry,
    get_available_providers,
    get_provider_registry_entry,
    register_provider_class,
)
from codeweaver._types.providers import PROVIDER_REGISTRY as BACKEND_PROVIDER_REGISTRY

# Provider types from providers module
from codeweaver._types.providers import (
    EmbeddingProviderBase,
    ProviderInfo,
    RerankProviderBase,
    get_provider_info,
    register_backend_class,
)
from codeweaver._types.providers import ProviderRegistryEntry as BackendProviderRegistryEntry
from codeweaver._types.providers import get_available_providers as get_available_backend_providers

# Service types
from codeweaver._types.service_config import (
    CacheServiceConfig,
    ChunkingServiceConfig,
    FilteringServiceConfig,
    MetricsServiceConfig,
    MonitoringServiceConfig,
    ServiceConfig,
    ServicesConfig,
    ValidationServiceConfig,
)
from codeweaver._types.service_data import (
    CacheStats,
    ChunkingStats,
    DirectoryStats,
    FileMetadata,
    FilteringStats,
    HealthStatus,
    ServiceCapabilities,
    ServiceHealth,
    ServiceInstanceInfo,
    ServiceProviderInfo,
    ServiceRegistryHealth,
    ServicesHealthReport,
    ValidationRule,
    ValidationStats,
)
from codeweaver._types.service_data import ValidationResult as ServiceValidationResult
from codeweaver._types.service_exceptions import (
    ChunkingError,
    FilteringError,
    ServiceConfigurationError,
    ServiceCreationError,
    ServiceError,
    ServiceInitializationError,
    ServiceNotFoundError,
)
from codeweaver._types.service_exceptions import ValidationError as ServiceValidationError
from codeweaver._types.services import (
    CacheService,
    ChunkingService,
    FilteringService,
    MetricsService,
    MonitoringService,
    ServiceProvider,
    ValidationService,
)

# Source types
from codeweaver._types.source_capabilities import SourceCapabilities
from codeweaver._types.source_enums import (
    APIType,
    AuthType,
    ContentType,
    DatabaseType,
    SourceCapability,
    SourceProvider,
)
from codeweaver._types.source_providers import SOURCE_PROVIDERS, SourceProviderInfo


__all__ = [
    "BACKEND_PROVIDER_REGISTRY",
    "PROVIDER_REGISTRY",
    "SOURCE_PROVIDERS",
    "APIType",
    "AuthType",
    "BackendAuthError",
    # Backend capabilities
    "BackendCapabilities",
    "BackendCollectionNotFoundError",
    "BackendConnectionError",
    # Exception types
    "BackendError",
    # Provider enums and types
    "BackendProvider",
    "BackendProviderRegistryEntry",
    "BackendUnsupportedOperationError",
    "BackendVectorDimensionMismatchError",
    # Core types and protocols
    "BaseCapabilities",
    # Configuration types
    "BaseComponentConfig",
    "BaseComponentInfo",
    # Base enum
    "BaseEnum",
    # Service types
    "CacheService",
    "CacheServiceConfig",
    "CacheStats",
    "CapabilityQueryMixin",
    "ChunkingError",
    "ChunkingService",
    "ChunkingServiceConfig",
    "ChunkingStats",
    "ChunkingStrategy",
    # Content models
    "CodeChunk",
    "CodeWeaverError",
    "CodeWeaverFactoryError",
    "CohereModels",
    "CohereRerankModels",
    # Backend types
    "CollectionInfo",
    "ComponentCreationError",
    # Data structures
    "ComponentInstances",
    "ComponentLifecycle",
    "ComponentNotFoundError",
    "ComponentRegistration",
    "ComponentRegistry",
    # Enums
    "ComponentState",
    "ComponentType",
    "ComponentUnavailableError",
    "ConfigFormat",
    "ConfigurationError",
    "ContentItem",
    "ContentSearchResult",
    "ContentType",
    "CreationResult",
    "DatabaseType",
    "DimensionSize",
    "DirectoryStats",
    "DistanceMetric",
    # Provider types from providers module
    "EmbeddingProviderBase",
    "EmbeddingProviderInfo",
    "ErrorCategory",
    "ErrorSeverity",
    "ExtensibilityConfig",
    "FileMetadata",
    "FilterCondition",
    "FilterOperator",
    "FilteringError",
    "FilteringService",
    "FilteringServiceConfig",
    "FilteringStats",
    "HealthStatus",
    "HybridFusionStrategy",
    "HybridStrategy",
    "IndexType",
    "InitializationContext",
    "InitializationResult",
    "Language",
    "MetricsService",
    "MetricsServiceConfig",
    "ModelFamily",
    "MonitoringService",
    "MonitoringServiceConfig",
    "OpenAIModels",
    "PerformanceMode",
    "PluginError",
    "PluginInfo",
    # Provider capabilities and registry
    "ProviderCapabilities",
    "ProviderCapability",
    "ProviderInfo",
    "ProviderKind",
    "ProviderRegistration",
    "ProviderRegistryEntry",
    "ProviderType",
    "RegistrationError",
    "RegistrationResult",
    "RerankProviderBase",
    "RerankResult",
    "SearchComplexity",
    "SearchFilter",
    "SearchResult",
    "ServiceCapabilities",
    "ServiceConfig",
    "ServiceConfigurationError",
    "ServiceCreationError",
    "ServiceError",
    "ServiceHealth",
    "ServiceInitializationError",
    "ServiceInstanceInfo",
    "ServiceNotFoundError",
    "ServiceProvider",
    "ServiceProviderInfo",
    "ServiceRegistryHealth",
    "ServiceType",
    "ServiceValidationError",
    "ServiceValidationResult",
    "ServicesConfig",
    "ServicesHealthReport",
    # Source types
    "SourceCapabilities",
    "SourceCapability",
    "SourceProvider",
    "SourceProviderInfo",
    "SparseIndexType",
    "StorageType",
    "T",
    "ValidationError",
    "ValidationLevel",
    "ValidationResult",
    "ValidationRule",
    "ValidationService",
    "ValidationServiceConfig",
    "ValidationStats",
    "VectorPoint",
    "VoyageModels",
    "VoyageRerankModels",
    # Submodules for backwards compatibility
    "backends",
    "capabilities",
    "get_all_backend_capabilities",
    "get_available_backend_providers",
    "get_available_providers",
    "get_provider_info",
    "get_provider_registry_entry",
    "provider_capabilities",
    "provider_enums",
    "provider_registry",
    "providers",
    "register_backend_class",
    "register_provider_class",
    "source_capabilities",
    "source_enums",
    "source_providers",
]
