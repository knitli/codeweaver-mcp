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
from __future__ import annotations

from typing import TYPE_CHECKING, Literal


if TYPE_CHECKING:
    from fastmcp import Client, FastMCP
    from fastmcp.client import FastMCPTransport

from codeweaver.cw_types.backends import (
    BACKEND_PROVIDER_REGISTRY,
    BackendCapabilities,
    BackendProvider,
    CollectionInfo,
    DistanceMetric,
    EmbeddingProviderBase,
    FilterCondition,
    FilterOperator,
    HybridFusionStrategy,
    HybridStrategy,
    IndexType,
    ProviderInfo,
    RerankProviderBase,
    SearchFilter,
    SearchResult,
    SparseIndexType,
    StorageType,
    VectorPoint,
    get_all_backend_capabilities,
)
from codeweaver.cw_types.base_enum import BaseEnum

# Configuration types
from codeweaver.cw_types.config import (
    BaseComponentConfig,
    ComponentType,
    ConfigFormat,
    ServiceType,
    ValidationLevel,
)

# Content models
from codeweaver.cw_types.content import CodeChunk, ContentSearchResult

# Enums
from codeweaver.cw_types.enums import (
    ChunkingStrategy,
    ComponentState,
    ErrorCategory,
    ErrorSeverity,
    Language,
    PerformanceMode,
    SearchComplexity,
)

# Exception types
# Backend exceptions and provider exceptions
from codeweaver.cw_types.exceptions import (
    BackendAuthError,
    BackendCollectionNotFoundError,
    BackendConnectionError,
    BackendError,
    BackendUnsupportedOperationError,
    BackendVectorDimensionMismatchError,
    CodeWeaverError,
    ComponentError,
    EmbeddingProviderError,
    ProviderAuthError,
    ProviderCompatibilityError,
    ProviderConfigurationError,
    ProviderConnectionError,
    ProviderError,
    ProviderResourceError,
    ServiceProviderError,
    SourceProviderError,
)

# Factories and data structures
from codeweaver.cw_types.factories import (
    BaseCapabilities,
    BaseComponentInfo,
    CapabilityQueryMixin,
    ComponentInstances,
    ComponentLifecycle,
    ComponentRegistration,
    ComponentRegistry,
    ContentItem,
    CreationResult,
    DimensionSize,
    ExtensibilityConfig,
    InitializationContext,
    InitializationResult,
    PluginInfo,
    ProviderRegistration,
    RegistrationResult,
    T,
    ValidationResult,
)

# Intent types
from codeweaver.cw_types.intent import (
    # Phase 3 implicit learning types
    BehavioralPatterns,
    # Base intent types
    Complexity,
    ContextAdequacy,
    ContextIntelligenceService,
    ImplicitFeedback,
    ImplicitLearningService,
    IntentError,
    IntentParsingError,
    IntentResult,
    IntentStrategy,
    IntentType,
    LearningSignal,
    LLMProfile,
    OptimizationStrategy,
    OptimizedIntentPlan,
    ParsedIntent,
    SatisfactionSignals,
    Scope,
    ServiceIntegrationError,
    SessionSignals,
    StrategyExecutionError,
    StrategySelectionError,
    SuccessPrediction,
    ZeroShotMetrics,
    ZeroShotOptimizationService,
)
from codeweaver.cw_types.language import ConfigLanguage, LanguageConfigFile, SemanticSearchLanguage

# Provider capabilities and registry
# Provider types from providers module
from codeweaver.cw_types.providers import (
    PROVIDER_REGISTRY,
    CohereModel,
    CohereRerankModel,
    EmbeddingProviderInfo,
    ModelFamily,
    NLPCapability,
    NLPModelSize,
    OpenAIModel,
    ProviderCapabilities,
    ProviderCapability,
    ProviderKind,
    ProviderRegistryEntry,
    ProviderType,
    RerankResult,
    VoyageModel,
    VoyageRerankModel,
    get_available_providers,
    get_provider_registry_entry,
    register_provider_class,
)
from codeweaver.cw_types.services import (
    AccessDeniedError,
    AutoIndexingConfig,
    CacheError,
    CacheService,
    CacheServiceConfig,
    CacheStats,
    ChunkingError,
    ChunkingService,
    ChunkingServiceConfig,
    ChunkingStats,
    ContextIntelligenceServiceConfig,
    DependencyInjectionError,
    DependencyResolutionError,
    DirectoryNotFoundError,
    DirectoryStats,
    DuplicateProviderError,
    ErrorHandlingService,
    ErrorHandlingServiceConfig,
    FileMetadata,
    FilteringError,
    FilteringService,
    FilteringServiceConfig,
    FilteringStats,
    HealthStatus,
    ImplicitLearningServiceConfig,
    IntentServiceConfig,
    LoggingService,
    LoggingServiceConfig,
    MemoryUsage,
    MetricsService,
    MetricsServiceConfig,
    MonitoringService,
    MonitoringServiceConfig,
    PerformanceProfile,
    ProviderNotFoundError,
    ProviderRegistrationError,
    ProviderStatus,
    RateLimitingService,
    RateLimitingServiceConfig,
    ReconfigurationError,
    ServiceCapabilities,
    ServiceConfig,
    ServiceConfigurationError,
    ServiceCreationError,
    ServiceError,
    ServiceHealth,
    ServiceInitializationError,
    ServiceInstanceInfo,
    ServiceNotFoundError,
    ServiceNotReadyError,
    ServiceProvider,
    ServiceProviderInfo,
    ServiceRegistryHealth,
    ServiceRestartError,
    ServicesConfig,
    ServicesHealthReport,
    ServiceStartError,
    ServiceStopError,
    TelemetryService,
    TelemetryServiceConfig,
    TimingService,
    TimingServiceConfig,
    UnsupportedLanguageError,
    ValidationErrorData,
    ValidationRule,
    ValidationService,
    ValidationServiceConfig,
    ValidationSeverity,
    ValidationStats,
    ValidationWarning,
    ZeroShotOptimizationServiceConfig,
)
from codeweaver.cw_types.sources import (
    SOURCE_PROVIDERS,
    APIType,
    AuthType,
    ContentType,
    DatabaseType,
    SourceCapabilities,
    SourceCapability,
    SourceProvider,
    SourceProviderInfo,
)


CodeWeaver = FastMCP[Literal["CodeWeaver"]]
CodeWeaverClient = Client[FastMCPTransport]


# Configuration exceptions - define locally to avoid circular imports
class ConfigurationError(Exception):
    """Configuration-related errors."""


class ProfileError(ConfigurationError):
    """Profile-related configuration errors."""


class PluginConfigurationError(ConfigurationError):
    """Plugin configuration errors."""


# Factory exceptions - define locally to avoid circular import with factories module
class CodeWeaverFactoryError(Exception):
    """Base exception for factory-related errors."""


class ComponentCreationError(CodeWeaverFactoryError):
    """Error during component creation."""


class ComponentNotFoundError(CodeWeaverFactoryError):
    """Requested component not found in registry."""


class ComponentUnavailableError(CodeWeaverFactoryError):
    """Component found but not available for use."""


class PluginError(CodeWeaverFactoryError):
    """Plugin-related error."""


class RegistrationError(CodeWeaverFactoryError):
    """Component registration error."""


class ValidationError(CodeWeaverFactoryError):
    """Validation failure error."""


__all__ = (
    "BACKEND_PROVIDER_REGISTRY",
    "PROVIDER_REGISTRY",
    "SOURCE_PROVIDERS",
    "APIType",
    "AccessDeniedError",
    "AuthType",
    "AutoIndexingConfig",
    "BackendAuthError",
    "BackendCapabilities",
    "BackendCollectionNotFoundError",
    "BackendConnectionError",
    "BackendError",
    "BackendProvider",
    "BackendUnsupportedOperationError",
    "BackendVectorDimensionMismatchError",
    "BaseCapabilities",
    "BaseComponentConfig",
    "BaseComponentInfo",
    "BaseEnum",
    "BehavioralPatterns",
    "CacheError",
    "CacheService",
    "CacheServiceConfig",
    "CacheStats",
    "CapabilityQueryMixin",
    "ChunkingError",
    "ChunkingService",
    "ChunkingServiceConfig",
    "ChunkingStats",
    "ChunkingStrategy",
    "CodeChunk",
    "CodeWeaver",
    "CodeWeaverClient",
    "CodeWeaverError",
    "CodeWeaverFactoryError",
    "CohereModel",
    "CohereRerankModel",
    "CollectionInfo",
    "Complexity",
    "ComponentCreationError",
    "ComponentError",
    "ComponentInstances",
    "ComponentLifecycle",
    "ComponentNotFoundError",
    "ComponentRegistration",
    "ComponentRegistry",
    "ComponentState",
    "ComponentType",
    "ComponentUnavailableError",
    "ConfigFormat",
    "ConfigLanguage",
    "ConfigurationError",
    "ContentItem",
    "ContentSearchResult",
    "ContentType",
    "ContextAdequacy",
    "ContextIntelligenceService",
    "ContextIntelligenceServiceConfig",
    "CreationResult",
    "DatabaseType",
    "DependencyInjectionError",
    "DependencyResolutionError",
    "DimensionSize",
    "DirectoryNotFoundError",
    "DirectoryStats",
    "DistanceMetric",
    "DuplicateProviderError",
    "EmbeddingProviderBase",
    "EmbeddingProviderError",
    "EmbeddingProviderInfo",
    "ErrorCategory",
    "ErrorHandlingService",
    "ErrorHandlingServiceConfig",
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
    "ImplicitFeedback",
    "ImplicitLearningService",
    "ImplicitLearningServiceConfig",
    "IndexType",
    "InitializationContext",
    "InitializationResult",
    "IntentError",
    "IntentParsingError",
    "IntentResult",
    "IntentServiceConfig",
    "IntentStrategy",
    "IntentType",
    "LLMProfile",
    "Language",
    "LanguageConfigFile",
    "LearningSignal",
    "LoggingService",
    "LoggingServiceConfig",
    "MemoryUsage",
    "MetricsService",
    "MetricsServiceConfig",
    "ModelFamily",
    "MonitoringService",
    "MonitoringServiceConfig",
    "NLPCapability",
    "NLPModelSize",
    "OpenAIModel",
    "OptimizationStrategy",
    "OptimizedIntentPlan",
    "ParsedIntent",
    "PerformanceMode",
    "PerformanceProfile",
    "PluginConfigurationError",
    "PluginError",
    "PluginInfo",
    "ProfileError",
    "ProviderAuthError",
    "ProviderCapabilities",
    "ProviderCapability",
    "ProviderCompatibilityError",
    "ProviderConfigurationError",
    "ProviderConnectionError",
    "ProviderError",
    "ProviderInfo",
    "ProviderKind",
    "ProviderNotFoundError",
    "ProviderRegistration",
    "ProviderRegistrationError",
    "ProviderRegistryEntry",
    "ProviderResourceError",
    "ProviderStatus",
    "ProviderType",
    "RateLimitingService",
    "RateLimitingServiceConfig",
    "ReconfigurationError",
    "RegistrationError",
    "RegistrationResult",
    "RerankProviderBase",
    "RerankResult",
    "SatisfactionSignals",
    "Scope",
    "SearchComplexity",
    "SearchFilter",
    "SearchResult",
    "SemanticSearchLanguage",
    "ServiceCapabilities",
    "ServiceConfig",
    "ServiceConfigurationError",
    "ServiceCreationError",
    "ServiceError",
    "ServiceHealth",
    "ServiceInitializationError",
    "ServiceInstanceInfo",
    "ServiceIntegrationError",
    "ServiceNotFoundError",
    "ServiceNotReadyError",
    "ServiceProvider",
    "ServiceProviderError",
    "ServiceProviderInfo",
    "ServiceRegistryHealth",
    "ServiceRestartError",
    "ServiceStartError",
    "ServiceStopError",
    "ServiceType",
    "ServicesConfig",
    "ServicesHealthReport",
    "SessionSignals",
    "SourceCapabilities",
    "SourceCapability",
    "SourceProvider",
    "SourceProviderError",
    "SourceProviderInfo",
    "SparseIndexType",
    "StorageType",
    "StrategyExecutionError",
    "StrategySelectionError",
    "SuccessPrediction",
    "T",
    "TelemetryService",
    "TelemetryServiceConfig",
    "TimingService",
    "TimingServiceConfig",
    "UnsupportedLanguageError",
    "ValidationError",
    "ValidationErrorData",
    "ValidationLevel",
    "ValidationResult",
    "ValidationRule",
    "ValidationService",
    "ValidationServiceConfig",
    "ValidationSeverity",
    "ValidationStats",
    "ValidationWarning",
    "VectorPoint",
    "VoyageModel",
    "VoyageRerankModel",
    "ZeroShotMetrics",
    "ZeroShotOptimizationService",
    "ZeroShotOptimizationServiceConfig",
    "get_all_backend_capabilities",
    "get_available_providers",
    "get_provider_registry_entry",
    "register_provider_class",
)
