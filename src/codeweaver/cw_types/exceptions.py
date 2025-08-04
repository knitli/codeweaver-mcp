# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Consolidated exception types for CodeWeaver.

This module contains all exception types that are used across multiple modules
or serve as core infrastructure exceptions that other modules might need to catch.
Exceptions are organized into clear hierarchies with comprehensive documentation.
"""


# ============================================================================
# Base Exceptions
# ============================================================================


class CodeWeaverError(Exception):
    """
    Base exception for all CodeWeaver errors.

    This is the root exception for all errors in the CodeWeaver system.
    All other exceptions should inherit from this class to provide a
    consistent error hierarchy and enable comprehensive error handling.

    Attributes:
        component: The component or module that raised the error
        original_error: The original exception that caused this error (if any)
    """

    def __init__(
        self, message: str, component: str | None = None, original_error: Exception | None = None
    ):
        """
        Initialize CodeWeaver error with context information.

        Args:
            message: Human-readable error description
            component: Component or module that raised the error
            original_error: Original exception that caused this error
        """
        super().__init__(message)
        self.component = component
        self.original_error = original_error


# ============================================================================
# Component and Provider Base Exceptions
# ============================================================================


class ComponentError(CodeWeaverError):
    """
    Base exception for all component operations.

    This serves as a unified error base for all components in the CodeWeaver
    system including backends, providers, sources, and services. It provides
    consistent error context and standardized error handling patterns.

    Attributes:
        component_type: The type of component (backend, provider, source, service)
        component_name: The specific name of the component instance
        operation: The operation that failed (optional)
    """

    def __init__(
        self,
        message: str,
        component_type: str,
        component_name: str,
        operation: str | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize component error with comprehensive context.

        Args:
            message: Human-readable error description
            component_type: Type of component (backend, provider, source, service)
            component_name: Specific component instance name
            operation: Operation that failed (e.g., "embed", "search", "index")
            original_error: Original exception that caused this error
        """
        super().__init__(
            message, component=f"{component_type}:{component_name}", original_error=original_error
        )
        self.component_type = component_type
        self.component_name = component_name
        self.operation = operation


class ProviderError(ComponentError):
    """
    Base exception for all provider operations.

    This exception provides a unified error interface for all provider types
    including embedding providers, backend providers, source providers, and
    service providers. It ensures consistent error handling patterns and
    context across the entire provider ecosystem.

    Attributes:
        provider_type: The category of provider (embedding, backend, source, service)
        provider_name: The specific provider implementation name
        operation: The operation that failed
        recovery_suggestions: List of potential recovery strategies
    """

    def __init__(
        self,
        message: str,
        provider_type: str,
        provider_name: str,
        operation: str | None = None,
        original_error: Exception | None = None,
        recovery_suggestions: list[str] | None = None,
    ):
        """
        Initialize provider error with comprehensive context and recovery guidance.

        Args:
            message: Human-readable error description
            provider_type: Category of provider (embedding, backend, source, service)
            provider_name: Specific provider implementation name
            operation: Operation that failed (e.g., "embed", "search", "index")
            original_error: Original exception that caused this error
            recovery_suggestions: List of potential recovery strategies
        """
        super().__init__(message, provider_type, provider_name, operation, original_error)
        self.recovery_suggestions = recovery_suggestions or []


# ============================================================================
# Provider-Specific Error Categories
# ============================================================================


class EmbeddingProviderError(ProviderError):
    """
    Exception for embedding and reranking provider operations.

    Specialized error type for embedding providers (VoyageAI, OpenAI,
    HuggingFace, etc.) that includes embedding-specific context and
    recovery strategies.
    """

    def __init__(
        self,
        message: str,
        provider_name: str,
        operation: str | None = None,
        model_name: str | None = None,
        original_error: Exception | None = None,
        recovery_suggestions: list[str] | None = None,
    ):
        """
        Initialize embedding provider error with context and recovery guidance."""
        super().__init__(
            message, "embedding", provider_name, operation, original_error, recovery_suggestions
        )
        self.model_name = model_name


class SourceProviderError(ProviderError):
    """
    Exception for data source provider operations.

    Specialized error type for source providers (filesystem, git, database,
    web, API) that includes source-specific context and recovery strategies.
    """

    def __init__(
        self,
        message: str,
        provider_name: str,
        operation: str | None = None,
        source_path: str | None = None,
        original_error: Exception | None = None,
        recovery_suggestions: list[str] | None = None,
    ):
        """Initialize source provider error with context and recovery guidance."""
        super().__init__(
            message, "source", provider_name, operation, original_error, recovery_suggestions
        )
        self.source_path = source_path


class ServiceProviderError(ProviderError):
    """
    Exception for service provider operations.

    Specialized error type for service providers (chunking, filtering,
    validation, etc.) that includes service-specific context.
    """

    def __init__(
        self,
        message: str,
        provider_name: str,
        operation: str | None = None,
        service_context: dict | None = None,
        original_error: Exception | None = None,
        recovery_suggestions: list[str] | None = None,
    ):
        """Initialize service provider error with context and recovery guidance."""
        super().__init__(
            message, "service", provider_name, operation, original_error, recovery_suggestions
        )
        self.service_context = service_context or {}


# ============================================================================
# Operation-Specific Error Types
# ============================================================================


class ProviderConnectionError(ProviderError):
    """
    Provider connection establishment failed.

    Raised when a provider cannot establish connection to its underlying
    service (API endpoints, databases, file systems, etc.).

    Recovery strategies:
    - Verify network connectivity and service availability
    - Check endpoint URLs and connection parameters
    - Implement retry logic with exponential backoff
    - Validate firewall and proxy configurations
    """


class ProviderAuthError(ProviderError):
    """
    Provider authentication failed.

    Raised when authentication credentials are invalid, expired, or
    insufficient permissions for the requested operation.

    Recovery strategies:
    - Verify API keys, tokens, and credentials
    - Check permission levels for the operation
    - Refresh or rotate credentials if expired
    - Validate credential scope and access rights
    """


class ProviderConfigurationError(ProviderError):
    """
    Provider configuration validation failed.

    Raised when provider configuration is missing, invalid, or incompatible
    with the requested operation or underlying service.

    Recovery strategies:
    - Validate all required configuration parameters
    - Check parameter types and value ranges
    - Verify compatibility with provider version
    - Review default configuration fallbacks
    """


class ProviderResourceError(ProviderError):
    """
    Provider resource limits exceeded.

    Raised when provider encounters resource limitations such as rate limits,
    quota exceeded, insufficient capacity, or temporary service unavailability.

    Recovery strategies:
    - Implement rate limiting and backoff strategies
    - Check quota usage and limits
    - Retry after specified delay periods
    - Consider alternative providers or scaling options
    """


class ProviderCompatibilityError(ProviderError):
    """
    Provider compatibility validation failed.

    Raised when provider version, feature set, or API compatibility
    is insufficient for the requested operation.

    Recovery strategies:
    - Update provider to compatible version
    - Check feature availability and alternatives
    - Implement fallback for unsupported operations
    - Validate API version compatibility
    """


# ============================================================================
# Backend Exceptions (Updated to inherit from ProviderError)
# ============================================================================


class BackendError(ProviderError):
    """
    Base exception for all backend operations.

    This is the root exception for all vector database backend errors.
    Inherits from ProviderError to maintain consistency with the unified
    provider error hierarchy while preserving existing backend_type compatibility.

    Attributes:
        backend_type: The type of backend that caused the error (legacy compatibility)
        provider_name: The backend provider name (unified interface)
        original_error: The original exception that caused this error (if any)
    """

    def __init__(
        self,
        message: str,
        backend_type: str,
        original_error: Exception | None = None,
        operation: str | None = None,
        recovery_suggestions: list[str] | None = None,
    ):
        """
        Initialize backend error with context information.

        Args:
            message: Human-readable error description
            backend_type: Type of backend (e.g., "qdrant", "pinecone", "weaviate")
            original_error: Original exception that caused this error
            operation: Operation that failed (optional)
            recovery_suggestions: Recovery strategies (optional)
        """
        # Map to unified provider interface while maintaining legacy compatibility
        super().__init__(
            message, "backend", backend_type, operation, original_error, recovery_suggestions
        )
        self.backend_type = backend_type  # Legacy compatibility


class BackendConnectionError(BackendError, ProviderConnectionError):
    """
    Backend connection establishment failed.

    Raised when the backend cannot be reached or connection is refused.
    This typically indicates network issues, incorrect URLs, or backend
    service unavailability.

    Recovery strategies:
    - Verify backend URL and network connectivity
    - Check if backend service is running
    - Implement retry logic with exponential backoff
    """


class BackendAuthError(BackendError, ProviderAuthError):
    """
    Backend authentication failed.

    Raised when authentication credentials are invalid, expired, or
    insufficient permissions for the requested operation.

    Recovery strategies:
    - Verify API keys and credentials
    - Check permission levels for the operation
    - Refresh or rotate credentials if expired
    """


class BackendCollectionNotFoundError(BackendError):
    """
    Requested collection does not exist in the backend.

    Raised when attempting to access a collection that hasn't been created
    or has been deleted. This is often recoverable by creating the collection.

    Recovery strategies:
    - Create the collection if it should exist
    - Verify collection name spelling and casing
    - Check if collection was accidentally deleted
    """


class BackendVectorDimensionMismatchError(BackendError):
    """
    Vector dimensions don't match collection configuration.

    Raised when attempting to insert vectors with dimensions that don't
    match the collection's configured dimension size.

    Recovery strategies:
    - Verify embedding model produces correct dimensions
    - Check collection configuration
    - Re-create collection with correct dimensions if needed
    """


class BackendUnsupportedOperationError(BackendError):
    """
    Operation not supported by the current backend.

    Raised when attempting an operation that the specific backend
    implementation doesn't support (e.g., hybrid search on backends
    that only support vector search).

    Recovery strategies:
    - Use a different backend that supports the operation
    - Implement operation fallback logic
    - Check backend capabilities before attempting operation
    """


# ============================================================================
# Factory System Exceptions
# ============================================================================
# NOTE: Factory exceptions are defined in [factories.exceptions](../factories/exceptions) module
# This avoids circular imports and keeps factory-specific exceptions together
