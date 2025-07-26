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
# Backend Exceptions
# ============================================================================


class BackendError(Exception):
    """
    Base exception for all backend operations.

    This is the root exception for all vector database backend errors.
    Modules that interact with backends should catch this exception
    to handle any backend-related failures gracefully.

    Attributes:
        backend_type: The type of backend that caused the error
        original_error: The original exception that caused this error (if any)
    """

    def __init__(self, message: str, backend_type: str, original_error: Exception | None = None):
        """
        Initialize backend error with context information.

        Args:
            message: Human-readable error description
            backend_type: Type of backend (e.g., "qdrant", "pinecone", "weaviate")
            original_error: Original exception that caused this error
        """
        super().__init__(message)
        self.backend_type = backend_type
        self.original_error = original_error


class BackendConnectionError(BackendError):
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


class BackendAuthError(BackendError):
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


class CodeWeaverFactoryError(Exception):
    """
    Base exception for all factory system errors.

    This is the root exception for all factory-related errors including
    configuration, component registration, creation, and plugin management.
    Modules that use the factory system should catch this exception to
    handle factory failures gracefully.
    """


class ConfigurationError(CodeWeaverFactoryError):
    """
    Configuration validation or processing error.

    Raised when configuration data is invalid, missing required fields,
    contains conflicting values, or fails validation rules.

    Common causes:
    - Missing required configuration keys
    - Invalid configuration value types or formats
    - Conflicting configuration combinations
    - Environment variable parsing errors

    Recovery strategies:
    - Validate configuration against schema
    - Provide default values for optional settings
    - Clear error messages indicating specific validation failures
    """


class ComponentNotFoundError(CodeWeaverFactoryError):
    """
    Requested component not found in registry.

    Raised when attempting to create or access a component that hasn't
    been registered with the factory system.

    Common causes:
    - Component never registered
    - Typo in component name or identifier
    - Component registration failed silently
    - Plugin containing component not loaded

    Recovery strategies:
    - Verify component name spelling
    - Check if component registration occurred
    - Load required plugins before accessing components
    """


class ComponentUnavailableError(CodeWeaverFactoryError):
    """
    Component found but not available for use.

    Raised when a component is registered but cannot be used due to
    missing dependencies, configuration issues, or runtime constraints.

    Common causes:
    - Missing optional dependencies
    - Invalid component configuration
    - Resource constraints (memory, network, etc.)
    - Component in disabled or maintenance state

    Recovery strategies:
    - Install missing dependencies
    - Fix component configuration
    - Wait for resources to become available
    - Use alternative component if available
    """


class ComponentCreationError(CodeWeaverFactoryError):
    """
    Error occurred during component instantiation.

    Raised when component factory function or constructor fails during
    the creation process, typically due to invalid parameters or
    initialization failures.

    Common causes:
    - Invalid constructor parameters
    - Resource allocation failures
    - Dependency injection failures
    - Component-specific initialization errors

    Recovery strategies:
    - Validate parameters before creation
    - Implement component health checks
    - Provide meaningful error messages from component constructors
    """


class PluginError(CodeWeaverFactoryError):
    """
    Plugin system error.

    Raised when plugin loading, initialization, or execution fails.
    This includes both plugin infrastructure errors and plugin-specific
    failures.

    Common causes:
    - Plugin file not found or corrupted
    - Plugin API version mismatch
    - Plugin dependency conflicts
    - Plugin initialization failures

    Recovery strategies:
    - Verify plugin file integrity
    - Check plugin API compatibility
    - Resolve dependency conflicts
    - Implement plugin sandboxing and error isolation
    """


class RegistrationError(CodeWeaverFactoryError):
    """
    Component registration error.

    Raised when attempting to register a component fails due to
    validation issues, naming conflicts, or registration system problems.

    Common causes:
    - Duplicate component names
    - Invalid component interface
    - Registration system not initialized
    - Component fails validation checks

    Recovery strategies:
    - Use unique component names
    - Validate component interface compliance
    - Initialize registration system properly
    - Implement component validation before registration
    """


class ValidationError(CodeWeaverFactoryError):
    """
    Validation failure error.

    Raised when data, configuration, or component validation fails
    against defined schemas, rules, or constraints.

    Common causes:
    - Schema validation failures
    - Business rule violations
    - Type checking errors
    - Constraint validation failures

    Recovery strategies:
    - Provide detailed validation error messages
    - Implement validation error recovery
    - Use progressive validation with helpful suggestions
    - Validate early and often in the pipeline
    """
