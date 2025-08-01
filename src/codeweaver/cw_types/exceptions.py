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
# NOTE: Factory exceptions are defined in factories.exceptions module
# This avoids circular imports and keeps factory-specific exceptions together
