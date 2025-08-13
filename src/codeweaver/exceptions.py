# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Unified exception hierarchy for CodeWeaver.

This module provides a single, unified exception hierarchy to prevent exception
proliferation. All CodeWeaver exceptions inherit from CodeWeaverError and
are organized into five primary categories.
"""

from __future__ import annotations

from typing import Any


class CodeWeaverError(Exception):
    """Base exception for all CodeWeaver errors.

    Provides structured error information including details and suggestions
    for resolution.
    """

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        suggestions: list[str] | None = None,
    ) -> None:
        """Initialize CodeWeaver error.

        Args:
            message: Human-readable error message
            details: Additional context about the error
            suggestions: Actionable suggestions for resolving the error
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.suggestions = suggestions or []


class ConfigurationError(CodeWeaverError):
    """Configuration and settings errors.

    Raised when there are issues with configuration files, environment variables,
    settings validation, or provider configuration.
    """


class ProviderError(CodeWeaverError):
    """Provider integration errors.

    Raised when there are issues with embedding providers, vector stores,
    or other external service integrations.
    """


class IndexingError(CodeWeaverError):
    """File indexing and processing errors.

    Raised when there are issues with file discovery, content processing,
    or index building operations.
    """


class QueryError(CodeWeaverError):
    """Query processing and search errors.

    Raised when there are issues with query validation, search execution,
    or result processing.
    """


class ValidationError(CodeWeaverError):
    """Input validation and schema errors.

    Raised when there are issues with input validation, data model validation,
    or schema compliance.
    """
