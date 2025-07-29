# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
FastMCP middleware components for CodeWeaver.

Provides domain-specific middleware implementations for chunking and filtering.

Also re-exports FastMCP's built-in middleware for convenience.
"""

from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware, RetryMiddleware
from fastmcp.server.middleware.logging import LoggingMiddleware, StructuredLoggingMiddleware
from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware
from fastmcp.server.middleware.timing import TimingMiddleware

from codeweaver.middleware.chunking import ChunkingMiddleware
from codeweaver.middleware.filtering import FileFilteringMiddleware


__all__ = [
    "ChunkingMiddleware",
    "ErrorHandlingMiddleware",
    "FileFilteringMiddleware",
    "LoggingMiddleware",
    "RateLimitingMiddleware",
    "RetryMiddleware",
    "StructuredLoggingMiddleware",
    "TimingMiddleware",
]
