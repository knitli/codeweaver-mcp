# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
FastMCP middleware components for CodeWeaver.

Provides domain-specific middleware implementations for chunking and filtering.
For rate limiting, logging, timing, and error handling, use FastMCP's built-in middleware.
"""

from codeweaver.middleware.chunking import ChunkingMiddleware
from codeweaver.middleware.filtering import FileFilteringMiddleware


__all__ = ["ChunkingMiddleware", "FileFilteringMiddleware"]
