# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Enums for intent types, scopes, and complexities."""

from codeweaver.types.base_enum import BaseEnum


class IntentType(BaseEnum):
    """
    Intent types supported by the intent layer.

    Note: INDEX is intentionally excluded - indexing is handled transparently
    by the AutoIndexingService background service.
    """

    SEARCH = "search"
    """Find specific code elements, functions, or patterns."""

    UNDERSTAND = "understand"
    """Understand system architecture or code organization."""

    ANALYZE = "analyze"
    """Analyze code for issues, patterns, or improvements."""


class Scope(BaseEnum):
    """Scope of intent processing."""

    FILE = "file"
    """Single file scope."""

    MODULE = "module"
    """Module or directory scope."""

    PROJECT = "project"
    """Entire project scope."""

    SYSTEM = "system"
    """System-wide scope."""


class Complexity(BaseEnum):
    """Complexity level of intent processing."""

    SIMPLE = "simple"
    """Simple, direct operations."""

    MODERATE = "moderate"
    """Moderate complexity with multiple steps."""

    COMPLEX = "complex"
    """Complex operations requiring advanced processing."""

    ADAPTIVE = "adaptive"
    """Adaptive complexity based on context."""
