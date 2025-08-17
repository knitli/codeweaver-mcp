# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
# sourcery skip: avoid-single-character-names-variables

"""Statistics middleware for FastMCP."""

from __future__ import annotations

from typing import Any, TypeVar

from fastmcp.server.middleware import Middleware, MiddlewareContext

from codeweaver._statistics import SessionStatistics


T = TypeVar("T")


class StatisticsMiddleware(Middleware):
    """Middleware to track request statistics and performance metrics."""

    def __init__(self, statistics: SessionStatistics | None = None) -> None:
        """Initialize statistics middleware.

        Args:
            statistics: Statistics instance to use for tracking
        """
        self.statistics = statistics or SessionStatistics()

    async def on_call_tool(self, context: MiddlewareContext[T], call_next: Any) -> T:
        """Handle incoming requests and track statistics."""

    async def on_read_resource(self, context: MiddlewareContext[T], call_next: Any) -> T:
        """Handle resource read requests and track statistics."""

    async def on_get_prompt(self, context, call_next):
        """Handle prompt retrieval requests and track statistics."""

    async def on_list_tools(self, context, call_next):
        """Handle tool listing requests and track statistics."""

    async def on_list_resources(self, context, call_next):
        """Handle resource listing requests and track statistics."""

    async def on_list_resource_templates(self, context, call_next):
        """Handle resource template listing requests and track statistics."""

    async def on_list_prompts(self, context, call_next):
        """Handle prompt listing requests and track statistics."""

    def get_statistics(self) -> SessionStatistics:
        """Get current statistics.

        Returns:
            Current session statistics
        """
        return self.statistics

    def reset_statistics(self) -> None:
        """Reset all statistics to initial state."""
        self.statistics.reset()
