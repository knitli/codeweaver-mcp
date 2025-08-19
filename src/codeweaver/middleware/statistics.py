# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
# sourcery skip: avoid-single-character-names-variables

"""Statistics middleware for FastMCP."""

from __future__ import annotations

import logging
import time

from typing import Any, cast, overload

from fastmcp.prompts import Prompt
from fastmcp.resources import Resource, ResourceTemplate
from fastmcp.server.middleware import CallNext, MiddlewareContext
from fastmcp.server.middleware.middleware import Middleware
from fastmcp.tools import Tool
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    GetPromptRequestParams,
    GetPromptResult,
    ListPromptsRequest,
    ListResourcesRequest,
    ListResourceTemplatesRequest,
    ListToolsRequest,
    ReadResourceRequestParams,
    ReadResourceResult,
)
from typing_extensions import TypeIs

from codeweaver._statistics import (
    McpOperationRequests,
    SessionStatistics,
    TimingStatistics,
    TimingStatisticsDict,
)
from codeweaver.exceptions import ProviderError


class StatisticsMiddleware(Middleware):
    """Middleware to track request statistics and performance metrics."""

    def __init__(
        self,
        statistics: SessionStatistics | None = None,
        logger: logging.Logger | None = None,
        log_level: int = logging.INFO,
    ) -> None:
        """Initialize statistics middleware.

        Args:
            statistics: Statistics instance to use for tracking
            logger: Logger instance to use for logging
            log_level: Logging level to use
        """
        self.statistics = statistics or SessionStatistics()
        self.timing_statistics = self.timing_statistics or TimingStatistics(
            {}, {}, {}, [], [], [], []
        )
        self.logger = logger or logging.getLogger("codeweaver.middleware.statistics")
        self.log_level = log_level or logging.INFO
        self._we_are_not_none()

    def _stats_is_stats(self) -> TypeIs[SessionStatistics]:  # type: ignore
        return isinstance(self.statistics, SessionStatistics)  # type: ignore

    def _timing_stats_is_stats(self) -> TypeIs[TimingStatistics]:  # type: ignore
        return isinstance(self.timing_statistics, TimingStatistics)  # type: ignore

    def _we_are_not_none(self) -> None:
        """Ensure that all required statistics are present."""
        if not self.statistics:
            raise ProviderError("Failed to initialize statistics middleware provider.")
        if not self.timing_statistics:
            raise ProviderError("Failed to initialize timing statistics.")

    # Trust me, I tried to define this with generics, but it was a nightmare. I blame the fastmcp types.
    @overload
    async def _time_operation(
        self,
        context: MiddlewareContext[CallToolRequestParams],
        call_next: CallNext[CallToolRequestParams, CallToolResult],
        operation_name: str,
    ) -> CallToolResult: ...
    @overload
    async def _time_operation(
        self,
        context: MiddlewareContext[ReadResourceRequestParams],
        call_next: CallNext[ReadResourceRequestParams, ReadResourceResult],
        operation_name: str,
    ) -> ReadResourceResult: ...
    @overload
    async def _time_operation(
        self,
        context: MiddlewareContext[GetPromptRequestParams],
        call_next: CallNext[GetPromptRequestParams, GetPromptResult],
        operation_name: str,
    ) -> GetPromptResult: ...
    @overload
    async def _time_operation(
        self,
        context: MiddlewareContext[ListResourcesRequest],
        call_next: CallNext[ListResourcesRequest, list[Resource]],
        operation_name: str,
    ) -> list[Resource]: ...
    @overload
    async def _time_operation(
        self,
        context: MiddlewareContext[ListResourceTemplatesRequest],
        call_next: CallNext[ListResourceTemplatesRequest, list[ResourceTemplate]],
        operation_name: str,
    ) -> list[ResourceTemplate]: ...
    @overload
    async def _time_operation(
        self,
        context: MiddlewareContext[ListPromptsRequest],
        call_next: CallNext[ListPromptsRequest, list[Prompt]],
        operation_name: str,
    ) -> list[Prompt]: ...
    @overload
    async def _time_operation(
        self,
        context: MiddlewareContext[ListToolsRequest],
        call_next: CallNext[ListToolsRequest, list[Tool]],
        operation_name: str,
    ) -> list[Tool]: ...
    async def _time_operation(
        self, context: MiddlewareContext[Any], call_next: CallNext[Any, Any], operation_name: str
    ) -> Any:
        """Helper method to time any operation."""
        if not self._stats_is_stats() or not self._timing_stats_is_stats():
            raise ProviderError("Statistics middleware is not properly initialized.")
        start_time = time.perf_counter()
        try:
            result = await call_next(context)
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.statistics.add_successful_request()
            self.timing_statistics.update(cast(McpOperationRequests, operation_name), duration_ms)

        except Exception:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.statistics.add_failed_request()
            self.logger.exception(
                "Operation in %s failed after %.2fms",
                operation_name,
                duration_ms,
                extra={"failed_operation": operation_name, "duration_ms": duration_ms},
            )
            raise
        else:
            return result

    async def on_call_tool(
        self,
        context: MiddlewareContext[CallToolRequestParams],
        call_next: CallNext[CallToolRequestParams, CallToolResult],
    ) -> CallToolResult:
        """Handle incoming requests and track statistics."""
        return await self._time_operation(context, call_next, "on_call_tool_requests", context.)

    async def on_read_resource(
        self,
        context: MiddlewareContext[ReadResourceRequestParams],
        call_next: CallNext[ReadResourceRequestParams, ReadResourceResult],
    ) -> ReadResourceResult:
        """Handle resource read requests and track statistics."""
        return await self._time_operation(context, call_next, "on_read_resource_requests")

    async def on_get_prompt(
        self,
        context: MiddlewareContext[GetPromptRequestParams],
        call_next: CallNext[GetPromptRequestParams, GetPromptResult],
    ) -> GetPromptResult:
        """Handle prompt retrieval requests and track statistics."""
        return await self._time_operation(context, call_next, "on_get_prompt_requests")

    async def on_list_tools(
        self,
        context: MiddlewareContext[ListToolsRequest],
        call_next: CallNext[ListToolsRequest, list[Tool]],
    ) -> list[Tool]:
        """Handle tool listing requests and track statistics."""
        return await self._time_operation(context, call_next, "on_list_tools_requests")

    async def on_list_resources(
        self,
        context: MiddlewareContext[ListResourcesRequest],
        call_next: CallNext[ListResourcesRequest, list[Resource]],
    ) -> list[Resource]:
        """Handle resource listing requests and track statistics."""
        return await self._time_operation(context, call_next, "on_list_resources_requests")

    async def on_list_resource_templates(
        self,
        context: MiddlewareContext[ListResourceTemplatesRequest],
        call_next: CallNext[ListResourceTemplatesRequest, list[ResourceTemplate]],
    ) -> list[ResourceTemplate]:
        """Handle resource template listing requests and track statistics."""
        return await self._time_operation(context, call_next, "on_list_resource_templates_requests")

    async def on_list_prompts(
        self,
        context: MiddlewareContext[ListPromptsRequest],
        call_next: CallNext[ListPromptsRequest, list[Prompt]],
    ) -> list[Prompt]:
        """Handle prompt listing requests and track statistics."""
        return await self._time_operation(context, call_next, "on_list_prompts_requests")

    def get_statistics(self) -> SessionStatistics:
        """Get current statistics.

        Returns:
            Current session statistics
        """
        return self.statistics

    def get_timing_statistics(self) -> TimingStatisticsDict:
        """Get current timing statistics.

        Returns:
            Current timing statistics
        """
        return self.timing_statistics.timing_summary()

    def reset_statistics(self) -> None:
        """Reset all statistics to initial state."""
        self.statistics.reset()
