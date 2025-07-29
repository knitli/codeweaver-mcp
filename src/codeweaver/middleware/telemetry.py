# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Telemetry middleware for automatic event tracking."""

import logging
import time

from typing import Any

from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.middleware.middleware import CallNext

from codeweaver.types import TelemetryService


class TelemetryMiddleware(Middleware):
    """FastMCP middleware for automatic telemetry tracking."""

    def __init__(self, telemetry_service: TelemetryService, logger: logging.Logger | None = None):
        """Initialize telemetry middleware."""
        self.telemetry_service = telemetry_service
        self.logger = logger or logging.getLogger("codeweaver.middleware.telemetry")

    async def __call__(
        self,
        context: MiddlewareContext,
        call_next: CallNext,
    ) -> Any:
        """Process request and track telemetry data."""
        start_time = time.time()
        method = getattr(context.request, 'method', 'unknown')

        try:
            # Call the actual handler
            response = await call_next(context)
            duration = time.time() - start_time

            # Track successful operation
            await self._track_operation_success(method, context.request.params, response, duration)

        except Exception as e:
            duration = time.time() - start_time

            # Track operation error
            await self._track_operation_error(method, getattr(context.request, 'params', None), e, duration)

            # Re-raise the exception
            raise
        else:
            return response

    async def _track_operation_success(
        self,
        method: str,
        params: dict[str, Any] | None,
        response: Any,
        duration: float,
    ) -> None:
        """Track successful operations."""
        try:
            if method == "index_codebase":
                await self._track_indexing_success(params, response, duration)
            elif method == "search_code":
                await self._track_search_success(params, response, duration)
            elif method == "ast_grep_search":
                await self._track_ast_grep_success(params, response, duration)
            else:
                # Track general performance metric
                await self.telemetry_service.track_performance(
                    operation=method,
                    duration=duration,
                    metadata={"success": True},
                )

        except Exception as e:
            self.logger.warning("Failed to track operation success: %s", e)

    async def _track_operation_error(
        self,
        method: str,
        params: dict[str, Any] | None,
        error: Exception,
        duration: float,
    ) -> None:
        """Track operation errors."""
        try:
            await self.telemetry_service.track_error(
                error_type=type(error).__name__,
                error_category="operation_error",
                operation=method,
                context={
                    "duration": duration,
                    "has_params": params is not None,
                },
            )

        except Exception as e:
            self.logger.warning("Failed to track operation error: %s", e)

    async def _track_indexing_success(
        self,
        params: dict[str, Any] | None,
        response: Any,
        duration: float,
    ) -> None:
        """Track successful indexing operations."""
        if not params:
            return

        # Extract safe information from parameters
        repository_path = params.get("path", "unknown")

        # Try to extract results from response
        result_data = response if isinstance(response, dict) else {}
        file_count = result_data.get("files_processed", 0)
        language_distribution = result_data.get("languages", {})

        await self.telemetry_service.track_indexing(
            repository_path=repository_path,
            file_count=file_count,
            language_distribution=language_distribution,
            indexing_time=duration,
            success=True,
        )

    async def _track_search_success(
        self,
        params: dict[str, Any] | None,
        response: Any,
        duration: float,
    ) -> None:
        """Track successful search operations."""
        if not params:
            return

        # Extract safe information from parameters
        query = params.get("query", "")
        filters = params.get("filters", [])

        # Try to extract results from response
        result_data = response if isinstance(response, dict) else {}
        results = result_data.get("results", [])
        result_count = len(results) if isinstance(results, list) else 0

        # Determine query complexity
        query_complexity = self._assess_query_complexity(query)

        await self.telemetry_service.track_search(
            query_type="semantic",
            result_count=result_count,
            search_time=duration,
            query_complexity=query_complexity,
            filters_used=filters,
            success=True,
        )

    async def _track_ast_grep_success(
        self,
        params: dict[str, Any] | None,
        response: Any,
        duration: float,
    ) -> None:
        """Track successful AST grep search operations."""
        if not params:
            return

        # Extract safe information from parameters
        pattern = params.get("pattern", "")
        language = params.get("language")

        # Try to extract results from response
        result_data = response if isinstance(response, dict) else {}
        results = result_data.get("results", [])
        result_count = len(results) if isinstance(results, list) else 0

        # Determine pattern complexity
        query_complexity = self._assess_pattern_complexity(pattern)

        filters_used = []
        if language:
            filters_used.append(f"language:{language}")

        await self.telemetry_service.track_search(
            query_type="ast_grep",
            result_count=result_count,
            search_time=duration,
            query_complexity=query_complexity,
            filters_used=filters_used,
            success=True,
        )

    def _assess_query_complexity(self, query: str) -> str:
        """Assess the complexity of a search query."""
        if not query:
            return "empty"

        word_count = len(query.split())
        has_operators = any(op in query for op in ["AND", "OR", "NOT", "+", "-"])
        has_quotes = '"' in query or "'" in query
        has_wildcards = "*" in query or "?" in query

        if word_count > 10 or has_operators:
            return "complex"
        if word_count > 3 or has_quotes or has_wildcards:
            return "medium"
        return "simple"

    def _assess_pattern_complexity(self, pattern: str) -> str:
        """Assess the complexity of an AST grep pattern."""
        if not pattern:
            return "empty"

        # Count special AST grep constructs
        ast_constructs = ["$_", "$A", "$B", "$C", "kind:", "has:", "inside:", "follows:"]
        construct_count = sum(1 for construct in ast_constructs if construct in pattern)

        line_count = len(pattern.split('\n'))

        if construct_count > 3 or line_count > 10:
            return "complex"
        if construct_count > 1 or line_count > 3:
            return "medium"
        return "simple"


class TelemetryMiddlewareProvider:
    """Provider for telemetry middleware integration with FastMCP."""

    def __init__(self, telemetry_service: TelemetryService):
        """Initialize the telemetry middleware provider."""
        self.telemetry_service = telemetry_service
        self.middleware = TelemetryMiddleware(telemetry_service)

    def get_middleware_instance(self) -> TelemetryMiddleware:
        """Get the telemetry middleware instance for FastMCP."""
        return self.middleware

    async def initialize(self) -> None:
        """Initialize the middleware provider."""
        # No special initialization needed

    async def shutdown(self) -> None:
        """Shutdown the middleware provider."""
        # Ensure any pending telemetry is flushed
        await self.telemetry_service.flush()
