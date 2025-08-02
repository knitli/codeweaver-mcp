# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""FastMCP middleware service providers."""

import logging

from typing import Any

from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
from fastmcp.server.middleware.logging import LoggingMiddleware
from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware
from fastmcp.server.middleware.timing import TimingMiddleware

from codeweaver.cw_types import (
    ErrorHandlingService,
    ErrorHandlingServiceConfig,
    LoggingService,
    LoggingServiceConfig,
    RateLimitingService,
    RateLimitingServiceConfig,
    ServiceType,
    TimingService,
    TimingServiceConfig,
)
from codeweaver.services.providers.base_provider import BaseServiceProvider


class FastMCPLoggingProvider(BaseServiceProvider, LoggingService):
    """FastMCP-based logging service provider."""

    VERSION = "1.0.0"

    def __init__(self, service_type: ServiceType, config: LoggingServiceConfig) -> None:
        """Initialize the logging service provider."""
        super().__init__(service_type, config)
        self._config = config  # Type-specific config
        self._middleware: LoggingMiddleware | None = None
        self._logger = logging.getLogger(f"codeweaver.services.{self.name}")

    async def _initialize_provider(self) -> None:
        """Initialize FastMCP logging middleware."""
        middleware_config = {
            "log_level": self._config.log_level,
            "include_payloads": self._config.include_payloads,
            "max_payload_length": self._config.max_payload_length,
            "structured": self._config.structured_logging,
            "performance_metrics": self._config.log_performance_metrics,
        }

        # Only include methods filter if specified
        if self._config.methods is not None:
            middleware_config["methods"] = self._config.methods

        self._middleware = LoggingMiddleware(middleware_config)
        self._logger.info("FastMCP logging provider initialized")

    async def _shutdown_provider(self) -> None:
        """Shutdown logging provider."""
        self._middleware = None
        self._logger.info("FastMCP logging provider shut down")

    async def _check_health(self) -> bool:
        """Check if logging service is healthy."""
        return self._middleware is not None

    def set_log_level(self, level: str) -> None:
        """Set the logging level."""
        self._config.log_level = level
        if self._middleware:
            # Update middleware configuration if possible
            self._logger.info("Updated log level to %s", level)

    def set_include_payloads(self, *, include: bool) -> None:
        """Set whether to include payloads in logs."""
        self._config.include_payloads = include
        if self._middleware:
            self._logger.info("Updated include_payloads to %s", include)

    def set_max_payload_length(self, length: int) -> None:
        """Set maximum payload length to log."""
        self._config.max_payload_length = length
        if self._middleware:
            self._logger.info("Updated max_payload_length to %d", length)

    async def log_request(self, method: str, params: dict[str, Any]) -> None:
        """Log an incoming request."""
        if not self._middleware:
            return

        self._logger.debug("Request: %s with params", method)
        if self._config.include_payloads:
            payload = str(params)[: self._config.max_payload_length]
            self._logger.debug("Request payload: %s", payload)

    async def log_response(self, method: str, result: Any, duration: float) -> None:
        """Log an outgoing response."""
        if not self._middleware:
            return

        self._logger.debug("Response: %s (%.2fms)", method, duration * 1000)
        if self._config.include_payloads:
            result = str(result)[: self._config.max_payload_length]
            self._logger.debug("Response payload: %s", result)

    def get_middleware_instance(self) -> LoggingMiddleware | None:
        """Get the FastMCP middleware instance."""
        return self._middleware

    async def create_service_context(
        self, base_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create service context for logging operations."""
        context = base_context.copy() if base_context else {}

        # Add service reference and basic info
        context.update({
            "logging_service": self,
            "service_type": self.service_type,
            "provider_name": self.name,
            "provider_version": self.version,
        })

        # Add capabilities and configuration
        context.update({
            "capabilities": self.capabilities,
            "configuration": {
                "log_level": self._config.log_level,
                "include_payloads": self._config.include_payloads,
                "max_payload_length": self._config.max_payload_length,
            },
            "middleware_available": self._middleware is not None,
        })

        # Add health status
        health = await self.health_check()
        context.update({
            "health_status": health.status,
            "service_healthy": health.status.name == "HEALTHY",
            "last_error": health.last_error,
        })

        return context


class FastMCPTimingProvider(BaseServiceProvider, TimingService):
    """FastMCP-based timing service provider."""

    VERSION = "1.0.0"

    def __init__(self, service_type: ServiceType, config: TimingServiceConfig) -> None:
        """Initialize the timing service provider."""
        super().__init__(service_type, config)
        self._config = config  # Type-specific config
        self._middleware: TimingMiddleware | None = None
        self._timing_stats: dict[str, list[float]] = {}
        self._logger = logging.getLogger(f"codeweaver.services.{self.name}")

    async def _initialize_provider(self) -> None:
        """Initialize FastMCP timing middleware."""
        middleware_config = {
            "log_level": self._config.log_level,
            "track_metrics": self._config.track_performance_metrics,
            "expose_endpoint": self._config.expose_metrics_endpoint,
            "aggregation_window": self._config.metric_aggregation_window,
        }

        self._middleware = TimingMiddleware(middleware_config)
        self._logger.info("FastMCP timing provider initialized")

    async def _shutdown_provider(self) -> None:
        """Shutdown timing provider."""
        self._middleware = None
        self._timing_stats.clear()
        self._logger.info("FastMCP timing provider shut down")

    async def _check_health(self) -> bool:
        """Check if timing service is healthy."""
        return self._middleware is not None

    def set_track_metrics(self, *, track: bool) -> None:
        """Set whether to track performance metrics."""
        self._config.track_performance_metrics = track
        if self._middleware:
            self._logger.info("Updated track_metrics to %s", track)

    async def record_timing(self, method: str, duration: float) -> None:
        """Record timing for a method call."""
        if not self._middleware:
            return

        # Store timing data
        if method not in self._timing_stats:
            self._timing_stats[method] = []

        self._timing_stats[method].append(duration)

        # Keep only recent timings (last 1000)
        if len(self._timing_stats[method]) > 1000:
            self._timing_stats[method] = self._timing_stats[method][-1000:]

        if self._config.track_performance_metrics:
            self._logger.debug("Recorded timing for %s: %.2fms", method, duration * 1000)

    async def get_timing_stats(self) -> dict[str, Any]:
        """Get timing statistics."""
        if not self._timing_stats:
            return {}

        return {
            method: {
                "count": len(timings),
                "avg": sum(timings) / len(timings),
                "min": min(timings),
                "max": max(timings),
                "recent_avg": (sum(timings[-10:]) / min(len(timings), 10) if timings else 0),
            }
            for method, timings in self._timing_stats.items()
            if timings
        }

    def get_middleware_instance(self) -> TimingMiddleware | None:
        """Get the FastMCP middleware instance."""
        return self._middleware

    async def create_service_context(
        self, base_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create service context for timing operations."""
        context = base_context.copy() if base_context else {}

        # Add service reference and basic info
        context.update({
            "timing_service": self,
            "service_type": self.service_type,
            "provider_name": self.name,
            "provider_version": self.version,
        })

        # Add capabilities and configuration
        context.update({
            "capabilities": self.capabilities,
            "configuration": {"track_performance_metrics": self._config.track_performance_metrics},
            "middleware_available": self._middleware is not None,
        })

        # Add health status
        health = await self.health_check()
        context.update({
            "health_status": health.status,
            "service_healthy": health.status.name == "HEALTHY",
            "last_error": health.last_error,
        })

        # Add runtime statistics
        context.update({"statistics": await self.get_timing_stats()})

        return context


class FastMCPErrorHandlingProvider(BaseServiceProvider, ErrorHandlingService):
    """FastMCP-based error handling service provider."""

    VERSION = "1.0.0"

    def __init__(self, service_type: ServiceType, config: ErrorHandlingServiceConfig) -> None:
        """Initialize the error handling service provider."""
        super().__init__(service_type, config)
        self._config = config  # Type-specific config
        self._middleware: ErrorHandlingMiddleware | None = None
        self._error_history: list[dict[str, Any]] = []
        self._error_stats = {"total_errors": 0, "errors_by_type": {}}
        self._logger = logging.getLogger(f"codeweaver.services.{self.name}")

    async def _initialize_provider(self) -> None:
        """Initialize FastMCP error handling middleware."""
        middleware_config = {
            "include_traceback": self._config.include_traceback,
            "transform_errors": self._config.transform_errors,
        }

        self._middleware = ErrorHandlingMiddleware(middleware_config)
        self._logger.info("FastMCP error handling provider initialized")

    async def _shutdown_provider(self) -> None:
        """Shutdown error handling provider."""
        self._middleware = None
        self._error_history.clear()
        self._error_stats = {"total_errors": 0, "errors_by_type": {}}
        self._logger.info("FastMCP error handling provider shut down")

    async def _check_health(self) -> bool:
        """Check if error handling service is healthy."""
        return self._middleware is not None

    def set_include_traceback(self, *, include: bool) -> None:
        """Set whether to include traceback in errors."""
        self._config.include_traceback = include
        if self._middleware:
            self._logger.info("Updated include_traceback to %s", include)

    def set_transform_errors(self, *, transform: bool) -> None:
        """Set whether to transform errors to MCP format."""
        self._config.transform_errors = transform
        if self._middleware:
            self._logger.info("Updated transform_errors to %s", transform)

    async def handle_error(self, error: Exception, context: dict[str, Any]) -> Any:
        """Handle an error with the middleware."""
        if not self._middleware:
            return None

        # Record error for statistics
        error_type = type(error).__name__
        self._error_stats["total_errors"] += 1
        self._error_stats["errors_by_type"][error_type] = (
            self._error_stats["errors_by_type"].get(error_type, 0) + 1
        )

        # Add to error history if aggregation is enabled
        if self._config.error_aggregation:
            error_record = {
                "error_type": error_type,
                "message": str(error),
                "context": context,
                "timestamp": self._get_current_timestamp(),
            }

            self._error_history.append(error_record)

            # Maintain history size limit
            if len(self._error_history) > self._config.max_error_history:
                self._error_history = self._error_history[-self._config.max_error_history :]

        self._logger.warning("Handled error: %s - %s", error_type, str(error))

        # Let middleware handle the error transformation
        return error

    async def get_error_stats(self) -> dict[str, Any]:
        """Get error handling statistics."""
        return {
            "total_errors": self._error_stats["total_errors"],
            "errors_by_type": self._error_stats["errors_by_type"].copy(),
            "error_history_size": len(self._error_history),
            "recent_errors": self._error_history[-10:] if self._error_history else [],
        }

    def get_middleware_instance(self) -> ErrorHandlingMiddleware | None:
        """Get the FastMCP middleware instance."""
        return self._middleware

    async def create_service_context(
        self, base_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create service context for error handling operations."""
        context = base_context.copy() if base_context else {}

        # Add service reference and basic info
        context.update({
            "error_handling_service": self,
            "service_type": self.service_type,
            "provider_name": self.name,
            "provider_version": self.version,
        })

        # Add capabilities and configuration
        context.update({
            "capabilities": self.capabilities,
            "configuration": {
                "include_traceback": self._config.include_traceback,
                "transform_errors": self._config.transform_errors,
                "max_error_history": self._config.max_error_history,
            },
            "middleware_available": self._middleware is not None,
        })

        # Add health status
        health = await self.health_check()
        context.update({
            "health_status": health.status,
            "service_healthy": health.status.name == "HEALTHY",
            "last_error": health.last_error,
        })

        # Add runtime statistics
        context.update({"statistics": await self.get_error_stats()})

        return context

    def _get_current_timestamp(self) -> str:
        """Get current timestamp as string."""
        from datetime import UTC, datetime

        return datetime.now(UTC).isoformat()


class FastMCPRateLimitingProvider(BaseServiceProvider, RateLimitingService):
    """FastMCP-based rate limiting service provider."""

    VERSION = "1.0.0"

    def __init__(self, service_type: ServiceType, config: RateLimitingServiceConfig) -> None:
        """Initialize the rate limiting service provider."""
        super().__init__(service_type, config)
        self._config = config  # Type-specific config
        self._middleware: RateLimitingMiddleware | None = None
        self._rate_limit_stats = {"total_requests": 0, "rejected_requests": 0, "current_rate": 0.0}
        self._logger = logging.getLogger(f"codeweaver.services.{self.name}")

    async def _initialize_provider(self) -> None:
        """Initialize FastMCP rate limiting middleware."""
        middleware_config = {
            "max_requests_per_second": self._config.max_requests_per_second,
            "burst_capacity": self._config.burst_capacity,
            "global_limit": self._config.global_limit,
        }

        self._middleware = RateLimitingMiddleware(**middleware_config)
        self._logger.info("FastMCP rate limiting provider initialized")

    async def _shutdown_provider(self) -> None:
        """Shutdown rate limiting provider."""
        self._middleware = None
        self._rate_limit_stats = {"total_requests": 0, "rejected_requests": 0, "current_rate": 0.0}
        self._logger.info("FastMCP rate limiting provider shut down")

    async def _check_health(self) -> bool:
        """Check if rate limiting service is healthy."""
        return self._middleware is not None

    def set_rate_limit(self, requests_per_second: float) -> None:
        """Set the rate limit."""
        self._config.max_requests_per_second = requests_per_second
        if self._middleware:
            self._logger.info("Updated rate limit to %.2f requests/second", requests_per_second)

    def set_burst_capacity(self, capacity: int) -> None:
        """Set the burst capacity."""
        self._config.burst_capacity = capacity
        if self._middleware:
            self._logger.info("Updated burst capacity to %d", capacity)

    async def check_rate_limit(self, client_id: str | None = None) -> bool:
        """Check if request is within rate limit."""
        if not self._middleware:
            return True

        # Update stats
        self._rate_limit_stats["total_requests"] += 1

        # For this implementation, we'll assume the middleware handles the actual limiting
        # and we just track statistics here
        allowed = True  # Middleware would determine this

        if not allowed:
            self._rate_limit_stats["rejected_requests"] += 1
            self._logger.debug("Rate limit exceeded for client: %s", client_id or "global")

        return allowed

    async def get_rate_limit_stats(self) -> dict[str, Any]:
        """Get rate limiting statistics."""
        total = self._rate_limit_stats["total_requests"]
        rejected = self._rate_limit_stats["rejected_requests"]

        return {
            "total_requests": total,
            "rejected_requests": rejected,
            "acceptance_rate": (total - rejected) / max(total, 1),
            "rejection_rate": rejected / max(total, 1),
            "current_limit": self._config.max_requests_per_second,
            "burst_capacity": self._config.burst_capacity,
            "global_limit": self._config.global_limit,
        }

    def get_middleware_instance(self) -> RateLimitingMiddleware | None:
        """Get the FastMCP middleware instance."""
        return self._middleware

    async def create_service_context(
        self, base_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create service context for middleware rate limiting operations."""
        context = base_context.copy() if base_context else {}

        # Add service reference and basic info
        context.update({
            "middleware_rate_limiting_service": self,
            "service_type": self.service_type,
            "provider_name": self.name,
            "provider_version": self.version,
        })

        # Add capabilities and configuration
        context.update({
            "capabilities": self.capabilities,
            "configuration": {
                "requests_per_second": self._config.requests_per_second,
                "burst_capacity": self._config.burst_capacity,
            },
            "middleware_available": self._middleware is not None,
        })

        # Add health status
        health = await self.health_check()
        context.update({
            "health_status": health.status,
            "service_healthy": health.status.name == "HEALTHY",
            "last_error": health.last_error,
        })

        # Add runtime statistics
        context.update({"statistics": await self.get_rate_limit_stats()})

        return context
