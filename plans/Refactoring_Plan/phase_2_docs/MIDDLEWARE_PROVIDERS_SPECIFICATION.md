<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Middleware Service Providers Implementation Specification

## Overview

This specification defines the implementation of middleware service providers that wrap FastMCP's builtin middleware components while exposing them as services within CodeWeaver's services layer architecture.

## Architecture

### Directory Structure

```
src/codeweaver/services/providers/middleware/
├── __init__.py
├── base_middleware_provider.py
├── logging_provider.py
├── timing_provider.py
├── error_handling_provider.py
└── rate_limiting_provider.py
```

## Base Middleware Provider

**File**: `src/codeweaver/services/providers/middleware/base_middleware_provider.py`

```python
"""Base class for FastMCP middleware service providers."""

import logging
from abc import abstractmethod
from typing import Any, Dict

from codeweaver.cw_types import ServiceConfig
from codeweaver.cw_types import ServiceProvider
from codeweaver.services.providers.base_provider import BaseServiceProvider


class BaseMiddlewareProvider(BaseServiceProvider):
    """Base class for middleware service providers that wrap FastMCP middleware."""

    def __init__(self, config: ServiceConfig):
        """Initialize the middleware provider."""
        super().__init__(config)
        self._middleware_instance: Any = None
        self._metrics: Dict[str, Any] = {}
        self._logger = logging.getLogger(f"codeweaver.middleware.{self._config.provider}")

    @abstractmethod
    async def _create_middleware_instance(self) -> Any:
        """Create and configure the FastMCP middleware instance."""
        ...

    async def _initialize_provider(self) -> None:
        """Initialize the middleware provider."""
        try:
            self._middleware_instance = await self._create_middleware_instance()
            self._logger.info("Middleware provider %s initialized", self._config.provider)
        except Exception as e:
            self._logger.exception("Failed to initialize middleware provider %s", self._config.provider)
            raise

    def get_middleware_instance(self) -> Any:
        """Get the FastMCP middleware instance for server registration."""
        if not self._middleware_instance:
            raise RuntimeError(f"Middleware provider {self._config.provider} not initialized")
        return self._middleware_instance

    async def get_metrics(self) -> Dict[str, Any]:
        """Get middleware-specific metrics."""
        return {
            "provider": self._config.provider,
            "initialized": self._middleware_instance is not None,
            "health_status": (await self.health_check()).status.value,
            **self._metrics
        }

    def _update_metrics(self, metrics_update: Dict[str, Any]) -> None:
        """Update internal metrics tracking."""
        self._metrics.update(metrics_update)
```

## Logging Service Provider

**File**: `src/codeweaver/services/providers/middleware/logging_provider.py`

```python
"""Logging middleware service provider."""

import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Deque, Dict, List

from fastmcp.server.middleware.logging import LoggingMiddleware, StructuredLoggingMiddleware

from codeweaver.cw_types import LoggingServiceConfig
from codeweaver.cw_types import LoggingService
from codeweaver.services.providers.middleware.base_middleware_provider import BaseMiddlewareProvider


class FastMCPLoggingProvider(BaseMiddlewareProvider, LoggingService):
    """FastMCP logging middleware as a service."""

    def __init__(self, config: LoggingServiceConfig):
        """Initialize the logging provider."""
        super().__init__(config)
        self._config: LoggingServiceConfig = config
        self._log_history: Deque[Dict[str, Any]] = deque(maxlen=1000)
        self._log_metrics = defaultdict(int)

    async def _create_middleware_instance(self) -> LoggingMiddleware | StructuredLoggingMiddleware:
        """Create and configure the FastMCP logging middleware instance."""
        log_level = getattr(logging, self._config.log_level.upper(), logging.INFO)

        if self._config.structured_logging:
            middleware = StructuredLoggingMiddleware(
                log_level=log_level,
                include_payloads=self._config.include_payloads,
                max_payload_length=self._config.max_payload_length,
                methods=self._config.methods,
            )
        else:
            middleware = LoggingMiddleware(
                log_level=log_level,
                include_payloads=self._config.include_payloads,
                max_payload_length=self._config.max_payload_length,
                methods=self._config.methods,
            )

        # If log_to_service_bridge is enabled, wrap middleware to capture logs
        if self._config.log_to_service_bridge:
            middleware = self._wrap_middleware_for_capture(middleware)

        return middleware

    def _wrap_middleware_for_capture(self, middleware: LoggingMiddleware) -> LoggingMiddleware:
        """Wrap middleware to capture logs for service access."""
        original_on_call = middleware.on_call_tool

        async def enhanced_on_call(context, call_next):
            # Capture request info
            request_info = {
                "timestamp": datetime.now(UTC).isoformat(),
                "tool_name": getattr(context.message, "name", "unknown"),
                "request_id": id(context),
            }

            if self._config.include_payloads:
                request_info["payload"] = str(context.message)[:self._config.max_payload_length]

            self._log_history.append(request_info)
            self._log_metrics["requests_logged"] += 1

            # Continue with original middleware
            return await original_on_call(context, call_next)

        middleware.on_call_tool = enhanced_on_call
        return middleware

    async def log_request(self, request_data: Dict[str, Any], context: Dict[str, Any] | None = None) -> None:
        """Log a request with optional context."""
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "type": "service_request",
            "data": request_data,
            "context": context or {},
        }

        self._log_history.append(log_entry)
        self._log_metrics["service_requests_logged"] += 1

        self._logger.info("Service request logged: %s", request_data.get("operation", "unknown"))

    async def log_response(self, response_data: Dict[str, Any], context: Dict[str, Any] | None = None) -> None:
        """Log a response with optional context."""
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "type": "service_response",
            "data": response_data,
            "context": context or {},
        }

        self._log_history.append(log_entry)
        self._log_metrics["service_responses_logged"] += 1

        self._logger.info("Service response logged: %s", response_data.get("status", "unknown"))

    async def get_log_metrics(self) -> Dict[str, Any]:
        """Get logging metrics and statistics."""
        return {
            **await self.get_metrics(),
            "log_history_size": len(self._log_history),
            "requests_logged": self._log_metrics["requests_logged"],
            "service_requests_logged": self._log_metrics["service_requests_logged"],
            "service_responses_logged": self._log_metrics["service_responses_logged"],
            "structured_logging_enabled": self._config.structured_logging,
            "include_payloads": self._config.include_payloads,
        }

    async def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent log entries."""
        return list(self._log_history)[-limit:]
```

## Timing Service Provider

**File**: `src/codeweaver/services/providers/middleware/timing_provider.py`

```python
"""Timing middleware service provider."""

import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Deque, Dict

from fastmcp.server.middleware.timing import TimingMiddleware

from codeweaver.cw_types import TimingServiceConfig
from codeweaver.cw_types import TimingService
from codeweaver.services.providers.middleware.base_middleware_provider import BaseMiddlewareProvider


class FastMCPTimingProvider(BaseMiddlewareProvider, TimingService):
    """FastMCP timing middleware as a service."""

    def __init__(self, config: TimingServiceConfig):
        """Initialize the timing provider."""
        super().__init__(config)
        self._config: TimingServiceConfig = config
        self._active_timings: Dict[str, float] = {}
        self._timing_history: Deque[Dict[str, Any]] = deque(maxlen=1000)
        self._performance_metrics = defaultdict(list)
        self._operation_stats = defaultdict(lambda: {"count": 0, "total_time": 0.0, "avg_time": 0.0})

    async def _create_middleware_instance(self) -> TimingMiddleware:
        """Create and configure the FastMCP timing middleware instance."""
        log_level = getattr(logging, self._config.log_level.upper(), logging.INFO)
        middleware = TimingMiddleware(log_level=log_level)

        if self._config.track_performance_metrics:
            middleware = self._wrap_middleware_for_metrics(middleware)

        return middleware

    def _wrap_middleware_for_metrics(self, middleware: TimingMiddleware) -> TimingMiddleware:
        """Wrap middleware to capture timing metrics."""
        original_on_call = middleware.on_call_tool

        async def enhanced_on_call(context, call_next):
            start_time = time.time()
            tool_name = getattr(context.message, "name", "unknown")

            try:
                # Continue with original middleware
                result = await original_on_call(context, call_next)

                # Record successful operation
                end_time = time.time()
                duration = end_time - start_time

                self._record_operation_timing(tool_name, duration, "success")
                return result

            except Exception as e:
                # Record failed operation
                end_time = time.time()
                duration = end_time - start_time

                self._record_operation_timing(tool_name, duration, "error")
                raise

        middleware.on_call_tool = enhanced_on_call
        return middleware

    def _record_operation_timing(self, operation: str, duration: float, status: str) -> None:
        """Record timing for an operation."""
        timing_record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "operation": operation,
            "duration": duration,
            "status": status,
        }

        self._timing_history.append(timing_record)

        # Update operation statistics
        stats = self._operation_stats[operation]
        stats["count"] += 1
        stats["total_time"] += duration
        stats["avg_time"] = stats["total_time"] / stats["count"]

        # Update performance metrics
        if self._config.track_performance_metrics:
            self._performance_metrics[operation].append(duration)
            # Keep only recent metrics within window
            window_seconds = self._config.metric_aggregation_window
            cutoff_time = time.time() - window_seconds
            self._performance_metrics[operation] = [
                d for d in self._performance_metrics[operation]
                if d["timestamp"] > cutoff_time
            ]

    async def start_timing(self, operation_id: str) -> None:
        """Start timing an operation."""
        self._active_timings[operation_id] = time.time()
        self._logger.debug("Started timing operation: %s", operation_id)

    async def end_timing(self, operation_id: str) -> Dict[str, float]:
        """End timing and return metrics."""
        if operation_id not in self._active_timings:
            raise ValueError(f"No active timing for operation: {operation_id}")

        start_time = self._active_timings.pop(operation_id)
        end_time = time.time()
        duration = end_time - start_time

        self._record_operation_timing(operation_id, duration, "manual")

        self._logger.debug("Ended timing operation %s: %.3fs", operation_id, duration)

        return {
            "operation_id": operation_id,
            "duration": duration,
            "start_time": start_time,
            "end_time": end_time,
        }

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get aggregated performance metrics."""
        metrics = await self.get_metrics()

        # Add operation statistics
        metrics["operations"] = dict(self._operation_stats)
        metrics["active_timings"] = len(self._active_timings)
        metrics["timing_history_size"] = len(self._timing_history)

        # Add performance data if enabled
        if self._config.track_performance_metrics:
            perf_summary = {}
            for operation, timings in self._performance_metrics.items():
                if timings:
                    perf_summary[operation] = {
                        "count": len(timings),
                        "avg": sum(timings) / len(timings),
                        "min": min(timings),
                        "max": max(timings),
                    }
            metrics["performance_window"] = perf_summary

        return metrics
```

## Error Handling Service Provider

**File**: `src/codeweaver/services/providers/middleware/error_handling_provider.py`

```python
"""Error handling middleware service provider."""

from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Deque, Dict, List

from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware

from codeweaver.cw_types import ErrorHandlingServiceConfig
from codeweaver.cw_types import ErrorHandlingService
from codeweaver.services.providers.middleware.base_middleware_provider import BaseMiddlewareProvider


class FastMCPErrorHandlingProvider(BaseMiddlewareProvider, ErrorHandlingService):
    """FastMCP error handling middleware as a service."""

    def __init__(self, config: ErrorHandlingServiceConfig):
        """Initialize the error handling provider."""
        super().__init__(config)
        self._config: ErrorHandlingServiceConfig = config
        self._error_history: Deque[Dict[str, Any]] = deque(maxlen=config.max_error_history)
        self._error_stats = defaultdict(int)

    async def _create_middleware_instance(self) -> ErrorHandlingMiddleware:
        """Create and configure the FastMCP error handling middleware instance."""
        middleware = ErrorHandlingMiddleware(
            include_traceback=self._config.include_traceback,
            transform_errors=self._config.transform_errors,
        )

        if self._config.error_aggregation:
            middleware = self._wrap_middleware_for_aggregation(middleware)

        return middleware

    def _wrap_middleware_for_aggregation(self, middleware: ErrorHandlingMiddleware) -> ErrorHandlingMiddleware:
        """Wrap middleware to aggregate error information."""
        original_on_call = middleware.on_call_tool

        async def enhanced_on_call(context, call_next):
            try:
                return await original_on_call(context, call_next)
            except Exception as e:
                # Capture error for aggregation
                await self.handle_error(e, {
                    "tool_name": getattr(context.message, "name", "unknown"),
                    "request_id": id(context),
                    "timestamp": datetime.now(UTC).isoformat(),
                })
                # Re-raise the exception to maintain middleware behavior
                raise

        middleware.on_call_tool = enhanced_on_call
        return middleware

    async def handle_error(self, error: Exception, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Handle an error and return processed error info."""
        error_info = {
            "timestamp": datetime.now(UTC).isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
        }

        if self._config.include_traceback:
            import traceback
            error_info["traceback"] = traceback.format_exc()

        # Store in history
        self._error_history.append(error_info)

        # Update statistics
        self._error_stats["total_errors"] += 1
        self._error_stats[f"error_type_{error_info['error_type']}"] += 1

        # Log the error
        self._logger.error(
            "Error handled by service: %s - %s",
            error_info["error_type"],
            error_info["error_message"]
        )

        # Send notification if enabled
        if self._config.error_notification_enabled:
            await self._send_error_notification(error_info)

        return error_info

    async def _send_error_notification(self, error_info: Dict[str, Any]) -> None:
        """Send error notification (placeholder for future implementation)."""
        # This could integrate with monitoring systems, email, Slack, etc.
        self._logger.warning("Error notification would be sent: %s", error_info["error_type"])

    async def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and trends."""
        metrics = await self.get_metrics()

        # Add error-specific statistics
        metrics.update({
            "error_history_size": len(self._error_history),
            "total_errors": self._error_stats["total_errors"],
            "error_types": {
                key.replace("error_type_", ""): value
                for key, value in self._error_stats.items()
                if key.startswith("error_type_")
            },
            "error_aggregation_enabled": self._config.error_aggregation,
            "notification_enabled": self._config.error_notification_enabled,
        })

        return metrics

    async def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent error history."""
        return list(self._error_history)[-limit:]
```

## Rate Limiting Service Provider

**File**: `src/codeweaver/services/providers/middleware/rate_limiting_provider.py`

```python
"""Rate limiting middleware service provider."""

import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict

from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware

from codeweaver.cw_types import RateLimitingServiceConfig
from codeweaver.cw_types import RateLimitingService
from codeweaver.services.providers.middleware.base_middleware_provider import BaseMiddlewareProvider


class FastMCPRateLimitingProvider(BaseMiddlewareProvider, RateLimitingService):
    """FastMCP rate limiting middleware as a service."""

    def __init__(self, config: RateLimitingServiceConfig):
        """Initialize the rate limiting provider."""
        super().__init__(config)
        self._config: RateLimitingServiceConfig = config
        self._rate_limit_metrics = defaultdict(int)
        self._client_stats = defaultdict(lambda: {"requests": 0, "blocked": 0, "last_request": None})

    async def _create_middleware_instance(self) -> RateLimitingMiddleware:
        """Create and configure the FastMCP rate limiting middleware instance."""
        middleware = RateLimitingMiddleware(
            max_requests_per_second=self._config.max_requests_per_second,
            burst_capacity=self._config.burst_capacity,
            global_limit=self._config.global_limit,
        )

        if self._config.rate_limit_metrics:
            middleware = self._wrap_middleware_for_metrics(middleware)

        return middleware

    def _wrap_middleware_for_metrics(self, middleware: RateLimitingMiddleware) -> RateLimitingMiddleware:
        """Wrap middleware to capture rate limiting metrics."""
        original_on_call = middleware.on_call_tool

        async def enhanced_on_call(context, call_next):
            client_id = "global" if self._config.global_limit else str(id(context))
            tool_name = getattr(context.message, "name", "unknown")

            # Record request attempt
            self._client_stats[client_id]["requests"] += 1
            self._client_stats[client_id]["last_request"] = datetime.now(UTC).isoformat()
            self._rate_limit_metrics["total_requests"] += 1

            try:
                result = await original_on_call(context, call_next)
                self._rate_limit_metrics["successful_requests"] += 1
                return result
            except Exception as e:
                # Check if this was a rate limiting error
                if "rate limit" in str(e).lower():
                    self._client_stats[client_id]["blocked"] += 1
                    self._rate_limit_metrics["blocked_requests"] += 1
                raise

        middleware.on_call_tool = enhanced_on_call
        return middleware

    async def check_rate_limit(self, client_id: str, operation: str | None = None) -> bool:
        """Check if request is within rate limits."""
        # This is a simplified check - actual rate limiting is handled by FastMCP middleware
        # This method can be used for internal rate limit checks
        current_time = time.time()
        client_stats = self._client_stats[client_id]

        # Simple time-window based check
        if client_stats["last_request"]:
            last_request_time = datetime.fromisoformat(client_stats["last_request"]).timestamp()
            time_since_last = current_time - last_request_time
            min_interval = 1.0 / self._config.max_requests_per_second

            return time_since_last >= min_interval

        return True

    async def get_rate_limit_status(self, client_id: str | None = None) -> Dict[str, Any]:
        """Get current rate limiting status."""
        if client_id:
            return {
                "client_id": client_id,
                "stats": dict(self._client_stats[client_id]),
                "max_requests_per_second": self._config.max_requests_per_second,
                "burst_capacity": self._config.burst_capacity,
            }

        return {
            "global_stats": {
                "total_clients": len(self._client_stats),
                "total_requests": self._rate_limit_metrics["total_requests"],
                "blocked_requests": self._rate_limit_metrics["blocked_requests"],
                "success_rate": (
                    self._rate_limit_metrics["successful_requests"] /
                    max(self._rate_limit_metrics["total_requests"], 1)
                ),
            },
            "configuration": {
                "max_requests_per_second": self._config.max_requests_per_second,
                "burst_capacity": self._config.burst_capacity,
                "global_limit": self._config.global_limit,
            }
        }

    async def get_rate_limit_metrics(self) -> Dict[str, Any]:
        """Get rate limiting metrics."""
        metrics = await self.get_metrics()

        metrics.update({
            "rate_limit_stats": dict(self._rate_limit_metrics),
            "active_clients": len(self._client_stats),
            "configuration": {
                "max_requests_per_second": self._config.max_requests_per_second,
                "burst_capacity": self._config.burst_capacity,
                "global_limit": self._config.global_limit,
            }
        })

        return metrics
```

## Package Initialization

**File**: `src/codeweaver/services/providers/middleware/__init__.py`

```python
"""Middleware service providers for FastMCP integration."""

from codeweaver.services.providers.middleware.error_handling_provider import FastMCPErrorHandlingProvider
from codeweaver.services.providers.middleware.logging_provider import FastMCPLoggingProvider
from codeweaver.services.providers.middleware.rate_limiting_provider import FastMCPRateLimitingProvider
from codeweaver.services.providers.middleware.timing_provider import FastMCPTimingProvider

__all__ = [
    "FastMCPLoggingProvider",
    "FastMCPTimingProvider",
    "FastMCPErrorHandlingProvider",
    "FastMCPRateLimitingProvider",
]
```

## Testing Strategy

### Unit Tests

Each provider should have comprehensive unit tests covering:
- Initialization and configuration
- Middleware instance creation
- Service method functionality
- Metrics collection
- Error handling
- Health checks

### Integration Tests

Integration tests should verify:
- FastMCP middleware registration
- Service bridge integration
- Configuration loading
- End-to-end middleware functionality

### Performance Tests

Performance tests should ensure:
- No degradation from service wrapping
- Metrics collection efficiency
- Memory usage optimization

## Implementation Notes

1. **Dependency Injection**: Providers need access to FastMCP server instance for registration
2. **Error Handling**: Robust error handling to prevent middleware failure from affecting core functionality
3. **Metrics Efficiency**: Lightweight metrics collection to avoid performance impact
4. **Thread Safety**: Consider thread safety for metrics collection and state management
5. **Configuration Validation**: Validate configuration parameters during initialization
6. **Backward Compatibility**: Ensure existing middleware behavior is preserved
