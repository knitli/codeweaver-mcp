"""Entrypoint for CodeWeaver middleware."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TelemetryService(Protocol):
    """Protocol for telemetry services."""

    def set_enabled(self, *, enabled: bool) -> None:
        """Enable or disable telemetry collection."""
        ...

    def set_anonymous(self, *, anonymous: bool) -> None:
        """Set anonymous tracking mode."""
        ...

    async def track_event(
        self,
        event_name: str,
        properties: dict[str, Any] | None = None,
        *,
        user_id: str | None = None,
    ) -> None:
        """Track a single event."""
        ...

    async def track_indexing(
        self,
        *,
        repository_path: str,
        file_count: int,
        language_distribution: dict[str, int],
        indexing_time: float,
        success: bool,
        error_message: str | None = None,
    ) -> None:
        """Track codebase indexing operations."""
        ...

    async def track_search(
        self,
        *,
        query_type: str,  # "semantic" or "ast_grep"
        result_count: int,
        search_time: float,
        query_complexity: str | None = None,
        filters_used: list[str] | None = None,
        success: bool,
        error_message: str | None = None,
    ) -> None:
        """Track search operations."""
        ...

    async def track_error(
        self,
        *,
        error_type: str,
        error_category: str,
        operation: str,
        error_message: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Track error events."""
        ...

    async def track_performance(
        self,
        *,
        operation: str,
        duration: float,
        resource_usage: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Track performance metrics."""
        ...

    async def flush(self) -> None:
        """Flush pending events to backend."""
        ...

    async def get_telemetry_stats(self) -> dict[str, Any]:
        """Get telemetry collection statistics."""
        ...

    def get_privacy_info(self) -> dict[str, Any]:
        """Get privacy and data collection information."""
        ...
