# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Service protocol interfaces for CodeWeaver."""

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from codeweaver._types.content import CodeChunk
from codeweaver._types.data_structures import ContentItem
from codeweaver._types.enums import ChunkingStrategy, Language
from codeweaver._types.service_data import (
    CacheStats,
    ChunkingStats,
    DirectoryStats,
    FileMetadata,
    FilteringStats,
    ServiceHealth,
    ValidationResult,
    ValidationRule,
)


if TYPE_CHECKING:
    from codeweaver._types.config import ServiceType


@runtime_checkable
class ServiceProvider(Protocol):
    """Base protocol for all service providers."""

    @property
    def name(self) -> str:
        """Provider name identifier."""
        ...

    @property
    def version(self) -> str:
        """Provider version."""
        ...

    async def initialize(self) -> None:
        """Initialize the service provider."""
        ...

    async def shutdown(self) -> None:
        """Shutdown the service provider gracefully."""
        ...

    async def health_check(self) -> ServiceHealth:
        """Check if the service is healthy."""
        ...


@runtime_checkable
class ChunkingService(Protocol):
    """Protocol for content chunking services."""

    async def chunk_content(
        self,
        content: str,
        file_path: Path,
        metadata: dict[str, Any] | None = None,
        strategy: ChunkingStrategy = ChunkingStrategy.AUTO,
    ) -> list[CodeChunk]:
        """Chunk content into code segments."""
        ...

    async def chunk_content_stream(
        self,
        content: str,
        file_path: Path,
        metadata: dict[str, Any] | None = None,
        strategy: ChunkingStrategy = ChunkingStrategy.AUTO,
    ) -> AsyncGenerator[CodeChunk]:
        """Stream chunks for large files."""
        ...

    def detect_language(self, file_path: Path, content: str | None = None) -> Language | None:
        """Detect programming language."""
        ...

    def get_supported_languages(self) -> dict[Language, dict[str, Any]]:
        """Get supported languages and capabilities."""
        ...

    def get_language_config(self, language: Language) -> dict[str, Any] | None:
        """Get configuration for a specific language."""
        ...

    def get_available_strategies(self) -> dict[ChunkingStrategy, dict[str, Any]]:
        """Get all available chunking strategies."""
        ...

    def validate_chunk_size(self, size: int, language: Language = None) -> bool:
        """Validate if a chunk size is appropriate."""
        ...

    async def get_chunking_stats(self) -> ChunkingStats:
        """Get chunking performance statistics."""
        ...

    async def reset_stats(self) -> None:
        """Reset chunking statistics."""
        ...


@runtime_checkable
class FilteringService(Protocol):
    """Protocol for content filtering services."""

    async def discover_files(
        self,
        base_path: Path,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        max_depth: int | None = None,
        *,
        follow_symlinks: bool = False,
    ) -> list[Path]:
        """Discover files matching criteria."""
        ...

    async def discover_files_stream(
        self,
        base_path: Path,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        max_depth: int | None = None,
        *,
        follow_symlinks: bool = False,
    ) -> AsyncGenerator[Path]:
        """Stream file discovery."""
        ...

    def should_include_file(
        self,
        file_path: Path,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> bool:
        """Check if file should be included."""
        ...

    def should_include_directory(
        self, dir_path: Path, exclude_patterns: list[str] | None = None
    ) -> bool:
        """Determine if a directory should be traversed."""
        ...

    async def get_file_metadata(self, file_path: Path) -> FileMetadata:
        """Get file metadata."""
        ...

    async def get_directory_stats(self, dir_path: Path) -> DirectoryStats:
        """Get statistics for a directory tree."""
        ...

    def add_include_pattern(self, pattern: str) -> None:
        """Add an include pattern to the service."""
        ...

    def add_exclude_pattern(self, pattern: str) -> None:
        """Add an exclude pattern to the service."""
        ...

    def remove_pattern(self, pattern: str, pattern_type: str) -> None:
        """Remove a pattern from the service."""
        ...

    def get_active_patterns(self) -> dict[str, Any]:
        """Get currently active patterns."""
        ...

    async def get_filtering_stats(self) -> FilteringStats:
        """Get filtering performance statistics."""
        ...

    async def reset_stats(self) -> None:
        """Reset filtering statistics."""
        ...


@runtime_checkable
class ValidationService(Protocol):
    """Protocol for content validation services."""

    async def validate_content(
        self, content: ContentItem, rules: list[ValidationRule] | None = None
    ) -> ValidationResult:
        """Validate a content item against rules."""
        ...

    async def validate_chunk(
        self, chunk: CodeChunk, rules: list[ValidationRule] | None = None
    ) -> ValidationResult:
        """Validate a code chunk against rules."""
        ...

    async def validate_batch(
        self, items: list[ContentItem | CodeChunk], rules: list[ValidationRule] | None = None
    ) -> list[ValidationResult]:
        """Validate multiple items in batch."""
        ...

    def add_validation_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule to the service."""
        ...

    def remove_validation_rule(self, rule_id: str) -> None:
        """Remove a validation rule from the service."""
        ...

    def get_validation_rules(self) -> list[ValidationRule]:
        """Get all active validation rules."""
        ...

    def get_rule_by_id(self, rule_id: str) -> ValidationRule | None:
        """Get a specific validation rule by ID."""
        ...

    def set_validation_level(self, level: str) -> None:
        """Set the validation strictness level."""
        ...

    def get_validation_level(self) -> str:
        """Get the current validation level."""
        ...

    async def get_validation_stats(self) -> dict[str, Any]:
        """Get validation performance statistics."""
        ...

    async def reset_stats(self) -> None:
        """Reset validation statistics."""
        ...


@runtime_checkable
class CacheService(Protocol):
    """Protocol for caching services."""

    async def get(self, key: str) -> Any | None:
        """Get a value from cache."""
        ...

    async def set(
        self, key: str, value: Any, ttl: int | None = None, tags: list[str] | None = None
    ) -> None:
        """Set a value in cache."""
        ...

    async def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        ...

    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        ...

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values from cache."""
        ...

    async def set_many(
        self, items: dict[str, Any], ttl: int | None = None, tags: list[str] | None = None
    ) -> None:
        """Set multiple values in cache."""
        ...

    async def delete_many(self, keys: list[str]) -> int:
        """Delete multiple values from cache."""
        ...

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching a pattern."""
        ...

    async def invalidate_tags(self, tags: list[str]) -> int:
        """Invalidate all keys with given tags."""
        ...

    async def clear(self) -> None:
        """Clear all cached data."""
        ...

    async def get_cache_stats(self) -> CacheStats:
        """Get cache statistics."""
        ...

    async def get_cache_info(self) -> dict[str, Any]:
        """Get cache configuration information."""
        ...


@runtime_checkable
class MonitoringService(Protocol):
    """Protocol for service monitoring."""

    async def start_monitoring(self) -> None:
        """Start monitoring services."""
        ...

    async def stop_monitoring(self) -> None:
        """Stop monitoring services."""
        ...

    async def add_service(self, service_type: "ServiceType", service: Any) -> None:
        """Add a service to monitoring."""
        ...

    async def remove_service(self, service_type: "ServiceType") -> None:
        """Remove a service from monitoring."""
        ...

    async def get_health_status(self) -> dict["ServiceType", ServiceHealth]:
        """Get health status of all monitored services."""
        ...

    async def check_service_health(self, service_type: "ServiceType") -> ServiceHealth:
        """Check health of a specific service."""
        ...

    def set_health_check_interval(self, interval_seconds: float) -> None:
        """Set the health check interval."""
        ...

    def get_health_check_interval(self) -> float:
        """Get the current health check interval."""
        ...


@runtime_checkable
class MetricsService(Protocol):
    """Protocol for metrics collection."""

    async def record_metric(
        self, name: str, value: float, tags: dict[str, str] | None = None
    ) -> None:
        """Record a metric value."""
        ...

    async def increment_counter(
        self, name: str, value: int = 1, tags: dict[str, str] | None = None
    ) -> None:
        """Increment a counter metric."""
        ...

    async def record_timing(
        self, name: str, duration_ms: float, tags: dict[str, str] | None = None
    ) -> None:
        """Record a timing metric."""
        ...

    async def get_metrics(
        self, name_pattern: str | None = None, tags: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Get collected metrics."""
        ...

    async def reset_metrics(self, name_pattern: str | None = None) -> None:
        """Reset metrics matching pattern."""
        ...

    def set_collection_interval(self, interval_seconds: float) -> None:
        """Set metrics collection interval."""
        ...

    def get_collection_interval(self) -> float:
        """Get current collection interval."""
        ...


# Middleware service protocols for FastMCP integration

@runtime_checkable
class LoggingService(Protocol):
    """Protocol for logging middleware service."""

    def set_log_level(self, level: str) -> None:
        """Set the logging level."""
        ...

    def set_include_payloads(self, *, include: bool) -> None:
        """Set whether to include payloads in logs."""
        ...

    def set_max_payload_length(self, length: int) -> None:
        """Set maximum payload length to log."""
        ...

    async def log_request(self, method: str, params: dict[str, Any]) -> None:
        """Log an incoming request."""
        ...

    async def log_response(self, method: str, result: Any, duration: float) -> None:
        """Log an outgoing response."""
        ...

    def get_middleware_instance(self) -> Any:
        """Get the FastMCP middleware instance."""
        ...


@runtime_checkable
class TimingService(Protocol):
    """Protocol for timing middleware service."""

    def set_track_metrics(self, *, track: bool) -> None:
        """Set whether to track performance metrics."""
        ...

    async def record_timing(self, method: str, duration: float) -> None:
        """Record timing for a method call."""
        ...

    async def get_timing_stats(self) -> dict[str, Any]:
        """Get timing statistics."""
        ...

    def get_middleware_instance(self) -> Any:
        """Get the FastMCP middleware instance."""
        ...


@runtime_checkable
class ErrorHandlingService(Protocol):
    """Protocol for error handling middleware service."""

    def set_include_traceback(self, *, include: bool) -> None:
        """Set whether to include traceback in errors."""
        ...

    def set_transform_errors(self, *, transform: bool) -> None:
        """Set whether to transform errors to MCP format."""
        ...

    async def handle_error(self, error: Exception, context: dict[str, Any]) -> Any:
        """Handle an error with the middleware."""
        ...

    async def get_error_stats(self) -> dict[str, Any]:
        """Get error handling statistics."""
        ...

    def get_middleware_instance(self) -> Any:
        """Get the FastMCP middleware instance."""
        ...


@runtime_checkable
class RateLimitingService(Protocol):
    """Protocol for rate limiting middleware service."""

    def set_rate_limit(self, requests_per_second: float) -> None:
        """Set the rate limit."""
        ...

    def set_burst_capacity(self, capacity: int) -> None:
        """Set the burst capacity."""
        ...

    async def check_rate_limit(self, client_id: str | None = None) -> bool:
        """Check if request is within rate limit."""
        ...

    async def get_rate_limit_stats(self) -> dict[str, Any]:
        """Get rate limiting statistics."""
        ...

    def get_middleware_instance(self) -> Any:
        """Get the FastMCP middleware instance."""
        ...
