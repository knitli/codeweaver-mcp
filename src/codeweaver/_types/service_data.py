# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Service data structures and health monitoring types."""

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, Field

from codeweaver._types import BaseEnum, ServiceType


# Health monitoring types
class HealthStatus(BaseEnum):
    """Health status of a service or system."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ServiceHealth(BaseModel):
    """Health status of a service."""

    service_type: Annotated[ServiceType, Field(description="Type of service")]
    status: Annotated[HealthStatus, Field(description="Current health status")]
    last_check: Annotated[datetime, Field(description="Last health check timestamp")]
    response_time: Annotated[float, Field(ge=0, description="Last response time in seconds")]
    error_count: Annotated[int, Field(ge=0, description="Number of recent errors")] = 0
    success_rate: Annotated[float, Field(ge=0, le=1, description="Success rate (0-1)")] = 1.0
    last_error: Annotated[str | None, Field(description="Last error message")] = None
    uptime: Annotated[float, Field(ge=0, description="Service uptime in seconds")] = 0.0
    memory_usage: Annotated[int, Field(ge=0, description="Memory usage in bytes")] = 0


class ServicesHealthReport(BaseModel):
    """Comprehensive health report for all services."""

    overall_status: Annotated[HealthStatus, Field(description="Overall system health")]
    services: Annotated[
        dict[ServiceType, ServiceHealth], Field(description="Individual service health")
    ]
    check_time: Annotated[datetime, Field(description="Health check timestamp")]
    metrics: Annotated[dict[str, Any], Field(description="Additional health metrics")]


# Statistics types
class ChunkingStats(BaseModel):
    """Statistics for chunking operations."""

    total_files_processed: Annotated[int, Field(ge=0, description="Total files processed")] = 0
    total_chunks_created: Annotated[int, Field(ge=0, description="Total chunks created")] = 0
    average_chunk_size: Annotated[float, Field(ge=0, description="Average chunk size")] = 0.0
    total_processing_time: Annotated[float, Field(ge=0, description="Total processing time")] = 0.0
    languages_processed: Annotated[dict[str, int], Field(description="Files by language")] = Field(
        default_factory=dict
    )
    error_count: Annotated[int, Field(ge=0, description="Number of errors")] = 0
    success_rate: Annotated[float, Field(ge=0, le=1, description="Success rate")] = 1.0


class FilteringStats(BaseModel):
    """Statistics for filtering operations."""

    total_files_scanned: Annotated[int, Field(ge=0, description="Total files scanned")] = 0
    total_files_included: Annotated[int, Field(ge=0, description="Total files included")] = 0
    total_files_excluded: Annotated[int, Field(ge=0, description="Total files excluded")] = 0
    total_directories_scanned: Annotated[
        int, Field(ge=0, description="Total directories scanned")
    ] = 0
    total_scan_time: Annotated[float, Field(ge=0, description="Total scan time")] = 0.0
    patterns_matched: Annotated[dict[str, int], Field(description="Files matched by pattern")] = (
        Field(default_factory=dict)
    )
    error_count: Annotated[int, Field(ge=0, description="Number of errors")] = 0


class ValidationStats(BaseModel):
    """Statistics for validation operations."""

    total_validations: Annotated[int, Field(ge=0, description="Total validations performed")] = 0
    total_passed: Annotated[int, Field(ge=0, description="Total validations passed")] = 0
    total_failed: Annotated[int, Field(ge=0, description="Total validations failed")] = 0
    total_warnings: Annotated[int, Field(ge=0, description="Total warnings generated")] = 0
    average_validation_time: Annotated[
        float, Field(ge=0, description="Average validation time")
    ] = 0.0
    rules_triggered: Annotated[dict[str, int], Field(description="Rules triggered count")] = Field(
        default_factory=dict
    )
    error_count: Annotated[int, Field(ge=0, description="Number of errors")] = 0


class CacheStats(BaseModel):
    """Statistics for cache operations."""

    total_gets: Annotated[int, Field(ge=0, description="Total get operations")] = 0
    total_sets: Annotated[int, Field(ge=0, description="Total set operations")] = 0
    total_deletes: Annotated[int, Field(ge=0, description="Total delete operations")] = 0
    cache_hits: Annotated[int, Field(ge=0, description="Cache hits")] = 0
    cache_misses: Annotated[int, Field(ge=0, description="Cache misses")] = 0
    hit_rate: Annotated[float, Field(ge=0, le=1, description="Cache hit rate")] = 0.0
    total_size: Annotated[int, Field(ge=0, description="Total cache size in bytes")] = 0
    item_count: Annotated[int, Field(ge=0, description="Number of cached items")] = 0
    evictions: Annotated[int, Field(ge=0, description="Number of evictions")] = 0


# File metadata
class FileMetadata(BaseModel):
    """Metadata for a file."""

    path: Annotated[Path, Field(description="File path")]
    size: Annotated[int, Field(ge=0, description="File size in bytes")]
    modified_time: Annotated[datetime | None, Field(description="Last modified time")] = None
    created_time: Annotated[datetime | None, Field(description="Creation time")] = None
    file_type: Annotated[str, Field(description="File type/extension")] = "unknown"
    permissions: Annotated[str, Field(description="File permissions")] = ""
    is_binary: Annotated[bool, Field(description="Whether file is binary")] = False


class DirectoryStats(BaseModel):
    """Statistics for a directory tree."""

    total_files: Annotated[int, Field(ge=0, description="Total number of files")]
    total_directories: Annotated[int, Field(ge=0, description="Total number of directories")]
    total_size: Annotated[int, Field(ge=0, description="Total size in bytes")]
    file_types: Annotated[dict[str, int], Field(description="File types and counts")] = Field(
        default_factory=dict
    )
    largest_file: Annotated[Path | None, Field(description="Path to largest file")] = None
    scan_time: Annotated[float, Field(ge=0, description="Time taken to scan")] = 0.0


# Validation types
class ValidationSeverity(BaseEnum):
    """Severity levels for validation errors and warnings."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationLevel(BaseEnum):
    """Validation levels for configuration and data validation."""

    STRICT = "strict"
    STANDARD = "standard"
    RELAXED = "relaxed"
    DISABLED = "disabled"


class ValidationError(BaseModel):
    """A validation error."""

    message: Annotated[str, Field(description="Error message")]
    severity: Annotated[ValidationSeverity, Field(description="Error severity")]
    location: Annotated[str | None, Field(description="Location of error")] = None
    code: Annotated[str | None, Field(description="Error code")] = None


class ValidationWarning(BaseModel):
    """A validation warning."""

    message: Annotated[str, Field(description="Warning message")]
    location: Annotated[str | None, Field(description="Location of warning")] = None
    suggestion: Annotated[str | None, Field(description="Suggested fix")] = None


class ValidationResult(BaseModel):
    """Result of a validation operation."""

    is_valid: Annotated[bool, Field(description="Whether validation passed")]
    errors: Annotated[list[ValidationError], Field(description="List of validation errors")]
    warnings: Annotated[list[ValidationWarning], Field(description="List of validation warnings")]
    metadata: Annotated[dict[str, Any], Field(description="Additional validation metadata")]
    validation_time: Annotated[float, Field(ge=0, description="Time taken for validation")]
    rules_applied: Annotated[list[str], Field(description="List of rule IDs applied")]


class ValidationRule(BaseModel):
    """A validation rule configuration."""

    id: Annotated[str, Field(description="Unique rule identifier")]
    name: Annotated[str, Field(description="Human-readable rule name")]
    description: Annotated[str, Field(description="Rule description")]
    severity: Annotated[ValidationSeverity, Field(description="Rule severity level")]
    enabled: Annotated[bool, Field(description="Whether rule is enabled")] = True
    parameters: Annotated[dict[str, Any], Field(description="Rule parameters")] = Field(
        default_factory=dict
    )
    tags: Annotated[list[str], Field(description="Rule tags for categorization")] = Field(
        default_factory=list
    )


# Service provider information
class ServiceCapabilities(BaseModel):
    """Capabilities of a service provider."""

    supports_streaming: Annotated[bool, Field(description="Supports streaming operations")] = False
    supports_batch: Annotated[bool, Field(description="Supports batch operations")] = True
    supports_async: Annotated[bool, Field(description="Supports async operations")] = True
    max_concurrency: Annotated[int, Field(ge=1, description="Maximum concurrent operations")] = 10
    memory_usage: Annotated[str, Field(description="Expected memory usage category")] = "medium"
    performance_profile: Annotated[str, Field(description="Performance characteristics")] = (
        "standard"
    )


class ProviderStatus(BaseEnum):
    """Status of a service provider."""

    REGISTERED = "registered"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    DEPRECATED = "deprecated"


class ServiceProviderInfo(BaseModel):
    """Information about a service provider."""

    name: Annotated[str, Field(description="Provider name")]
    version: Annotated[str, Field(description="Provider version")]
    capabilities: Annotated[ServiceCapabilities, Field(description="Provider capabilities")]
    configuration_schema: Annotated[dict[str, Any], Field(description="Configuration schema")]
    status: Annotated[ProviderStatus, Field(description="Provider status")]
    created_at: Annotated[datetime, Field(description="Creation timestamp")]
    last_modified: Annotated[datetime, Field(description="Last modification timestamp")]


class ServiceInstanceInfo(BaseModel):
    """Information about a service instance."""

    service_type: Annotated[ServiceType, Field(description="Type of service")]
    provider_name: Annotated[str, Field(description="Provider name")]
    instance_id: Annotated[str, Field(description="Instance identifier")]
    status: Annotated[str, Field(description="Instance status")]
    created_at: Annotated[datetime, Field(description="Creation timestamp")]
    last_health_check: Annotated[datetime | None, Field(description="Last health check")] = None
    config_hash: Annotated[str | None, Field(description="Configuration hash")] = None


class ServiceRegistryHealth(BaseModel):
    """Health status of the service registry."""

    total_services: Annotated[int, Field(ge=0, description="Total registered services")]
    healthy_services: Annotated[int, Field(ge=0, description="Number of healthy services")]
    total_providers: Annotated[int, Field(ge=0, description="Total registered providers")]
    last_check: Annotated[datetime, Field(description="Last health check timestamp")]
    issues: Annotated[list[str], Field(description="List of issues found")] = Field(
        default_factory=list
    )
