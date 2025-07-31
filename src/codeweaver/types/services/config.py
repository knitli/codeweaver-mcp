# sourcery skip: lambdas-should-be-short
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Service configuration types for CodeWeaver."""

from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from codeweaver.types.enums import ChunkingStrategy, PerformanceMode


class ServiceConfig(BaseModel):
    """Base configuration for all services."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    enabled: Annotated[bool, Field(description="Whether service is enabled")] = True
    provider: Annotated[str, Field(description="Service provider name")]
    priority: Annotated[int, Field(ge=0, le=100, description="Service priority")] = 50
    timeout: Annotated[float, Field(gt=0, description="Timeout in seconds")] = 30.0
    max_retries: Annotated[int, Field(ge=0, description="Max retry attempts")] = 3
    retry_delay: Annotated[float, Field(ge=0, description="Retry delay in seconds")] = 1.0
    health_check_interval: Annotated[float, Field(gt=0, description="Health check interval")] = 60.0
    tags: Annotated[list[str], Field(description="Service tags")] = Field(default_factory=list)
    metadata: Annotated[dict[str, Any], Field(description="Additional metadata")] = Field(
        default_factory=dict
    )


class ChunkingServiceConfig(ServiceConfig):
    """Configuration for chunking services."""

    provider: str = "fastmcp_chunking"
    max_chunk_size: Annotated[int, Field(gt=0, le=10000, description="Max chunk size")] = 1500
    min_chunk_size: Annotated[int, Field(gt=0, le=1000, description="Min chunk size")] = 50
    overlap_size: Annotated[int, Field(ge=0, description="Chunk overlap size")] = 100
    ast_grep_enabled: Annotated[bool, Field(description="Enable AST chunking")] = True
    fallback_strategy: Annotated[ChunkingStrategy, Field(description="Fallback strategy")] = (
        ChunkingStrategy.SIMPLE
    )
    performance_mode: Annotated[PerformanceMode, Field(description="Performance mode")] = (
        PerformanceMode.BALANCED
    )

    # Language-specific configurations
    language_configs: Annotated[
        dict[str, dict[str, Any]], Field(description="Language-specific configs")
    ] = Field(default_factory=dict)

    # Advanced chunking options
    respect_code_structure: Annotated[
        bool, Field(description="Respect code structure boundaries")
    ] = True
    preserve_comments: Annotated[bool, Field(description="Keep comments with code")] = True
    include_imports: Annotated[bool, Field(description="Include import statements")] = True


class FilteringServiceConfig(ServiceConfig):
    """Configuration for filtering services."""

    provider: str = "fastmcp_filtering"
    include_patterns: Annotated[list[str], Field(description="Include patterns")] = Field(
        default_factory=list
    )
    exclude_patterns: Annotated[list[str], Field(description="Exclude patterns")] = Field(
        default_factory=list
    )
    max_file_size: Annotated[int, Field(gt=0, description="Max file size in bytes")] = 1024 * 1024
    max_depth: Annotated[int | None, Field(ge=0, description="Max directory depth")] = None
    follow_symlinks: Annotated[bool, Field(description="Follow symlinks")] = False
    ignore_hidden: Annotated[bool, Field(description="Ignore hidden files")] = True
    use_gitignore: Annotated[bool, Field(description="Respect .gitignore")] = True
    parallel_scanning: Annotated[bool, Field(description="Enable parallel scanning")] = True
    max_concurrent_scans: Annotated[int, Field(gt=0, description="Max concurrent scans")] = 10

    # File type filtering
    allowed_extensions: Annotated[list[str], Field(description="Allowed file extensions")] = Field(
        default_factory=list
    )
    blocked_extensions: Annotated[list[str], Field(description="Blocked file extensions")] = Field(
        default_factory=list
    )

    # Directory filtering
    ignore_directories: Annotated[list[str], Field(description="Directories to ignore")] = Field(
        default_factory=lambda: [
            ".git",
            ".svn",
            ".hg",
            ".bzr",
            "node_modules",
            "__pycache__",
            ".pytest_cache",
            "build",
            "dist",
            "target",
            ".next",
            ".nuxt",
            ".venv",
            "venv",
            ".env",
        ]
    )


class ValidationServiceConfig(ServiceConfig):
    """Configuration for validation services."""

    provider: str = "default_validation"
    validation_level: Annotated[str, Field(description="Validation strictness level")] = "standard"
    max_errors_per_item: Annotated[
        int, Field(ge=0, description="Max errors per validation item")
    ] = 10
    stop_on_first_error: Annotated[bool, Field(description="Stop validation on first error")] = (
        False
    )
    parallel_validation: Annotated[bool, Field(description="Enable parallel validation")] = True
    max_concurrent_validations: Annotated[
        int, Field(gt=0, description="Max concurrent validations")
    ] = 5
    cache_results: Annotated[bool, Field(description="Cache validation results")] = True
    result_cache_ttl: Annotated[int, Field(gt=0, description="Result cache TTL in seconds")] = 3600

    # Rule configuration
    enable_syntax_validation: Annotated[bool, Field(description="Enable syntax validation")] = True
    enable_style_validation: Annotated[bool, Field(description="Enable style validation")] = False
    enable_security_validation: Annotated[bool, Field(description="Enable security validation")] = (
        True
    )
    enable_performance_validation: Annotated[
        bool, Field(description="Enable performance validation")
    ] = False

    # Custom rules
    custom_rules: Annotated[list[dict[str, Any]], Field(description="Custom validation rules")] = (
        Field(default_factory=list)
    )


class CacheServiceConfig(ServiceConfig):
    """Configuration for cache services."""

    provider: str = "memory_cache"
    max_size: Annotated[int, Field(gt=0, description="Maximum cache size in bytes")] = (
        100 * 1024 * 1024
    )  # 100MB
    max_items: Annotated[int, Field(gt=0, description="Maximum number of cached items")] = 10000
    default_ttl: Annotated[int, Field(gt=0, description="Default TTL in seconds")] = 3600
    eviction_policy: Annotated[str, Field(description="Cache eviction policy")] = "lru"
    persistence_enabled: Annotated[bool, Field(description="Enable cache persistence")] = False
    persistence_path: Annotated[Path | None, Field(description="Path for cache persistence")] = None
    compression_enabled: Annotated[bool, Field(description="Enable cache compression")] = False
    metrics_enabled: Annotated[bool, Field(description="Enable cache metrics")] = True

    # Cache partitioning
    enable_partitioning: Annotated[bool, Field(description="Enable cache partitioning")] = False
    partition_count: Annotated[int, Field(gt=0, description="Number of cache partitions")] = 4

    # Advanced options
    cleanup_interval: Annotated[int, Field(gt=0, description="Cleanup interval in seconds")] = 300
    stats_collection_interval: Annotated[
        int, Field(gt=0, description="Stats collection interval")
    ] = 60


class MonitoringServiceConfig(ServiceConfig):
    """Configuration for monitoring services."""

    provider: str = "default_monitoring"
    check_interval: Annotated[
        float, Field(gt=0, description="Health check interval in seconds")
    ] = 30.0
    alert_threshold: Annotated[
        float, Field(ge=0, le=1, description="Alert threshold for health")
    ] = 0.8
    enable_alerts: Annotated[bool, Field(description="Enable alerting")] = True
    enable_auto_recovery: Annotated[bool, Field(description="Enable automatic recovery")] = True
    max_recovery_attempts: Annotated[int, Field(ge=0, description="Maximum recovery attempts")] = 3
    recovery_delay: Annotated[float, Field(ge=0, description="Delay between recovery attempts")] = (
        10.0
    )

    # Monitoring targets
    monitor_performance: Annotated[bool, Field(description="Monitor performance metrics")] = True
    monitor_memory: Annotated[bool, Field(description="Monitor memory usage")] = True
    monitor_disk: Annotated[bool, Field(description="Monitor disk usage")] = False
    monitor_network: Annotated[bool, Field(description="Monitor network metrics")] = False


class MetricsServiceConfig(ServiceConfig):
    """Configuration for metrics services."""

    provider: str = "default_metrics"
    collection_interval: Annotated[
        float, Field(gt=0, description="Metrics collection interval")
    ] = 60.0
    retention_period: Annotated[
        int, Field(gt=0, description="Metrics retention period in seconds")
    ] = 86400  # 24 hours
    enable_aggregation: Annotated[bool, Field(description="Enable metrics aggregation")] = True
    aggregation_window: Annotated[int, Field(gt=0, description="Aggregation window in seconds")] = (
        300  # 5 minutes
    )

    # Export options
    enable_export: Annotated[bool, Field(description="Enable metrics export")] = False
    export_format: Annotated[str, Field(description="Export format")] = "json"
    export_path: Annotated[Path | None, Field(description="Export file path")] = None
    export_interval: Annotated[int, Field(gt=0, description="Export interval in seconds")] = 300

    # Metric types to collect
    collect_performance_metrics: Annotated[
        bool, Field(description="Collect performance metrics")
    ] = True
    collect_resource_metrics: Annotated[bool, Field(description="Collect resource metrics")] = True
    collect_business_metrics: Annotated[bool, Field(description="Collect business metrics")] = False


class LoggingServiceConfig(ServiceConfig):
    """Configuration for logging middleware service."""

    provider: str = "fastmcp_logging"
    log_level: Annotated[str, Field(description="Log level for requests")] = "INFO"
    include_payloads: Annotated[bool, Field(description="Include request/response payloads")] = (
        False
    )
    max_payload_length: Annotated[int, Field(gt=0, description="Max payload length to log")] = 1000
    structured_logging: Annotated[bool, Field(description="Enable structured logging")] = False
    log_performance_metrics: Annotated[bool, Field(description="Log performance metrics")] = True
    log_to_service_bridge: Annotated[bool, Field(description="Log to service bridge")] = True
    methods: Annotated[list[str] | None, Field(description="Methods to log (None = all)")] = None


class TimingServiceConfig(ServiceConfig):
    """Configuration for timing middleware service."""

    provider: str = "fastmcp_timing"
    log_level: Annotated[str, Field(description="Log level for timing info")] = "INFO"
    track_performance_metrics: Annotated[bool, Field(description="Track performance metrics")] = (
        True
    )
    expose_metrics_endpoint: Annotated[bool, Field(description="Expose metrics endpoint")] = True
    metric_aggregation_window: Annotated[
        int, Field(gt=0, description="Metric aggregation window")
    ] = 300


class ErrorHandlingServiceConfig(ServiceConfig):
    """Configuration for error handling middleware service."""

    provider: str = "fastmcp_error_handling"
    include_traceback: Annotated[bool, Field(description="Include traceback in errors")] = False
    transform_errors: Annotated[bool, Field(description="Transform errors to MCP format")] = True
    error_aggregation: Annotated[bool, Field(description="Enable error aggregation")] = True
    error_notification_enabled: Annotated[bool, Field(description="Enable error notifications")] = (
        False
    )
    max_error_history: Annotated[int, Field(gt=0, description="Max error history entries")] = 100


class RateLimitingServiceConfig(ServiceConfig):
    """Configuration for rate limiting middleware service."""

    provider: str = "fastmcp_rate_limiting"
    max_requests_per_second: Annotated[
        float, Field(gt=0, description="Max requests per second")
    ] = 1.0
    burst_capacity: Annotated[int, Field(gt=0, description="Burst capacity")] = 10
    global_limit: Annotated[bool, Field(description="Apply limit globally")] = True
    expose_rate_limit_status: Annotated[bool, Field(description="Expose rate limit status")] = True
    rate_limit_metrics: Annotated[bool, Field(description="Track rate limit metrics")] = True


class TelemetryServiceConfig(ServiceConfig):
    """Configuration for telemetry services."""

    provider: str = "posthog_telemetry"

    # Privacy and opt-out settings
    enabled: bool = True
    anonymous_tracking: Annotated[bool, Field(description="Use anonymous tracking")] = True
    collect_sensitive_data: Annotated[
        bool, Field(description="Allow sensitive data collection")
    ] = False
    user_consent_required: Annotated[bool, Field(description="Require explicit user consent")] = (
        False
    )

    # PostHog configuration
    api_key: Annotated[str | None, Field(description="PostHog API key")] = None
    host: Annotated[str, Field(description="PostHog instance host")] = "https://app.posthog.com"

    # Testing and development
    mock_mode: Annotated[bool, Field(description="Enable mock mode for testing")] = False

    # Data collection settings
    batch_size: Annotated[int, Field(ge=1, le=1000, description="Event batch size")] = 50
    flush_interval: Annotated[float, Field(gt=0, description="Flush interval in seconds")] = 30.0
    max_queue_size: Annotated[int, Field(ge=1, description="Maximum queue size")] = 1000

    # Event filtering
    track_indexing: Annotated[bool, Field(description="Track indexing operations")] = True
    track_search: Annotated[bool, Field(description="Track search operations")] = True
    track_errors: Annotated[bool, Field(description="Track error events")] = True
    track_performance: Annotated[bool, Field(description="Track performance metrics")] = True

    # Data sanitization
    hash_file_paths: Annotated[bool, Field(description="Hash file paths")] = True
    hash_repository_names: Annotated[bool, Field(description="Hash repository names")] = True
    sanitize_queries: Annotated[bool, Field(description="Sanitize search queries")] = True

    # Local storage
    local_storage_enabled: Annotated[bool, Field(description="Enable local event storage")] = True
    local_storage_path: Annotated[Path | None, Field(description="Local storage path")] = None
    max_local_events: Annotated[int, Field(ge=0, description="Max local events to store")] = 10000


class IntentServiceConfig(ServiceConfig):
    """Configuration for intent orchestration services."""

    provider: str = "intent_orchestrator"
    default_strategy: Annotated[str, Field(description="Default strategy")] = "adaptive"
    confidence_threshold: Annotated[
        float, Field(ge=0.0, le=1.0, description="Minimum confidence threshold")
    ] = 0.6
    max_execution_time: Annotated[float, Field(gt=0, description="Maximum execution time")] = 30.0
    debug_mode: Annotated[bool, Field(description="Enable debug mode")] = False
    cache_ttl: Annotated[int, Field(gt=0, description="Cache TTL in seconds")] = 3600

    # Parser configuration
    use_nlp_fallback: Annotated[bool, Field(description="Enable NLP fallback parser")] = False
    pattern_matching: Annotated[bool, Field(description="Enable pattern matching")] = True

    # Strategy configuration
    enable_strategy_performance_tracking: Annotated[
        bool, Field(description="Enable strategy performance tracking")
    ] = True
    strategy_selection_timeout: Annotated[
        float, Field(gt=0, description="Strategy selection timeout")
    ] = 5.0

    # Workflow configuration
    max_workflow_steps: Annotated[int, Field(gt=0, description="Maximum workflow steps")] = 10
    workflow_step_timeout: Annotated[float, Field(gt=0, description="Per-step timeout")] = 15.0

    # Performance optimization settings
    circuit_breaker_enabled: Annotated[bool, Field(description="Enable circuit breaker")] = True
    circuit_breaker_threshold: Annotated[
        int, Field(ge=1, description="Circuit breaker failure threshold")
    ] = 5
    circuit_breaker_reset_time: Annotated[
        float, Field(gt=0, description="Circuit breaker reset time in seconds")
    ] = 60.0
    
    # Monitoring and metrics integration
    enable_performance_monitoring: Annotated[
        bool, Field(description="Enable comprehensive performance monitoring")
    ] = True
    enable_telemetry_tracking: Annotated[
        bool, Field(description="Enable telemetry tracking for intents")
    ] = True
    enable_metrics_collection: Annotated[
        bool, Field(description="Enable detailed metrics collection")
    ] = True
    
    # Concurrency settings
    max_concurrent_intents: Annotated[
        int, Field(gt=0, description="Maximum concurrent intent processing")
    ] = 10
    
    # Performance thresholds
    performance_excellent_threshold: Annotated[
        float, Field(gt=0, description="Excellent performance threshold in seconds")
    ] = 0.1
    performance_good_threshold: Annotated[
        float, Field(gt=0, description="Good performance threshold in seconds")
    ] = 0.5
    performance_acceptable_threshold: Annotated[
        float, Field(gt=0, description="Acceptable performance threshold in seconds")
    ] = 2.0


class AutoIndexingConfig(ServiceConfig):
    """Configuration for background auto-indexing services."""

    provider: str = "auto_indexing"
    watch_patterns: Annotated[list[str], Field(description="File patterns to watch")] = Field(
        default_factory=lambda: ["**/*.py", "**/*.js", "**/*.ts", "**/*.jsx", "**/*.tsx"]
    )
    ignore_patterns: Annotated[list[str], Field(description="Patterns to ignore")] = Field(
        default_factory=lambda: [".git", "node_modules", "__pycache__", "*.pyc", ".DS_Store"]
    )
    debounce_delay: Annotated[float, Field(ge=0, description="Debounce delay for file changes")] = (
        1.0
    )
    max_file_size: Annotated[int, Field(gt=0, description="Maximum file size to index")] = (
        1048576  # 1MB
    )

    # Monitoring configuration
    initial_scan_enabled: Annotated[bool, Field(description="Enable initial path scanning")] = True
    recursive_monitoring: Annotated[bool, Field(description="Enable recursive monitoring")] = True
    watch_for_new_directories: Annotated[bool, Field(description="Watch for new directories")] = (
        True
    )

    # Performance settings
    max_concurrent_indexing: Annotated[int, Field(gt=0, description="Max concurrent indexing")] = 5
    indexing_batch_size: Annotated[int, Field(gt=0, description="Indexing batch size")] = 10
    indexing_queue_size: Annotated[int, Field(gt=0, description="Max indexing queue size")] = 1000

    # Error handling
    max_indexing_failures: Annotated[int, Field(ge=0, description="Max indexing failures")] = 10
    failure_retry_delay: Annotated[float, Field(ge=0, description="Retry delay after failure")] = (
        5.0
    )


class ServicesConfig(BaseModel):
    """Root configuration for all services."""

    # Core services
    chunking: Annotated[ChunkingServiceConfig, Field(description="Chunking config")] = Field(
        default_factory=ChunkingServiceConfig
    )
    filtering: Annotated[FilteringServiceConfig, Field(description="Filtering config")] = Field(
        default_factory=FilteringServiceConfig
    )

    # Middleware services
    logging: Annotated[LoggingServiceConfig, Field(description="Logging middleware config")] = (
        Field(default_factory=LoggingServiceConfig)
    )
    timing: Annotated[TimingServiceConfig, Field(description="Timing middleware config")] = Field(
        default_factory=TimingServiceConfig
    )
    error_handling: Annotated[
        ErrorHandlingServiceConfig, Field(description="Error handling config")
    ] = Field(default_factory=ErrorHandlingServiceConfig)
    rate_limiting: Annotated[
        RateLimitingServiceConfig, Field(description="Rate limiting config")
    ] = Field(default_factory=RateLimitingServiceConfig)

    # Optional services
    validation: Annotated[ValidationServiceConfig, Field(description="Validation config")] = Field(
        default_factory=ValidationServiceConfig
    )
    cache: Annotated[CacheServiceConfig, Field(description="Cache config")] = Field(
        default_factory=CacheServiceConfig
    )
    monitoring: Annotated[MonitoringServiceConfig, Field(description="Monitoring config")] = Field(
        default_factory=MonitoringServiceConfig
    )
    metrics: Annotated[MetricsServiceConfig, Field(description="Metrics config")] = Field(
        default_factory=MetricsServiceConfig
    )
    telemetry: Annotated[TelemetryServiceConfig, Field(description="Telemetry config")] = Field(
        default_factory=TelemetryServiceConfig
    )

    # Intent layer services
    intent: Annotated[IntentServiceConfig, Field(description="Intent orchestration config")] = (
        Field(default_factory=IntentServiceConfig)
    )
    auto_indexing: Annotated[AutoIndexingConfig, Field(description="Auto-indexing config")] = Field(
        default_factory=AutoIndexingConfig
    )

    # Global service settings
    global_timeout: Annotated[float, Field(gt=0, description="Global timeout")] = 300.0
    health_check_enabled: Annotated[bool, Field(description="Enable health checks")] = True
    metrics_enabled: Annotated[bool, Field(description="Enable metrics")] = True
    auto_recovery: Annotated[bool, Field(description="Enable auto recovery")] = True

    # Service discovery and registration
    enable_service_discovery: Annotated[bool, Field(description="Enable service discovery")] = False
    service_discovery_interval: Annotated[float, Field(gt=0, description="Discovery interval")] = (
        60.0
    )

    # Middleware management settings
    middleware_auto_registration: Annotated[
        bool, Field(description="Auto-register middleware with FastMCP server")
    ] = True
    middleware_initialization_order: Annotated[
        list[str], Field(description="Order for initializing middleware services")
    ] = Field(default_factory=lambda: ["error_handling", "rate_limiting", "logging", "timing"])

    # Performance settings
    max_concurrent_services: Annotated[int, Field(gt=0, description="Max concurrent services")] = 10
    service_startup_timeout: Annotated[
        float, Field(gt=0, description="Service startup timeout")
    ] = 30.0
    service_shutdown_timeout: Annotated[
        float, Field(gt=0, description="Service shutdown timeout")
    ] = 10.0
