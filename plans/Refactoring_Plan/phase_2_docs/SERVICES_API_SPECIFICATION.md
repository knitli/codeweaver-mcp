# Services API Specification

**CodeWeaver MCP Server - Services Layer API Reference**

*Document Version: 1.0*  
*API Version: 1.0.0*  
*Design Date: 2025-07-26*  
*Specification Type: Complete API Contract*

---

## API Overview

This specification defines the complete API contract for CodeWeaver's services layer, including service protocols, data structures, configuration schemas, and integration patterns.

**API Design Principles:**
- 🔒 **Type Safety**: Full type annotations with runtime checking
- 🧩 **Protocol-Based**: Duck typing through Python protocols
- ⚡ **Async First**: All I/O operations are async by default
- 📋 **Immutable Data**: Immutable data structures where possible
- 🛡️ **Error Handling**: Comprehensive error types and handling
- 📊 **Observable**: Built-in metrics and health monitoring

---

## Core Service Protocols

### ChunkingService Protocol

**Location:** `src/codeweaver/_types/services.py`

```python
from typing import Protocol, runtime_checkable, AsyncGenerator
from pathlib import Path
from codeweaver._types.data_structures import CodeChunk, ChunkingStats
from codeweaver._types.enums import Language, ChunkingStrategy

@runtime_checkable
class ChunkingService(Protocol):
    """Protocol for content chunking and language processing services."""
    
    # Core chunking operations
    async def chunk_content(
        self,
        content: str,
        file_path: Path,
        metadata: dict[str, Any] = None,
        strategy: ChunkingStrategy = ChunkingStrategy.AUTO
    ) -> list[CodeChunk]:
        """
        Chunk content into semantically meaningful code segments.
        
        Args:
            content: Raw file content to chunk
            file_path: Path to the source file
            metadata: Optional metadata about the content
            strategy: Chunking strategy to use
            
        Returns:
            List of code chunks with metadata
            
        Raises:
            ChunkingError: If chunking fails
            UnsupportedLanguageError: If language not supported
        """
        ...
    
    async def chunk_content_stream(
        self,
        content: str,
        file_path: Path,
        metadata: dict[str, Any] = None,
        strategy: ChunkingStrategy = ChunkingStrategy.AUTO
    ) -> AsyncGenerator[CodeChunk, None]:
        """
        Stream chunks as they are processed for large files.
        
        Args:
            content: Raw file content to chunk
            file_path: Path to the source file
            metadata: Optional metadata about the content
            strategy: Chunking strategy to use
            
        Yields:
            Code chunks as they are processed
            
        Raises:
            ChunkingError: If chunking fails
            UnsupportedLanguageError: If language not supported
        """
        ...
    
    # Language detection and capabilities
    def detect_language(self, file_path: Path, content: str = None) -> Language | None:
        """
        Detect the programming language of a file.
        
        Args:
            file_path: Path to the file
            content: Optional file content for detection
            
        Returns:
            Detected language or None if unknown
        """
        ...
    
    def get_supported_languages(self) -> dict[Language, LanguageCapabilities]:
        """
        Get all supported languages and their capabilities.
        
        Returns:
            Dictionary mapping languages to their capabilities
        """
        ...
    
    def get_language_config(self, language: Language) -> LanguageConfig | None:
        """
        Get configuration for a specific language.
        
        Args:
            language: Language to get config for
            
        Returns:
            Language configuration or None if not supported
        """
        ...
    
    # Chunking strategies and configuration
    def get_available_strategies(self) -> dict[ChunkingStrategy, StrategyInfo]:
        """
        Get all available chunking strategies.
        
        Returns:
            Dictionary mapping strategies to their information
        """
        ...
    
    def validate_chunk_size(self, size: int, language: Language = None) -> bool:
        """
        Validate if a chunk size is appropriate.
        
        Args:
            size: Chunk size to validate
            language: Optional language context
            
        Returns:
            True if size is valid
        """
        ...
    
    # Statistics and monitoring
    async def get_chunking_stats(self) -> ChunkingStats:
        """
        Get statistics about chunking performance.
        
        Returns:
            Current chunking statistics
        """
        ...
    
    async def reset_stats(self) -> None:
        """Reset chunking statistics."""
        ...
```

### FilteringService Protocol

```python
@runtime_checkable
class FilteringService(Protocol):
    """Protocol for content filtering and file discovery services."""
    
    # File discovery operations
    async def discover_files(
        self,
        base_path: Path,
        include_patterns: list[str] = None,
        exclude_patterns: list[str] = None,
        max_depth: int = None,
        follow_symlinks: bool = False
    ) -> list[Path]:
        """
        Discover files matching the given criteria.
        
        Args:
            base_path: Root directory to search
            include_patterns: Glob patterns for files to include
            exclude_patterns: Glob patterns for files to exclude
            max_depth: Maximum directory depth to search
            follow_symlinks: Whether to follow symbolic links
            
        Returns:
            List of matching file paths
            
        Raises:
            FilteringError: If discovery fails
            AccessDeniedError: If directory access is denied
        """
        ...
    
    async def discover_files_stream(
        self,
        base_path: Path,
        include_patterns: list[str] = None,
        exclude_patterns: list[str] = None,
        max_depth: int = None,
        follow_symlinks: bool = False
    ) -> AsyncGenerator[Path, None]:
        """
        Stream file discovery for large directory trees.
        
        Args:
            base_path: Root directory to search
            include_patterns: Glob patterns for files to include
            exclude_patterns: Glob patterns for files to exclude
            max_depth: Maximum directory depth to search
            follow_symlinks: Whether to follow symbolic links
            
        Yields:
            File paths as they are discovered
            
        Raises:
            FilteringError: If discovery fails
            AccessDeniedError: If directory access is denied
        """
        ...
    
    # File filtering operations
    def should_include_file(
        self,
        file_path: Path,
        include_patterns: list[str] = None,
        exclude_patterns: list[str] = None
    ) -> bool:
        """
        Determine if a file should be included based on patterns.
        
        Args:
            file_path: Path to evaluate
            include_patterns: Patterns that must match
            exclude_patterns: Patterns that must not match
            
        Returns:
            True if file should be included
        """
        ...
    
    def should_include_directory(
        self,
        dir_path: Path,
        exclude_patterns: list[str] = None
    ) -> bool:
        """
        Determine if a directory should be traversed.
        
        Args:
            dir_path: Directory path to evaluate
            exclude_patterns: Patterns that exclude directories
            
        Returns:
            True if directory should be traversed
        """
        ...
    
    # File metadata operations
    async def get_file_metadata(self, file_path: Path) -> FileMetadata:
        """
        Get metadata for a specific file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File metadata including size, type, etc.
            
        Raises:
            FileNotFoundError: If file doesn't exist
            AccessDeniedError: If file access is denied
        """
        ...
    
    async def get_directory_stats(self, dir_path: Path) -> DirectoryStats:
        """
        Get statistics for a directory tree.
        
        Args:
            dir_path: Directory to analyze
            
        Returns:
            Directory statistics including file counts, sizes
            
        Raises:
            DirectoryNotFoundError: If directory doesn't exist
            AccessDeniedError: If directory access is denied
        """
        ...
    
    # Pattern management
    def add_include_pattern(self, pattern: str) -> None:
        """Add an include pattern to the service."""
        ...
    
    def add_exclude_pattern(self, pattern: str) -> None:
        """Add an exclude pattern to the service."""
        ...
    
    def remove_pattern(self, pattern: str, pattern_type: PatternType) -> None:
        """Remove a pattern from the service."""
        ...
    
    def get_active_patterns(self) -> PatternInfo:
        """Get currently active patterns."""
        ...
    
    # Statistics and monitoring
    async def get_filtering_stats(self) -> FilteringStats:
        """
        Get statistics about filtering performance.
        
        Returns:
            Current filtering statistics
        """
        ...
    
    async def reset_stats(self) -> None:
        """Reset filtering statistics."""
        ...
```

### ValidationService Protocol

```python
@runtime_checkable
class ValidationService(Protocol):
    """Protocol for content validation services."""
    
    # Content validation operations
    async def validate_content(
        self,
        content: ContentItem,
        rules: list[ValidationRule] = None
    ) -> ValidationResult:
        """
        Validate a content item against rules.
        
        Args:
            content: Content item to validate
            rules: Optional custom validation rules
            
        Returns:
            Validation result with details
            
        Raises:
            ValidationError: If validation process fails
        """
        ...
    
    async def validate_chunk(
        self,
        chunk: CodeChunk,
        rules: list[ValidationRule] = None
    ) -> ValidationResult:
        """
        Validate a code chunk against rules.
        
        Args:
            chunk: Code chunk to validate
            rules: Optional custom validation rules
            
        Returns:
            Validation result with details
            
        Raises:
            ValidationError: If validation process fails
        """
        ...
    
    async def validate_batch(
        self,
        items: list[ContentItem | CodeChunk],
        rules: list[ValidationRule] = None
    ) -> list[ValidationResult]:
        """
        Validate multiple items in batch.
        
        Args:
            items: Items to validate
            rules: Optional custom validation rules
            
        Returns:
            List of validation results
            
        Raises:
            ValidationError: If validation process fails
        """
        ...
    
    # Rule management
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
    
    # Validation configuration
    def set_validation_level(self, level: ValidationLevel) -> None:
        """Set the validation strictness level."""
        ...
    
    def get_validation_level(self) -> ValidationLevel:
        """Get the current validation level."""
        ...
    
    # Statistics and monitoring
    async def get_validation_stats(self) -> ValidationStats:
        """
        Get statistics about validation performance.
        
        Returns:
            Current validation statistics
        """
        ...
    
    async def reset_stats(self) -> None:
        """Reset validation statistics."""
        ...
```

### CacheService Protocol

```python
@runtime_checkable
class CacheService(Protocol):
    """Protocol for caching services."""
    
    # Basic cache operations
    async def get(self, key: str) -> Any | None:
        """
        Get a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
            
        Raises:
            CacheError: If cache operation fails
        """
        ...
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = None,
        tags: list[str] = None
    ) -> None:
        """
        Set a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            tags: Optional tags for cache invalidation
            
        Raises:
            CacheError: If cache operation fails
        """
        ...
    
    async def delete(self, key: str) -> bool:
        """
        Delete a value from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was deleted
            
        Raises:
            CacheError: If cache operation fails
        """
        ...
    
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in cache.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists
            
        Raises:
            CacheError: If cache operation fails
        """
        ...
    
    # Batch operations
    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """
        Get multiple values from cache.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary of key-value pairs
            
        Raises:
            CacheError: If cache operation fails
        """
        ...
    
    async def set_many(
        self,
        items: dict[str, Any],
        ttl: int = None,
        tags: list[str] = None
    ) -> None:
        """
        Set multiple values in cache.
        
        Args:
            items: Dictionary of key-value pairs
            ttl: Time to live in seconds
            tags: Optional tags for cache invalidation
            
        Raises:
            CacheError: If cache operation fails
        """
        ...
    
    async def delete_many(self, keys: list[str]) -> int:
        """
        Delete multiple values from cache.
        
        Args:
            keys: List of cache keys to delete
            
        Returns:
            Number of keys deleted
            
        Raises:
            CacheError: If cache operation fails
        """
        ...
    
    # Pattern operations
    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching a pattern.
        
        Args:
            pattern: Glob pattern for keys to invalidate
            
        Returns:
            Number of keys invalidated
            
        Raises:
            CacheError: If cache operation fails
        """
        ...
    
    async def invalidate_tags(self, tags: list[str]) -> int:
        """
        Invalidate all keys with given tags.
        
        Args:
            tags: Tags to invalidate
            
        Returns:
            Number of keys invalidated
            
        Raises:
            CacheError: If cache operation fails
        """
        ...
    
    # Cache management
    async def clear(self) -> None:
        """
        Clear all cached data.
        
        Raises:
            CacheError: If cache operation fails
        """
        ...
    
    async def get_cache_stats(self) -> CacheStats:
        """
        Get cache statistics.
        
        Returns:
            Current cache statistics
            
        Raises:
            CacheError: If cache operation fails
        """
        ...
    
    async def get_cache_info(self) -> CacheInfo:
        """
        Get cache configuration information.
        
        Returns:
            Cache configuration and status
        """
        ...
```

---

## Service Registry APIs

### ServiceRegistry Class

**Location:** `src/codeweaver/factories/service_registry.py`

```python
class ServiceRegistry:
    """Registry for managing service providers and instances."""
    
    def __init__(self, factory: 'CodeWeaverFactory'):
        """Initialize service registry with factory reference."""
        ...
    
    # Provider registration
    def register_provider(
        self,
        service_type: ServiceType,
        provider_name: str,
        provider_class: type[ServiceProvider],
        capabilities: ServiceCapabilities = None
    ) -> None:
        """
        Register a service provider.
        
        Args:
            service_type: Type of service
            provider_name: Unique provider name
            provider_class: Provider implementation class
            capabilities: Optional provider capabilities
            
        Raises:
            ProviderRegistrationError: If registration fails
            DuplicateProviderError: If provider already exists
        """
        ...
    
    def unregister_provider(
        self,
        service_type: ServiceType,
        provider_name: str
    ) -> None:
        """
        Unregister a service provider.
        
        Args:
            service_type: Type of service
            provider_name: Provider name to unregister
            
        Raises:
            ProviderNotFoundError: If provider doesn't exist
        """
        ...
    
    def get_providers(
        self,
        service_type: ServiceType
    ) -> dict[str, ServiceProviderInfo]:
        """
        Get all providers for a service type.
        
        Args:
            service_type: Type of service
            
        Returns:
            Dictionary of provider name to provider info
        """
        ...
    
    def get_provider_info(
        self,
        service_type: ServiceType,
        provider_name: str
    ) -> ServiceProviderInfo | None:
        """
        Get information about a specific provider.
        
        Args:
            service_type: Type of service
            provider_name: Provider name
            
        Returns:
            Provider information or None if not found
        """
        ...
    
    # Service instance management
    def create_service(
        self,
        service_type: ServiceType,
        config: ServiceConfig = None,
        provider_name: str = None
    ) -> Any:
        """
        Create a service instance.
        
        Args:
            service_type: Type of service to create
            config: Optional service configuration
            provider_name: Optional specific provider name
            
        Returns:
            Service instance implementing the service protocol
            
        Raises:
            ServiceCreationError: If service creation fails
            ProviderNotFoundError: If provider doesn't exist
            ConfigurationError: If configuration is invalid
        """
        ...
    
    def get_service(
        self,
        service_type: ServiceType,
        create_if_missing: bool = True
    ) -> Any:
        """
        Get an existing service instance or create one.
        
        Args:
            service_type: Type of service
            create_if_missing: Create instance if it doesn't exist
            
        Returns:
            Service instance implementing the service protocol
            
        Raises:
            ServiceNotFoundError: If service doesn't exist and create_if_missing is False
            ServiceCreationError: If service creation fails
        """
        ...
    
    def destroy_service(self, service_type: ServiceType) -> None:
        """
        Destroy a service instance.
        
        Args:
            service_type: Type of service to destroy
            
        Raises:
            ServiceNotFoundError: If service doesn't exist
        """
        ...
    
    # Configuration management
    def configure_service(
        self,
        service_type: ServiceType,
        config: ServiceConfig
    ) -> None:
        """
        Configure a service.
        
        Args:
            service_type: Type of service
            config: Service configuration
            
        Raises:
            ServiceNotFoundError: If service doesn't exist
            ConfigurationError: If configuration is invalid
        """
        ...
    
    def get_service_config(
        self,
        service_type: ServiceType
    ) -> ServiceConfig | None:
        """
        Get configuration for a service.
        
        Args:
            service_type: Type of service
            
        Returns:
            Service configuration or None if not configured
        """
        ...
    
    # Registry inspection
    def get_registered_services(self) -> dict[ServiceType, list[str]]:
        """
        Get all registered services and their providers.
        
        Returns:
            Dictionary mapping service types to provider names
        """
        ...
    
    def get_active_services(self) -> dict[ServiceType, ServiceInstanceInfo]:
        """
        Get all active service instances.
        
        Returns:
            Dictionary mapping service types to instance info
        """
        ...
    
    async def health_check(self) -> ServiceRegistryHealth:
        """
        Check health of all services.
        
        Returns:
            Health status of all services
        """
        ...
```

### ServicesManager Class

**Location:** `src/codeweaver/services/services_manager.py`

```python
class ServicesManager:
    """Central manager for service lifecycle and dependencies."""
    
    def __init__(
        self,
        registry: ServiceRegistry,
        config: ServicesConfig,
        health_monitor: ServiceHealthMonitor = None
    ):
        """Initialize services manager."""
        ...
    
    # Lifecycle management
    async def initialize_services(self) -> None:
        """
        Initialize all configured services.
        
        Raises:
            ServiceInitializationError: If initialization fails
        """
        ...
    
    async def start_services(self) -> None:
        """
        Start all initialized services.
        
        Raises:
            ServiceStartError: If service startup fails
        """
        ...
    
    async def stop_services(self) -> None:
        """
        Stop all running services.
        
        Raises:
            ServiceStopError: If service shutdown fails
        """
        ...
    
    async def restart_service(self, service_type: ServiceType) -> None:
        """
        Restart a specific service.
        
        Args:
            service_type: Type of service to restart
            
        Raises:
            ServiceRestartError: If restart fails
        """
        ...
    
    # Service access
    def get_service(self, service_type: ServiceType) -> Any:
        """
        Get a service instance.
        
        Args:
            service_type: Type of service
            
        Returns:
            Service instance implementing the service protocol
            
        Raises:
            ServiceNotFoundError: If service doesn't exist
            ServiceNotReadyError: If service is not ready
        """
        ...
    
    def get_all_services(self) -> dict[ServiceType, Any]:
        """
        Get all active services.
        
        Returns:
            Dictionary mapping service types to instances
        """
        ...
    
    # Dependency injection
    def inject_dependencies(self, target: Any) -> None:
        """
        Inject service dependencies into target object.
        
        Args:
            target: Object to inject dependencies into
            
        Raises:
            DependencyInjectionError: If injection fails
        """
        ...
    
    def resolve_dependencies(
        self,
        dependencies: list[ServiceType]
    ) -> dict[ServiceType, Any]:
        """
        Resolve multiple service dependencies.
        
        Args:
            dependencies: List of required service types
            
        Returns:
            Dictionary mapping service types to instances
            
        Raises:
            DependencyResolutionError: If resolution fails
        """
        ...
    
    # Health monitoring
    async def health_check(self) -> ServicesHealthReport:
        """
        Check health of all services.
        
        Returns:
            Comprehensive health report
        """
        ...
    
    async def health_check_service(
        self,
        service_type: ServiceType
    ) -> ServiceHealth:
        """
        Check health of a specific service.
        
        Args:
            service_type: Type of service to check
            
        Returns:
            Service health status
        """
        ...
    
    # Configuration management
    async def reconfigure_service(
        self,
        service_type: ServiceType,
        config: ServiceConfig
    ) -> None:
        """
        Reconfigure a service at runtime.
        
        Args:
            service_type: Type of service
            config: New configuration
            
        Raises:
            ReconfigurationError: If reconfiguration fails
        """
        ...
    
    def get_service_configuration(
        self,
        service_type: ServiceType
    ) -> ServiceConfig | None:
        """
        Get current configuration for a service.
        
        Args:
            service_type: Type of service
            
        Returns:
            Current service configuration
        """
        ...
```

---

## Data Structures

### Core Data Types

**Location:** `src/codeweaver/_types/service_data.py`

```python
from typing import Any, Literal
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

# Service types
class ServiceType(BaseEnum):
    CHUNKING = "chunking"
    FILTERING = "filtering"
    VALIDATION = "validation"
    CACHE = "cache"
    MONITORING = "monitoring"
    METRICS = "metrics"

# Service provider information
class ServiceProviderInfo(BaseModel):
    """Information about a service provider."""
    
    name: Annotated[str, Field(description="Provider name")]
    version: Annotated[str, Field(description="Provider version")]
    capabilities: Annotated[ServiceCapabilities, Field(description="Provider capabilities")]
    configuration_schema: Annotated[dict[str, Any], Field(description="Configuration schema")]
    status: Annotated[ProviderStatus, Field(description="Provider status")]
    created_at: Annotated[datetime, Field(description="Creation timestamp")]
    last_modified: Annotated[datetime, Field(description="Last modification timestamp")]

class ServiceCapabilities(BaseModel):
    """Capabilities of a service provider."""
    
    supports_streaming: Annotated[bool, Field(description="Supports streaming operations")] = False
    supports_batch: Annotated[bool, Field(description="Supports batch operations")] = True
    supports_async: Annotated[bool, Field(description="Supports async operations")] = True
    max_concurrency: Annotated[int, Field(ge=1, description="Maximum concurrent operations")] = 10
    memory_usage: Annotated[str, Field(description="Expected memory usage category")] = "medium"
    performance_profile: Annotated[str, Field(description="Performance characteristics")] = "standard"

# Health monitoring
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

class HealthStatus(BaseEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class ServicesHealthReport(BaseModel):
    """Comprehensive health report for all services."""
    
    overall_status: Annotated[HealthStatus, Field(description="Overall system health")]
    services: Annotated[dict[ServiceType, ServiceHealth], Field(description="Individual service health")]
    check_time: Annotated[datetime, Field(description="Health check timestamp")]
    metrics: Annotated[dict[str, Any], Field(description="Additional health metrics")]

# Validation types
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
    parameters: Annotated[dict[str, Any], Field(description="Rule parameters")]
    tags: Annotated[list[str], Field(description="Rule tags for categorization")]

class ValidationSeverity(BaseEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ValidationLevel(BaseEnum):
    STRICT = "strict"
    STANDARD = "standard"
    RELAXED = "relaxed"
    DISABLED = "disabled"

# Statistics types
class ChunkingStats(BaseModel):
    """Statistics for chunking operations."""
    
    total_files_processed: Annotated[int, Field(ge=0, description="Total files processed")] = 0
    total_chunks_created: Annotated[int, Field(ge=0, description="Total chunks created")] = 0
    average_chunk_size: Annotated[float, Field(ge=0, description="Average chunk size in characters")] = 0.0
    total_processing_time: Annotated[float, Field(ge=0, description="Total processing time in seconds")] = 0.0
    languages_processed: Annotated[dict[str, int], Field(description="Files processed by language")]
    error_count: Annotated[int, Field(ge=0, description="Number of errors")] = 0
    success_rate: Annotated[float, Field(ge=0, le=1, description="Success rate")] = 1.0

class FilteringStats(BaseModel):
    """Statistics for filtering operations."""
    
    total_files_scanned: Annotated[int, Field(ge=0, description="Total files scanned")] = 0
    total_files_included: Annotated[int, Field(ge=0, description="Total files included")] = 0
    total_files_excluded: Annotated[int, Field(ge=0, description="Total files excluded")] = 0
    total_directories_scanned: Annotated[int, Field(ge=0, description="Total directories scanned")] = 0
    total_scan_time: Annotated[float, Field(ge=0, description="Total scan time in seconds")] = 0.0
    patterns_matched: Annotated[dict[str, int], Field(description="Files matched by pattern")]
    error_count: Annotated[int, Field(ge=0, description="Number of errors")] = 0

class ValidationStats(BaseModel):
    """Statistics for validation operations."""
    
    total_validations: Annotated[int, Field(ge=0, description="Total validations performed")] = 0
    total_passed: Annotated[int, Field(ge=0, description="Total validations passed")] = 0
    total_failed: Annotated[int, Field(ge=0, description="Total validations failed")] = 0
    total_warnings: Annotated[int, Field(ge=0, description="Total warnings generated")] = 0
    average_validation_time: Annotated[float, Field(ge=0, description="Average validation time")] = 0.0
    rules_triggered: Annotated[dict[str, int], Field(description="Rules triggered count")]
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
```

---

## Configuration APIs

### Service Configuration Schema

**Location:** `src/codeweaver/_types/service_config.py`

```python
class ServiceConfig(BaseModel):
    """Base configuration for all services."""
    
    model_config = ConfigDict(extra="allow")
    
    enabled: Annotated[bool, Field(description="Whether service is enabled")] = True
    provider: Annotated[str, Field(description="Service provider name")]
    priority: Annotated[int, Field(ge=0, le=100, description="Service priority (0-100)")] = 50
    timeout: Annotated[float, Field(gt=0, description="Service timeout in seconds")] = 30.0
    max_retries: Annotated[int, Field(ge=0, description="Maximum retry attempts")] = 3
    retry_delay: Annotated[float, Field(ge=0, description="Delay between retries in seconds")] = 1.0
    health_check_interval: Annotated[float, Field(gt=0, description="Health check interval")] = 60.0
    tags: Annotated[list[str], Field(description="Service tags")] = Field(default_factory=list)
    metadata: Annotated[dict[str, Any], Field(description="Additional metadata")] = Field(default_factory=dict)

class ChunkingServiceConfig(ServiceConfig):
    """Configuration for chunking services."""
    
    provider: str = "fastmcp_chunking"
    max_chunk_size: Annotated[int, Field(gt=0, le=10000, description="Maximum chunk size in characters")] = 1500
    min_chunk_size: Annotated[int, Field(gt=0, le=1000, description="Minimum chunk size in characters")] = 50
    overlap_size: Annotated[int, Field(ge=0, description="Overlap between chunks in characters")] = 100
    ast_grep_enabled: Annotated[bool, Field(description="Enable AST-based chunking")] = True
    fallback_strategy: Annotated[ChunkingStrategy, Field(description="Fallback when AST fails")] = ChunkingStrategy.SIMPLE
    language_detection: Annotated[LanguageDetectionConfig, Field(description="Language detection config")] = Field(default_factory=LanguageDetectionConfig)
    performance_mode: Annotated[PerformanceMode, Field(description="Performance optimization mode")] = PerformanceMode.BALANCED

class FilteringServiceConfig(ServiceConfig):
    """Configuration for filtering services."""
    
    provider: str = "fastmcp_filtering"
    include_patterns: Annotated[list[str], Field(description="Default include patterns")] = Field(default_factory=list)
    exclude_patterns: Annotated[list[str], Field(description="Default exclude patterns")] = Field(default_factory=list)
    max_file_size: Annotated[int, Field(gt=0, description="Maximum file size in bytes")] = 1024 * 1024  # 1MB
    max_depth: Annotated[int | None, Field(ge=0, description="Maximum directory depth")] = None
    follow_symlinks: Annotated[bool, Field(description="Follow symbolic links")] = False
    ignore_hidden: Annotated[bool, Field(description="Ignore hidden files and directories")] = True
    use_gitignore: Annotated[bool, Field(description="Respect .gitignore files")] = True
    parallel_scanning: Annotated[bool, Field(description="Enable parallel directory scanning")] = True
    max_concurrent_scans: Annotated[int, Field(gt=0, description="Max concurrent directory scans")] = 10

class ValidationServiceConfig(ServiceConfig):
    """Configuration for validation services."""
    
    provider: str = "default_validation"
    validation_level: Annotated[ValidationLevel, Field(description="Validation strictness level")] = ValidationLevel.STANDARD
    max_errors_per_item: Annotated[int, Field(ge=0, description="Max errors per validation item")] = 10
    stop_on_first_error: Annotated[bool, Field(description="Stop validation on first error")] = False
    parallel_validation: Annotated[bool, Field(description="Enable parallel validation")] = True
    max_concurrent_validations: Annotated[int, Field(gt=0, description="Max concurrent validations")] = 5
    cache_results: Annotated[bool, Field(description="Cache validation results")] = True
    result_cache_ttl: Annotated[int, Field(gt=0, description="Result cache TTL in seconds")] = 3600

class CacheServiceConfig(ServiceConfig):
    """Configuration for cache services."""
    
    provider: str = "memory_cache"
    max_size: Annotated[int, Field(gt=0, description="Maximum cache size in bytes")] = 100 * 1024 * 1024  # 100MB
    max_items: Annotated[int, Field(gt=0, description="Maximum number of cached items")] = 10000
    default_ttl: Annotated[int, Field(gt=0, description="Default TTL in seconds")] = 3600
    eviction_policy: Annotated[EvictionPolicy, Field(description="Cache eviction policy")] = EvictionPolicy.LRU
    persistence_enabled: Annotated[bool, Field(description="Enable cache persistence")] = False
    persistence_path: Annotated[Path | None, Field(description="Path for cache persistence")] = None
    compression_enabled: Annotated[bool, Field(description="Enable cache compression")] = False
    metrics_enabled: Annotated[bool, Field(description="Enable cache metrics")] = True

class ServicesConfig(BaseModel):
    """Root configuration for all services."""
    
    chunking: Annotated[ChunkingServiceConfig, Field(description="Chunking service config")] = Field(default_factory=ChunkingServiceConfig)
    filtering: Annotated[FilteringServiceConfig, Field(description="Filtering service config")] = Field(default_factory=FilteringServiceConfig)
    validation: Annotated[ValidationServiceConfig, Field(description="Validation service config")] = Field(default_factory=ValidationServiceConfig)
    cache: Annotated[CacheServiceConfig, Field(description="Cache service config")] = Field(default_factory=CacheServiceConfig)
    global_timeout: Annotated[float, Field(gt=0, description="Global service timeout")] = 300.0
    health_check_enabled: Annotated[bool, Field(description="Enable health monitoring")] = True
    metrics_enabled: Annotated[bool, Field(description="Enable service metrics")] = True
    auto_recovery: Annotated[bool, Field(description="Enable automatic service recovery")] = True
```

---

## Error Handling APIs

### Service-Specific Exceptions

**Location:** `src/codeweaver/_types/service_exceptions.py`

```python
class ServiceError(CodeWeaverError):
    """Base exception for service-related errors."""
    pass

class ServiceNotFoundError(ServiceError):
    """Exception raised when a service is not found."""
    
    def __init__(self, service_type: ServiceType):
        super().__init__(f"Service not found: {service_type.value}")
        self.service_type = service_type

class ServiceCreationError(ServiceError):
    """Exception raised when service creation fails."""
    
    def __init__(self, service_type: ServiceType, reason: str):
        super().__init__(f"Failed to create service {service_type.value}: {reason}")
        self.service_type = service_type
        self.reason = reason

class ServiceConfigurationError(ServiceError):
    """Exception raised for service configuration errors."""
    
    def __init__(self, service_type: ServiceType, config_error: str):
        super().__init__(f"Configuration error for {service_type.value}: {config_error}")
        self.service_type = service_type
        self.config_error = config_error

class ChunkingError(ServiceError):
    """Exception raised for chunking-related errors."""
    
    def __init__(self, file_path: Path, reason: str):
        super().__init__(f"Chunking failed for {file_path}: {reason}")
        self.file_path = file_path
        self.reason = reason

class FilteringError(ServiceError):
    """Exception raised for filtering-related errors."""
    
    def __init__(self, path: Path, reason: str):
        super().__init__(f"Filtering failed for {path}: {reason}")
        self.path = path
        self.reason = reason

class ValidationError(ServiceError):
    """Exception raised for validation errors."""
    
    def __init__(self, item_id: str, rule_id: str, message: str):
        super().__init__(f"Validation failed for {item_id} (rule {rule_id}): {message}")
        self.item_id = item_id
        self.rule_id = rule_id

class CacheError(ServiceError):
    """Exception raised for cache-related errors."""
    
    def __init__(self, operation: str, key: str, reason: str):
        super().__init__(f"Cache {operation} failed for key '{key}': {reason}")
        self.operation = operation
        self.key = key
        self.reason = reason
```

---

## Integration Patterns

### Factory Integration API

**Location:** `src/codeweaver/factories/codeweaver_factory.py`

```python
class CodeWeaverFactory:
    """Extended factory with services integration."""
    
    def __init__(self, config: CodeWeaverConfig = None):
        """Initialize factory with services support."""
        ...
    
    # Service factory methods
    def create_service(
        self,
        service_type: ServiceType,
        config: ServiceConfig = None,
        provider_name: str = None
    ) -> Any:
        """
        Create a service instance through the factory.
        
        Args:
            service_type: Type of service to create
            config: Optional service configuration
            provider_name: Optional specific provider
            
        Returns:
            Service instance implementing the service protocol
            
        Raises:
            ServiceCreationError: If service creation fails
        """
        ...
    
    def get_services_manager(self) -> ServicesManager:
        """
        Get the services manager instance.
        
        Returns:
            Services manager for lifecycle management
        """
        ...
    
    def create_source_with_services(
        self,
        source_type: SourceType,
        config: SourceConfig
    ) -> AbstractDataSource:
        """
        Create a data source with service dependencies injected.
        
        Args:
            source_type: Type of source to create
            config: Source configuration
            
        Returns:
            Data source with injected services
            
        Raises:
            SourceCreationError: If source creation fails
        """
        ...
    
    # Service configuration
    async def configure_services(self, services_config: ServicesConfig) -> None:
        """
        Configure all services.
        
        Args:
            services_config: Services configuration
            
        Raises:
            ServiceConfigurationError: If configuration fails
        """
        ...
    
    async def initialize_services(self) -> None:
        """
        Initialize all configured services.
        
        Raises:
            ServiceInitializationError: If initialization fails
        """
        ...
    
    async def shutdown_services(self) -> None:
        """
        Shutdown all services gracefully.
        
        Raises:
            ServiceShutdownError: If shutdown fails
        """
        ...
```

### Plugin Integration API

**Example:** Clean plugin implementation with service injection

```python
class FileSystemSource(AbstractDataSource):
    """Clean filesystem source implementation."""
    
    def __init__(
        self,
        config: FileSystemSourceConfig,
        chunking_service: ChunkingService,
        filtering_service: FilteringService,
        validation_service: ValidationService | None = None,
        cache_service: CacheService | None = None
    ):
        """Initialize with injected services."""
        super().__init__(config)
        self._chunking_service = chunking_service
        self._filtering_service = filtering_service
        self._validation_service = validation_service
        self._cache_service = cache_service
    
    async def index_content(
        self,
        path: Path,
        context: dict[str, Any] = None
    ) -> list[ContentItem]:
        """Index content using injected services."""
        
        # Check cache first if available
        if self._cache_service:
            cache_key = f"index:{path}:{hash(str(context))}"
            cached_result = await self._cache_service.get(cache_key)
            if cached_result:
                return cached_result
        
        try:
            # Use filtering service for file discovery
            files = await self._filtering_service.discover_files(
                path,
                self.config.include_patterns,
                self.config.exclude_patterns
            )
            
            content_items = []
            for file_path in files:
                # Process each file using chunking service
                content = await self._read_file_content(file_path)
                chunks = await self._chunking_service.chunk_content(
                    content,
                    file_path,
                    context
                )
                
                # Validate chunks if validation service available
                if self._validation_service:
                    validated_chunks = []
                    for chunk in chunks:
                        validation_result = await self._validation_service.validate_chunk(chunk)
                        if validation_result.is_valid:
                            validated_chunks.append(chunk)
                        else:
                            # Log validation failures
                            logger.warning("Chunk validation failed: %s", validation_result.errors)
                    chunks = validated_chunks
                
                content_items.extend(chunks)
            
            # Cache result if cache service available
            if self._cache_service:
                await self._cache_service.set(
                    cache_key,
                    content_items,
                    ttl=3600  # 1 hour
                )
            
            return content_items
            
        except Exception as e:
            logger.exception("Failed to index content at %s", path)
            raise IndexingError(path, str(e)) from e
```

---

## Conclusion

This API specification provides a comprehensive contract for CodeWeaver's services layer, enabling:

- **Type-Safe Service Contracts** through protocol-based interfaces
- **Flexible Service Management** via registry and manager APIs
- **Comprehensive Configuration** with validation and defaults
- **Robust Error Handling** with specific exception types
- **Clean Integration Patterns** for factory and plugin systems

The API design ensures extensibility, testability, and maintainability while providing clear contracts for all service interactions.