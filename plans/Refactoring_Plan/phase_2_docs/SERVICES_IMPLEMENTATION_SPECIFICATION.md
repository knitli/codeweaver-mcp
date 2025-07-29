<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Services Implementation Specification

**CodeWeaver MCP Server - Services Layer Implementation Guide**

*Document Version: 1.0*
*Implementation Guide: Complete*
*Design Date: 2025-07-26*
*Target: Clean Implementation (Pre-Release)*

---

## Implementation Overview

This specification provides detailed implementation guidance for CodeWeaver's services registry and abstraction layer, including file organization, code patterns, implementation sequences, and testing strategies.

**Implementation Approach:**
- 🏗️ **Clean Slate Implementation**: No legacy code preservation
- 📁 **Modular Organization**: Clear separation of concerns
- 🔄 **Iterative Development**: Phase-based implementation
- 🧪 **Test-Driven Design**: Tests define service contracts
- 📊 **Observable Implementation**: Built-in monitoring and metrics

---

## File Organization Structure

### New Directory Structure

```
src/codeweaver/
├── _types/
│   ├── services.py              # Service protocol interfaces (NEW)
│   ├── service_config.py        # Service configuration types (NEW)
│   ├── service_data.py          # Service data structures (NEW)
│   ├── service_exceptions.py    # Service-specific exceptions (NEW)
│   └── config.py                # Extended with ServiceType enum
│
├── services/                    # NEW: Services implementation layer
│   ├── __init__.py
│   ├── services_manager.py      # Central services orchestrator
│   ├── health_monitor.py        # Service health monitoring
│   ├── dependency_resolver.py   # Service dependency resolution
│   │
│   └── providers/               # Service provider implementations
│       ├── __init__.py
│       ├── chunking/
│       │   ├── __init__.py
│       │   ├── fastmcp_provider.py
│       │   └── base_provider.py
│       ├── filtering/
│       │   ├── __init__.py
│       │   ├── fastmcp_provider.py
│       │   └── base_provider.py
│       ├── validation/
│       │   ├── __init__.py
│       │   ├── default_provider.py
│       │   └── base_provider.py
│       └── cache/
│           ├── __init__.py
│           ├── memory_provider.py
│           ├── redis_provider.py
│           └── base_provider.py
│
├── factories/
│   ├── service_registry.py      # NEW: Service registration and management
│   ├── codeweaver_factory.py    # UPDATED: Service integration
│   └── ...
│
├── sources/
│   ├── filesystem.py            # UPDATED: Clean service injection
│   ├── git.py                   # UPDATED: Clean service injection
│   └── ...
│
└── middleware/                  # UPDATED: Service integration
    ├── service_middleware.py    # NEW: Service-aware middleware
    └── ...
```

### Test Organization Structure

```
tests/
├── unit/
│   ├── services/
│   │   ├── test_services_manager.py
│   │   ├── test_service_registry.py
│   │   ├── test_health_monitor.py
│   │   └── providers/
│   │       ├── test_chunking_providers.py
│   │       ├── test_filtering_providers.py
│   │       ├── test_validation_providers.py
│   │       └── test_cache_providers.py
│   │
│   ├── factories/
│   │   ├── test_service_registry.py
│   │   └── test_factory_service_integration.py
│   │
│   └── sources/
│       ├── test_filesystem_service_injection.py
│       └── test_source_service_contracts.py
│
├── integration/
│   ├── test_services_integration.py
│   ├── test_service_lifecycle.py
│   ├── test_service_health_monitoring.py
│   └── test_end_to_end_services.py
│
└── fixtures/
    ├── service_configs/
    ├── mock_services/
    └── test_data/
```

---

## Phase 1: Core Infrastructure Implementation

### 1.1 Service Type System

**File:** `src/codeweaver/_types/config.py` (UPDATE)

```python
from codeweaver.types import BaseEnum

class ComponentType(BaseEnum):
    """Extended component types including services."""
    BACKEND = "backend"
    PROVIDER = "provider"
    SOURCE = "source"
    SERVICE = "service"      # NEW
    MIDDLEWARE = "middleware" # NEW
    FACTORY = "factory"
    PLUGIN = "plugin"

# NEW enum for service types
class ServiceType(BaseEnum):
    """Types of services in the system."""
    CHUNKING = "chunking"
    FILTERING = "filtering"
    VALIDATION = "validation"
    CACHE = "cache"
    MONITORING = "monitoring"
    METRICS = "metrics"

    @classmethod
    def get_core_services(cls) -> list['ServiceType']:
        """Get core services required for basic operation."""
        return [cls.CHUNKING, cls.FILTERING]

    @classmethod
    def get_optional_services(cls) -> list['ServiceType']:
        """Get optional services for enhanced functionality."""
        return [cls.VALIDATION, cls.CACHE, cls.MONITORING, cls.METRICS]
```

### 1.2 Service Protocol Interfaces

**File:** `src/codeweaver/_types/services.py` (NEW)

```python
from typing import Protocol, runtime_checkable, AsyncGenerator, Any
from pathlib import Path
from abc import abstractmethod

from codeweaver.types import CodeChunk, ContentItem
from codeweaver.types import (
    ChunkingStats, FilteringStats, ValidationResult, CacheStats,
    ValidationRule, FileMetadata, DirectoryStats
)
from codeweaver.types import Language, ChunkingStrategy

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

    async def health_check(self) -> bool:
        """Check if the service is healthy."""
        ...

@runtime_checkable
class ChunkingService(Protocol):
    """Protocol for content chunking services."""

    async def chunk_content(
        self,
        content: str,
        file_path: Path,
        metadata: dict[str, Any] = None,
        strategy: ChunkingStrategy = ChunkingStrategy.AUTO
    ) -> list[CodeChunk]:
        """Chunk content into code segments."""
        ...

    async def chunk_content_stream(
        self,
        content: str,
        file_path: Path,
        metadata: dict[str, Any] = None,
        strategy: ChunkingStrategy = ChunkingStrategy.AUTO
    ) -> AsyncGenerator[CodeChunk, None]:
        """Stream chunks for large files."""
        ...

    def detect_language(self, file_path: Path, content: str = None) -> Language | None:
        """Detect programming language."""
        ...

    def get_supported_languages(self) -> dict[Language, dict[str, Any]]:
        """Get supported languages and capabilities."""
        ...

    async def get_chunking_stats(self) -> ChunkingStats:
        """Get chunking performance statistics."""
        ...

@runtime_checkable
class FilteringService(Protocol):
    """Protocol for content filtering services."""

    async def discover_files(
        self,
        base_path: Path,
        include_patterns: list[str] = None,
        exclude_patterns: list[str] = None,
        max_depth: int = None,
        follow_symlinks: bool = False
    ) -> list[Path]:
        """Discover files matching criteria."""
        ...

    async def discover_files_stream(
        self,
        base_path: Path,
        include_patterns: list[str] = None,
        exclude_patterns: list[str] = None,
        max_depth: int = None,
        follow_symlinks: bool = False
    ) -> AsyncGenerator[Path, None]:
        """Stream file discovery."""
        ...

    def should_include_file(
        self,
        file_path: Path,
        include_patterns: list[str] = None,
        exclude_patterns: list[str] = None
    ) -> bool:
        """Check if file should be included."""
        ...

    async def get_file_metadata(self, file_path: Path) -> FileMetadata:
        """Get file metadata."""
        ...

    async def get_filtering_stats(self) -> FilteringStats:
        """Get filtering performance statistics."""
        ...

# Additional service protocols (ValidationService, CacheService) follow same pattern
```

### 1.3 Service Configuration System

**File:** `src/codeweaver/_types/service_config.py` (NEW)

```python
from typing import Annotated, Any
from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path

from codeweaver.types import ChunkingStrategy, ValidationLevel, PerformanceMode

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
    metadata: Annotated[dict[str, Any], Field(description="Additional metadata")] = Field(default_factory=dict)

class ChunkingServiceConfig(ServiceConfig):
    """Configuration for chunking services."""

    provider: str = "fastmcp_chunking"
    max_chunk_size: Annotated[int, Field(gt=0, le=10000, description="Max chunk size")] = 1500
    min_chunk_size: Annotated[int, Field(gt=0, le=1000, description="Min chunk size")] = 50
    overlap_size: Annotated[int, Field(ge=0, description="Chunk overlap size")] = 100
    ast_grep_enabled: Annotated[bool, Field(description="Enable AST chunking")] = True
    fallback_strategy: Annotated[ChunkingStrategy, Field(description="Fallback strategy")] = ChunkingStrategy.SIMPLE
    performance_mode: Annotated[PerformanceMode, Field(description="Performance mode")] = PerformanceMode.BALANCED

class FilteringServiceConfig(ServiceConfig):
    """Configuration for filtering services."""

    provider: str = "fastmcp_filtering"
    include_patterns: Annotated[list[str], Field(description="Include patterns")] = Field(default_factory=list)
    exclude_patterns: Annotated[list[str], Field(description="Exclude patterns")] = Field(default_factory=list)
    max_file_size: Annotated[int, Field(gt=0, description="Max file size in bytes")] = 1024 * 1024
    max_depth: Annotated[int | None, Field(ge=0, description="Max directory depth")] = None
    follow_symlinks: Annotated[bool, Field(description="Follow symlinks")] = False
    ignore_hidden: Annotated[bool, Field(description="Ignore hidden files")] = True
    use_gitignore: Annotated[bool, Field(description="Respect .gitignore")] = True
    parallel_scanning: Annotated[bool, Field(description="Enable parallel scanning")] = True
    max_concurrent_scans: Annotated[int, Field(gt=0, description="Max concurrent scans")] = 10

class ServicesConfig(BaseModel):
    """Root configuration for all services."""

    chunking: Annotated[ChunkingServiceConfig, Field(description="Chunking config")] = Field(default_factory=ChunkingServiceConfig)
    filtering: Annotated[FilteringServiceConfig, Field(description="Filtering config")] = Field(default_factory=FilteringServiceConfig)
    # Add other service configs as needed

    global_timeout: Annotated[float, Field(gt=0, description="Global timeout")] = 300.0
    health_check_enabled: Annotated[bool, Field(description="Enable health checks")] = True
    metrics_enabled: Annotated[bool, Field(description="Enable metrics")] = True
    auto_recovery: Annotated[bool, Field(description="Enable auto recovery")] = True
```

### 1.4 Service Registry Implementation

**File:** `src/codeweaver/factories/service_registry.py` (NEW)

```python
import logging
from typing import Any, Dict, Type, TYPE_CHECKING
from collections import defaultdict

from codeweaver.types import ServiceType
from codeweaver.types import ServiceProvider
from codeweaver.types import ServiceConfig
from codeweaver.types import ServiceProviderInfo, ServiceCapabilities
from codeweaver.types import (
    ServiceNotFoundError, ServiceCreationError, ProviderRegistrationError,
    DuplicateProviderError, ProviderNotFoundError
)

if TYPE_CHECKING:
    from codeweaver.factories.codeweaver_factory import CodeWeaverFactory

logger = logging.getLogger(__name__)

class ServiceRegistry:
    """Registry for managing service providers and instances."""

    def __init__(self, factory: 'CodeWeaverFactory'):
        """Initialize service registry."""
        self._factory = factory
        self._providers: dict[ServiceType, dict[str, Type[ServiceProvider]]] = defaultdict(dict)
        self._provider_info: dict[ServiceType, dict[str, ServiceProviderInfo]] = defaultdict(dict)
        self._instances: dict[ServiceType, Any] = {}
        self._configs: dict[ServiceType, ServiceConfig] = {}

    def register_provider(
        self,
        service_type: ServiceType,
        provider_name: str,
        provider_class: Type[ServiceProvider],
        capabilities: ServiceCapabilities = None
    ) -> None:
        """Register a service provider."""
        logger.debug("Registering provider %s for service %s", provider_name, service_type.value)

        if provider_name in self._providers[service_type]:
            raise DuplicateProviderError(service_type, provider_name)

        # Validate provider implements required protocol
        if not self._validate_provider_protocol(service_type, provider_class):
            raise ProviderRegistrationError(
                service_type,
                provider_name,
                "Provider does not implement required protocol"
            )

        self._providers[service_type][provider_name] = provider_class

        # Store provider information
        info = ServiceProviderInfo(
            name=provider_name,
            version=getattr(provider_class, '__version__', '1.0.0'),
            capabilities=capabilities or ServiceCapabilities(),
            configuration_schema=self._extract_config_schema(provider_class),
            status="registered",
            created_at=UTC,
            last_modified=UTC
        )
        self._provider_info[service_type][provider_name] = info

        logger.info("Successfully registered provider %s for service %s", provider_name, service_type.value)

    def unregister_provider(self, service_type: ServiceType, provider_name: str) -> None:
        """Unregister a service provider."""
        if provider_name not in self._providers[service_type]:
            raise ProviderNotFoundError(service_type, provider_name)

        # Remove instance if it exists
        if service_type in self._instances:
            instance_info = self._get_instance_info(service_type)
            if instance_info and instance_info.provider_name == provider_name:
                del self._instances[service_type]

        del self._providers[service_type][provider_name]
        del self._provider_info[service_type][provider_name]

        logger.info("Unregistered provider %s for service %s", provider_name, service_type.value)

    def create_service(
        self,
        service_type: ServiceType,
        config: ServiceConfig = None,
        provider_name: str = None
    ) -> Any:
        """Create a service instance."""
        logger.debug("Creating service %s with provider %s", service_type.value, provider_name)

        # Determine provider to use
        if provider_name is None:
            provider_name = self._get_default_provider(service_type, config)

        if provider_name not in self._providers[service_type]:
            raise ProviderNotFoundError(service_type, provider_name)

        provider_class = self._providers[service_type][provider_name]

        try:
            # Use config or default
            effective_config = config or self._configs.get(service_type)
            if effective_config is None:
                effective_config = self._create_default_config(service_type)

            # Create provider instance
            if effective_config:
                instance = provider_class(effective_config)
            else:
                instance = provider_class()

            logger.info("Successfully created service %s using provider %s", service_type.value, provider_name)
            return instance

        except Exception as e:
            logger.exception("Failed to create service %s with provider %s", service_type.value, provider_name)
            raise ServiceCreationError(service_type, str(e)) from e

    def get_service(self, service_type: ServiceType, create_if_missing: bool = True) -> Any:
        """Get existing service instance or create one."""
        if service_type in self._instances:
            return self._instances[service_type]

        if not create_if_missing:
            raise ServiceNotFoundError(service_type)

        # Create and cache instance
        instance = self.create_service(service_type)
        self._instances[service_type] = instance
        return instance

    def configure_service(self, service_type: ServiceType, config: ServiceConfig) -> None:
        """Configure a service."""
        self._configs[service_type] = config

        # If instance exists, reconfigure it
        if service_type in self._instances:
            instance = self._instances[service_type]
            if hasattr(instance, 'reconfigure'):
                instance.reconfigure(config)

    def _validate_provider_protocol(self, service_type: ServiceType, provider_class: Type) -> bool:
        """Validate that provider implements required protocol."""
        # Import protocol classes
        from codeweaver.types import ChunkingService, FilteringService

        protocol_map = {
            ServiceType.CHUNKING: ChunkingService,
            ServiceType.FILTERING: FilteringService,
            # Add other mappings
        }

        required_protocol = protocol_map.get(service_type)
        if required_protocol is None:
            return True  # No specific protocol required

        # Check if class implements protocol methods
        try:
            # This is a runtime check - in production, you might want more sophisticated validation
            return hasattr(provider_class, '__annotations__') or callable(provider_class)
        except Exception:
            return False

    def _get_default_provider(self, service_type: ServiceType, config: ServiceConfig = None) -> str:
        """Get default provider for service type."""
        providers = self._providers[service_type]
        if not providers:
            raise ProviderNotFoundError(service_type, "any")

        # If config specifies provider, use it
        if config and hasattr(config, 'provider'):
            return config.provider

        # Return first available provider (in production, this could be more sophisticated)
        return next(iter(providers.keys()))

    def _create_default_config(self, service_type: ServiceType) -> ServiceConfig | None:
        """Create default configuration for service type."""
        from codeweaver.types import ChunkingServiceConfig, FilteringServiceConfig

        config_map = {
            ServiceType.CHUNKING: ChunkingServiceConfig,
            ServiceType.FILTERING: FilteringServiceConfig,
            # Add other mappings
        }

        config_class = config_map.get(service_type)
        return config_class() if config_class else None
```

---

## Phase 2: Service Provider Implementation

### 2.1 Base Service Provider

**File:** `src/codeweaver/services/providers/base_provider.py` (NEW)

```python
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from codeweaver.types import ServiceConfig
from codeweaver.types import ServiceHealth, HealthStatus

logger = logging.getLogger(__name__)

class BaseServiceProvider(ABC):
    """Base class for all service providers."""

    def __init__(self, config: ServiceConfig):
        """Initialize base service provider."""
        self.config = config
        self._initialized = False
        self._running = False
        self._stats = {}
        self._last_health_check = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Provider version."""
        ...

    async def initialize(self) -> None:
        """Initialize the service provider."""
        if self._initialized:
            return

        logger.debug("Initializing service provider %s", self.name)
        await self._initialize_impl()
        self._initialized = True
        logger.info("Service provider %s initialized successfully", self.name)

    async def shutdown(self) -> None:
        """Shutdown the service provider gracefully."""
        if not self._initialized:
            return

        logger.debug("Shutting down service provider %s", self.name)
        await self._shutdown_impl()
        self._initialized = False
        self._running = False
        logger.info("Service provider %s shut down successfully", self.name)

    async def start(self) -> None:
        """Start the service provider."""
        if not self._initialized:
            await self.initialize()

        if self._running:
            return

        logger.debug("Starting service provider %s", self.name)
        await self._start_impl()
        self._running = True
        logger.info("Service provider %s started successfully", self.name)

    async def stop(self) -> None:
        """Stop the service provider."""
        if not self._running:
            return

        logger.debug("Stopping service provider %s", self.name)
        await self._stop_impl()
        self._running = False
        logger.info("Service provider %s stopped successfully", self.name)

    async def health_check(self) -> ServiceHealth:
        """Check service health."""
        try:
            health_status = await self._health_check_impl()

            health = ServiceHealth(
                service_type=self._get_service_type(),
                status=health_status,
                last_check=UTC,
                response_time=0.0,  # Implement timing
                error_count=self._get_error_count(),
                success_rate=self._get_success_rate(),
                uptime=self._get_uptime(),
                memory_usage=self._get_memory_usage()
            )

            self._last_health_check = health
            return health

        except Exception as e:
            logger.exception("Health check failed for provider %s", self.name)
            return ServiceHealth(
                service_type=self._get_service_type(),
                status=HealthStatus.UNHEALTHY,
                last_check=UTC,
                response_time=0.0,
                error_count=self._get_error_count() + 1,
                last_error=str(e)
            )

    def reconfigure(self, config: ServiceConfig) -> None:
        """Reconfigure the service provider."""
        logger.debug("Reconfiguring service provider %s", self.name)
        old_config = self.config
        self.config = config

        try:
            self._reconfigure_impl(old_config, config)
            logger.info("Service provider %s reconfigured successfully", self.name)
        except Exception as e:
            logger.exception("Failed to reconfigure service provider %s", self.name)
            self.config = old_config  # Rollback
            raise

    # Abstract methods for implementation
    @abstractmethod
    async def _initialize_impl(self) -> None:
        """Implementation-specific initialization."""
        ...

    @abstractmethod
    async def _shutdown_impl(self) -> None:
        """Implementation-specific shutdown."""
        ...

    async def _start_impl(self) -> None:
        """Implementation-specific start (optional)."""
        pass

    async def _stop_impl(self) -> None:
        """Implementation-specific stop (optional)."""
        pass

    async def _health_check_impl(self) -> HealthStatus:
        """Implementation-specific health check."""
        return HealthStatus.HEALTHY if self._running else HealthStatus.UNHEALTHY

    def _reconfigure_impl(self, old_config: ServiceConfig, new_config: ServiceConfig) -> None:
        """Implementation-specific reconfiguration (optional)."""
        pass

    @abstractmethod
    def _get_service_type(self) -> 'ServiceType':
        """Get the service type this provider implements."""
        ...

    # Helper methods for metrics
    def _get_error_count(self) -> int:
        """Get current error count."""
        return self._stats.get('error_count', 0)

    def _get_success_rate(self) -> float:
        """Get current success rate."""
        return self._stats.get('success_rate', 1.0)

    def _get_uptime(self) -> float:
        """Get current uptime in seconds."""
        return self._stats.get('uptime', 0.0)

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        return self._stats.get('memory_usage', 0)

    def _increment_error_count(self) -> None:
        """Increment error count."""
        self._stats['error_count'] = self._stats.get('error_count', 0) + 1

    def _update_success_rate(self, success: bool) -> None:
        """Update success rate."""
        total = self._stats.get('total_operations', 0) + 1
        successes = self._stats.get('successful_operations', 0) + (1 if success else 0)

        self._stats['total_operations'] = total
        self._stats['successful_operations'] = successes
        self._stats['success_rate'] = successes / total if total > 0 else 1.0
```

### 2.2 FastMCP Chunking Provider

**File:** `src/codeweaver/services/providers/chunking/fastmcp_provider.py` (NEW)

```python
import logging
from typing import Any, AsyncGenerator
from pathlib import Path

from codeweaver.services.providers.base_provider import BaseServiceProvider
from codeweaver.types import ServiceType
from codeweaver.types import ChunkingService
from codeweaver.types import ChunkingServiceConfig
from codeweaver.types import CodeChunk
from codeweaver.types import ChunkingStats
from codeweaver.types import Language, ChunkingStrategy
from codeweaver.types import ChunkingError, UnsupportedLanguageError

# Import FastMCP middleware
from codeweaver.middleware.chunking import ChunkingMiddleware

logger = logging.getLogger(__name__)

class ChunkingService(BaseServiceProvider, ChunkingService):
    """FastMCP-based chunking service provider."""

    def __init__(self, config: ChunkingServiceConfig):
        """Initialize FastMCP chunking provider."""
        super().__init__(config)
        self._middleware = None
        self._stats = ChunkingStats()

    @property
    def name(self) -> str:
        """Provider name."""
        return "fastmcp_chunking"

    @property
    def version(self) -> str:
        """Provider version."""
        return "1.0.0"

    async def _initialize_impl(self) -> None:
        """Initialize FastMCP middleware."""
        try:
            # Create middleware configuration from service config
            middleware_config = {
                "max_chunk_size": self.config.max_chunk_size,
                "min_chunk_size": self.config.min_chunk_size,
                "overlap_size": self.config.overlap_size,
                "ast_grep_enabled": self.config.ast_grep_enabled,
                "fallback_strategy": self.config.fallback_strategy.value,
                "performance_mode": self.config.performance_mode.value
            }

            self._middleware = ChunkingMiddleware(middleware_config)
            await self._middleware.initialize()

            logger.info("FastMCP chunking middleware initialized")

        except Exception as e:
            logger.exception("Failed to initialize FastMCP chunking middleware")
            raise ChunkingError(Path(""), f"Initialization failed: {e}") from e

    async def _shutdown_impl(self) -> None:
        """Shutdown FastMCP middleware."""
        if self._middleware:
            try:
                await self._middleware.shutdown()
                self._middleware = None
                logger.info("FastMCP chunking middleware shut down")
            except Exception as e:
                logger.exception("Error shutting down FastMCP chunking middleware")

    def _get_service_type(self) -> ServiceType:
        """Get service type."""
        return ServiceType.CHUNKING

    # ChunkingService protocol implementation
    async def chunk_content(
        self,
        content: str,
        file_path: Path,
        metadata: dict[str, Any] = None,
        strategy: ChunkingStrategy = ChunkingStrategy.AUTO
    ) -> list[CodeChunk]:
        """Chunk content using FastMCP middleware."""
        if not self._middleware:
            raise ChunkingError(file_path, "Middleware not initialized")

        try:
            logger.debug("Chunking content for %s with strategy %s", file_path, strategy.value)

            # Prepare context for middleware
            context = {
                "file_path": str(file_path),
                "strategy": strategy.value,
                "metadata": metadata or {}
            }

            # Use middleware to chunk content
            chunks = await self._middleware.chunk_file_content(content, context)

            # Convert middleware chunks to CodeChunk objects
            code_chunks = []
            for i, chunk in enumerate(chunks):
                code_chunk = CodeChunk(
                    content=chunk.get("content", ""),
                    file_path=file_path,
                    start_line=chunk.get("start_line", 0),
                    end_line=chunk.get("end_line", 0),
                    chunk_index=i,
                    language=chunk.get("language"),
                    metadata=chunk.get("metadata", {})
                )
                code_chunks.append(code_chunk)

            # Update statistics
            self._stats.total_files_processed += 1
            self._stats.total_chunks_created += len(code_chunks)
            self._update_success_rate(True)

            logger.debug("Successfully chunked %s into %d chunks", file_path, len(code_chunks))
            return code_chunks

        except Exception as e:
            logger.exception("Failed to chunk content for %s", file_path)
            self._increment_error_count()
            self._update_success_rate(False)

            if "unsupported language" in str(e).lower():
                raise UnsupportedLanguageError(file_path, str(e)) from e
            else:
                raise ChunkingError(file_path, str(e)) from e

    async def chunk_content_stream(
        self,
        content: str,
        file_path: Path,
        metadata: dict[str, Any] = None,
        strategy: ChunkingStrategy = ChunkingStrategy.AUTO
    ) -> AsyncGenerator[CodeChunk, None]:
        """Stream chunks for large content."""
        # For now, implement as batch then stream
        # In future, could implement true streaming in middleware
        chunks = await self.chunk_content(content, file_path, metadata, strategy)
        for chunk in chunks:
            yield chunk

    def detect_language(self, file_path: Path, content: str = None) -> Language | None:
        """Detect programming language."""
        if not self._middleware:
            return None

        try:
            language_str = self._middleware.detect_language(str(file_path), content)
            return Language(language_str) if language_str else None
        except Exception as e:
            logger.warning("Failed to detect language for %s: %s", file_path, e)
            return None

    def get_supported_languages(self) -> dict[Language, dict[str, Any]]:
        """Get supported languages."""
        if not self._middleware:
            return {}

        try:
            middleware_languages = self._middleware.get_supported_languages()

            # Convert to Language enum dict
            supported = {}
            for lang_str, capabilities in middleware_languages.items():
                try:
                    language = Language(lang_str)
                    supported[language] = capabilities
                except ValueError:
                    # Skip unknown languages
                    continue

            return supported

        except Exception as e:
            logger.warning("Failed to get supported languages: %s", e)
            return {}

    async def get_chunking_stats(self) -> ChunkingStats:
        """Get chunking statistics."""
        return self._stats

    async def reset_stats(self) -> None:
        """Reset chunking statistics."""
        self._stats = ChunkingStats()
```

### 2.3 FastMCP Filtering Provider

**File:** `src/codeweaver/services/providers/filtering/fastmcp_provider.py` (NEW)

```python
import logging
from typing import AsyncGenerator
from pathlib import Path

from codeweaver.services.providers.base_provider import BaseServiceProvider
from codeweaver.types import ServiceType
from codeweaver.types import FilteringService
from codeweaver.types import FilteringServiceConfig
from codeweaver.types import FilteringStats, FileMetadata, DirectoryStats
from codeweaver.types import FilteringError, AccessDeniedError

# Import FastMCP middleware
from codeweaver.middleware.filtering import FilteringMiddleware

logger = logging.getLogger(__name__)

class FilteringService(BaseServiceProvider, FilteringService):
    """FastMCP-based filtering service provider."""

    def __init__(self, config: FilteringServiceConfig):
        """Initialize FastMCP filtering provider."""
        super().__init__(config)
        self._middleware = None
        self._stats = FilteringStats()

    @property
    def name(self) -> str:
        """Provider name."""
        return "fastmcp_filtering"

    @property
    def version(self) -> str:
        """Provider version."""
        return "1.0.0"

    async def _initialize_impl(self) -> None:
        """Initialize FastMCP middleware."""
        try:
            middleware_config = {
                "include_patterns": self.config.include_patterns,
                "exclude_patterns": self.config.exclude_patterns,
                "max_file_size": self.config.max_file_size,
                "max_depth": self.config.max_depth,
                "follow_symlinks": self.config.follow_symlinks,
                "ignore_hidden": self.config.ignore_hidden,
                "use_gitignore": self.config.use_gitignore,
                "parallel_scanning": self.config.parallel_scanning,
                "max_concurrent_scans": self.config.max_concurrent_scans
            }

            self._middleware = FilteringMiddleware(middleware_config)
            await self._middleware.initialize()

            logger.info("FastMCP filtering middleware initialized")

        except Exception as e:
            logger.exception("Failed to initialize FastMCP filtering middleware")
            raise FilteringError(Path(""), f"Initialization failed: {e}") from e

    async def _shutdown_impl(self) -> None:
        """Shutdown FastMCP middleware."""
        if self._middleware:
            try:
                await self._middleware.shutdown()
                self._middleware = None
                logger.info("FastMCP filtering middleware shut down")
            except Exception as e:
                logger.exception("Error shutting down FastMCP filtering middleware")

    def _get_service_type(self) -> ServiceType:
        """Get service type."""
        return ServiceType.FILTERING

    # FilteringService protocol implementation
    async def discover_files(
        self,
        base_path: Path,
        include_patterns: list[str] = None,
        exclude_patterns: list[str] = None,
        max_depth: int = None,
        follow_symlinks: bool = False
    ) -> list[Path]:
        """Discover files using FastMCP middleware."""
        if not self._middleware:
            raise FilteringError(base_path, "Middleware not initialized")

        try:
            logger.debug("Discovering files in %s", base_path)

            # Prepare discovery context
            context = {
                "include_patterns": include_patterns or self.config.include_patterns,
                "exclude_patterns": exclude_patterns or self.config.exclude_patterns,
                "max_depth": max_depth or self.config.max_depth,
                "follow_symlinks": follow_symlinks or self.config.follow_symlinks
            }

            # Use middleware to discover files
            file_paths = await self._middleware.discover_files(str(base_path), context)

            # Convert to Path objects
            discovered = [Path(file_path) for file_path in file_paths]

            # Update statistics
            self._stats.total_files_scanned += len(discovered)
            self._stats.total_files_included += len(discovered)
            self._update_success_rate(True)

            logger.debug("Discovered %d files in %s", len(discovered), base_path)
            return discovered

        except PermissionError as e:
            logger.error("Access denied to %s: %s", base_path, e)
            self._increment_error_count()
            self._update_success_rate(False)
            raise AccessDeniedError(base_path, str(e)) from e

        except Exception as e:
            logger.exception("Failed to discover files in %s", base_path)
            self._increment_error_count()
            self._update_success_rate(False)
            raise FilteringError(base_path, str(e)) from e

    async def discover_files_stream(
        self,
        base_path: Path,
        include_patterns: list[str] = None,
        exclude_patterns: list[str] = None,
        max_depth: int = None,
        follow_symlinks: bool = False
    ) -> AsyncGenerator[Path, None]:
        """Stream file discovery."""
        # For now, implement as batch then stream
        # In future, could implement true streaming in middleware
        files = await self.discover_files(
            base_path, include_patterns, exclude_patterns, max_depth, follow_symlinks
        )
        for file_path in files:
            yield file_path

    def should_include_file(
        self,
        file_path: Path,
        include_patterns: list[str] = None,
        exclude_patterns: list[str] = None
    ) -> bool:
        """Check if file should be included."""
        if not self._middleware:
            return True  # Default to include if middleware not available

        try:
            context = {
                "include_patterns": include_patterns or self.config.include_patterns,
                "exclude_patterns": exclude_patterns or self.config.exclude_patterns
            }

            return self._middleware.should_include_file(str(file_path), context)

        except Exception as e:
            logger.warning("Failed to check file inclusion for %s: %s", file_path, e)
            return True  # Default to include on error

    async def get_file_metadata(self, file_path: Path) -> FileMetadata:
        """Get file metadata."""
        if not self._middleware:
            raise FilteringError(file_path, "Middleware not initialized")

        try:
            metadata_dict = await self._middleware.get_file_metadata(str(file_path))

            return FileMetadata(
                path=file_path,
                size=metadata_dict.get("size", 0),
                modified_time=metadata_dict.get("modified_time"),
                created_time=metadata_dict.get("created_time"),
                file_type=metadata_dict.get("file_type", "unknown"),
                permissions=metadata_dict.get("permissions", ""),
                is_binary=metadata_dict.get("is_binary", False)
            )

        except Exception as e:
            logger.exception("Failed to get metadata for %s", file_path)
            raise FilteringError(file_path, f"Metadata retrieval failed: {e}") from e

    async def get_filtering_stats(self) -> FilteringStats:
        """Get filtering statistics."""
        return self._stats

    async def reset_stats(self) -> None:
        """Reset filtering statistics."""
        self._stats = FilteringStats()
```

---

## Phase 3: Services Manager Implementation

### 3.1 Services Manager

**File:** `src/codeweaver/services/services_manager.py` (NEW)

```python
import logging
import asyncio
from typing import Any, Dict, List
from datetime import datetime

from codeweaver.types import ServiceType
from codeweaver.types import ServiceProvider
from codeweaver.types import ServicesConfig, ServiceConfig
from codeweaver.types import ServicesHealthReport, ServiceHealth, HealthStatus
from codeweaver.types import (
    ServiceInitializationError, ServiceStartError, ServiceStopError, ServiceNotFoundError,
    ServiceNotReadyError, DependencyInjectionError, DependencyResolutionError
)
from codeweaver.factories.service_registry import ServiceRegistry
from codeweaver.services.health_monitor import ServiceHealthMonitor
from codeweaver.services.dependency_resolver import ServiceDependencyResolver

logger = logging.getLogger(__name__)

class ServicesManager:
    """Central manager for service lifecycle and dependencies."""

    def __init__(
        self,
        registry: ServiceRegistry,
        config: ServicesConfig,
        health_monitor: ServiceHealthMonitor = None
    ):
        """Initialize services manager."""
        self._registry = registry
        self._config = config
        self._health_monitor = health_monitor or ServiceHealthMonitor(self)
        self._dependency_resolver = ServiceDependencyResolver()

        self._services: Dict[ServiceType, Any] = {}
        self._service_states: Dict[ServiceType, str] = {}
        self._initialization_order: List[ServiceType] = []
        self._startup_time = None

    async def initialize_services(self) -> None:
        """Initialize all configured services."""
        logger.info("Initializing services...")

        try:
            # Determine initialization order based on dependencies
            self._initialization_order = self._dependency_resolver.resolve_initialization_order(
                self._get_enabled_services()
            )

            logger.debug("Service initialization order: %s", [s.value for s in self._initialization_order])

            # Initialize services in order
            for service_type in self._initialization_order:
                await self._initialize_service(service_type)

            logger.info("All services initialized successfully")

        except Exception as e:
            logger.exception("Failed to initialize services")
            # Cleanup any partially initialized services
            await self._cleanup_partial_initialization()
            raise ServiceInitializationError(f"Service initialization failed: {e}") from e

    async def start_services(self) -> None:
        """Start all initialized services."""
        logger.info("Starting services...")

        try:
            self._startup_time = UTC

            # Start services in initialization order
            for service_type in self._initialization_order:
                await self._start_service(service_type)

            # Start health monitoring
            if self._config.health_check_enabled:
                await self._health_monitor.start()

            logger.info("All services started successfully")

        except Exception as e:
            logger.exception("Failed to start services")
            await self._cleanup_failed_startup()
            raise ServiceStartError(f"Service startup failed: {e}") from e

    async def stop_services(self) -> None:
        """Stop all running services."""
        logger.info("Stopping services...")

        try:
            # Stop health monitoring first
            if self._health_monitor:
                await self._health_monitor.stop()

            # Stop services in reverse order
            for service_type in reversed(self._initialization_order):
                await self._stop_service(service_type)

            logger.info("All services stopped successfully")

        except Exception as e:
            logger.exception("Failed to stop services gracefully")
            raise ServiceStopError(f"Service shutdown failed: {e}") from e

    async def restart_service(self, service_type: ServiceType) -> None:
        """Restart a specific service."""
        logger.info("Restarting service %s", service_type.value)

        try:
            # Stop service
            await self._stop_service(service_type)

            # Remove from services dict
            if service_type in self._services:
                del self._services[service_type]

            # Reinitialize and start
            await self._initialize_service(service_type)
            await self._start_service(service_type)

            logger.info("Service %s restarted successfully", service_type.value)

        except Exception as e:
            logger.exception("Failed to restart service %s", service_type.value)
            raise ServiceRestartError(service_type, str(e)) from e

    def get_service(self, service_type: ServiceType) -> Any:
        """Get a service instance."""
        if service_type not in self._services:
            raise ServiceNotFoundError(service_type)

        service = self._services[service_type]
        state = self._service_states.get(service_type, "unknown")

        if state not in ["initialized", "running"]:
            raise ServiceNotReadyError(service_type, state)

        return service

    def get_all_services(self) -> Dict[ServiceType, Any]:
        """Get all active services."""
        return self._services.copy()

    def inject_dependencies(self, target: Any) -> None:
        """Inject service dependencies into target object."""
        logger.debug("Injecting dependencies into %s", target.__class__.__name__)

        try:
            # Look for service dependency annotations
            dependencies = self._extract_service_dependencies(target)

            # Resolve and inject dependencies
            for dep_name, service_type in dependencies.items():
                service = self.get_service(service_type)
                setattr(target, dep_name, service)

            logger.debug("Successfully injected %d dependencies", len(dependencies))

        except Exception as e:
            logger.exception("Failed to inject dependencies into %s", target.__class__.__name__)
            raise DependencyInjectionError(str(e)) from e

    def resolve_dependencies(
        self,
        dependencies: List[ServiceType]
    ) -> Dict[ServiceType, Any]:
        """Resolve multiple service dependencies."""
        try:
            resolved = {}
            for service_type in dependencies:
                resolved[service_type] = self.get_service(service_type)
            return resolved

        except Exception as e:
            logger.exception("Failed to resolve dependencies: %s", dependencies)
            raise DependencyResolutionError(str(e)) from e

    async def health_check(self) -> ServicesHealthReport:
        """Check health of all services."""
        service_healths = {}
        overall_status = HealthStatus.HEALTHY

        for service_type, service in self._services.items():
            try:
                if hasattr(service, 'health_check'):
                    health = await service.health_check()
                else:
                    # Basic health check based on service state
                    state = self._service_states.get(service_type, "unknown")
                    status = HealthStatus.HEALTHY if state == "running" else HealthStatus.DEGRADED
                    health = ServiceHealth(
                        service_type=service_type,
                        status=status,
                        last_check=UTC,
                        response_time=0.0
                    )

                service_healths[service_type] = health

                # Update overall status
                if health.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif health.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED

            except Exception as e:
                logger.exception("Health check failed for service %s", service_type.value)
                service_healths[service_type] = ServiceHealth(
                    service_type=service_type,
                    status=HealthStatus.UNHEALTHY,
                    last_check=UTC,
                    response_time=0.0,
                    last_error=str(e)
                )
                overall_status = HealthStatus.UNHEALTHY

        return ServicesHealthReport(
            overall_status=overall_status,
            services=service_healths,
            check_time=UTC,
            metrics=self._get_system_metrics()
        )

    # Private helper methods
    async def _initialize_service(self, service_type: ServiceType) -> None:
        """Initialize a single service."""
        logger.debug("Initializing service %s", service_type.value)

        try:
            # Get service configuration
            config = self._get_service_config(service_type)

            if not config.enabled:
                logger.info("Service %s is disabled, skipping initialization", service_type.value)
                return

            # Create service instance
            service = self._registry.create_service(service_type, config)

            # Initialize service if it supports initialization
            if hasattr(service, 'initialize'):
                await service.initialize()

            # Store service and update state
            self._services[service_type] = service
            self._service_states[service_type] = "initialized"

            logger.info("Service %s initialized successfully", service_type.value)

        except Exception as e:
            logger.exception("Failed to initialize service %s", service_type.value)
            self._service_states[service_type] = "failed"
            raise

    async def _start_service(self, service_type: ServiceType) -> None:
        """Start a single service."""
        if service_type not in self._services:
            return

        logger.debug("Starting service %s", service_type.value)

        try:
            service = self._services[service_type]

            # Start service if it supports starting
            if hasattr(service, 'start'):
                await service.start()

            self._service_states[service_type] = "running"
            logger.info("Service %s started successfully", service_type.value)

        except Exception as e:
            logger.exception("Failed to start service %s", service_type.value)
            self._service_states[service_type] = "failed"
            raise

    async def _stop_service(self, service_type: ServiceType) -> None:
        """Stop a single service."""
        if service_type not in self._services:
            return

        logger.debug("Stopping service %s", service_type.value)

        try:
            service = self._services[service_type]

            # Stop service if it supports stopping
            if hasattr(service, 'stop'):
                await service.stop()

            # Shutdown service if it supports shutdown
            if hasattr(service, 'shutdown'):
                await service.shutdown()

            self._service_states[service_type] = "stopped"
            logger.info("Service %s stopped successfully", service_type.value)

        except Exception as e:
            logger.exception("Failed to stop service %s", service_type.value)
            # Continue with other services even if one fails to stop

    def _get_enabled_services(self) -> List[ServiceType]:
        """Get list of enabled services."""
        enabled = []

        for service_type in ServiceType:
            config = self._get_service_config(service_type)
            if config and config.enabled:
                enabled.append(service_type)

        return enabled

    def _get_service_config(self, service_type: ServiceType) -> ServiceConfig | None:
        """Get configuration for a service type."""
        config_mapping = {
            ServiceType.CHUNKING: self._config.chunking,
            ServiceType.FILTERING: self._config.filtering,
            # Add other service config mappings
        }

        return config_mapping.get(service_type)

    def _extract_service_dependencies(self, target: Any) -> Dict[str, ServiceType]:
        """Extract service dependencies from target object."""
        dependencies = {}

        # Look for type annotations that indicate service dependencies
        if hasattr(target, '__annotations__'):
            for attr_name, annotation in target.__annotations__.items():
                # Check if annotation is a service protocol
                if hasattr(annotation, '__origin__'):
                    # Handle typing constructs
                    continue

                # Simple mapping based on naming convention
                if 'chunking' in attr_name.lower():
                    dependencies[attr_name] = ServiceType.CHUNKING
                elif 'filtering' in attr_name.lower():
                    dependencies[attr_name] = ServiceType.FILTERING
                # Add other mappings

        return dependencies

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics."""
        return {
            "total_services": len(self._services),
            "running_services": len([s for s in self._service_states.values() if s == "running"]),
            "failed_services": len([s for s in self._service_states.values() if s == "failed"]),
            "uptime": (UTC - self._startup_time).total_seconds() if self._startup_time else 0
        }

    async def _cleanup_partial_initialization(self) -> None:
        """Clean up partially initialized services."""
        for service_type, service in list(self._services.items()):
            try:
                if hasattr(service, 'shutdown'):
                    await service.shutdown()
            except Exception:
                logger.exception("Error during cleanup of service %s", service_type.value)
            finally:
                del self._services[service_type]
                self._service_states[service_type] = "failed"

    async def _cleanup_failed_startup(self) -> None:
        """Clean up after failed startup."""
        await self.stop_services()
```

---

## Phase 4: Plugin Integration Implementation

### 4.1 Updated FileSystem Source

**File:** `src/codeweaver/sources/filesystem.py` (CLEAN REWRITE)

```python
import logging
from typing import Any, List
from pathlib import Path

from codeweaver.sources.base import AbstractDataSource
from codeweaver.types import SourceType
from codeweaver.types import ChunkingService, FilteringService, ValidationService, CacheService
from codeweaver.types import ContentItem, CodeChunk
from codeweaver.types import FileSystemSourceConfig
from codeweaver.types import IndexingError, SourceConfigurationError

logger = logging.getLogger(__name__)

class FileSystemSource(AbstractDataSource):
    """Clean filesystem source with service injection."""

    def __init__(
        self,
        config: FileSystemSourceConfig,
        chunking_service: ChunkingService,
        filtering_service: FilteringService,
        validation_service: ValidationService | None = None,
        cache_service: CacheService | None = None
    ):
        """Initialize filesystem source with injected services."""
        super().__init__(config)

        # Store injected services
        self._chunking_service = chunking_service
        self._filtering_service = filtering_service
        self._validation_service = validation_service
        self._cache_service = cache_service

        # Source-specific state
        self._stats = {
            "files_processed": 0,
            "chunks_created": 0,
            "errors": 0,
            "cache_hits": 0
        }

        logger.debug("FileSystemSource initialized with services")

    @property
    def source_type(self) -> SourceType:
        """Get source type."""
        return SourceType.FILESYSTEM

    async def index_content(
        self,
        path: Path,
        context: dict[str, Any] = None
    ) -> List[ContentItem]:
        """Index filesystem content using injected services."""
        logger.info("Indexing content at %s", path)

        if not path.exists():
            raise IndexingError(path, "Path does not exist")

        if not path.is_dir():
            raise IndexingError(path, "Path is not a directory")

        try:
            # Check cache first if available
            cache_key = None
            if self._cache_service:
                cache_key = self._generate_cache_key(path, context)
                cached_result = await self._cache_service.get(cache_key)
                if cached_result:
                    logger.debug("Cache hit for %s", path)
                    self._stats["cache_hits"] += 1
                    return cached_result

            # Discover files using filtering service
            logger.debug("Discovering files in %s", path)
            files = await self._filtering_service.discover_files(
                path,
                include_patterns=self.config.include_patterns,
                exclude_patterns=self.config.exclude_patterns,
                max_depth=self.config.max_depth,
                follow_symlinks=self.config.follow_symlinks
            )

            logger.debug("Found %d files to process", len(files))

            # Process files in batches for memory efficiency
            content_items = []
            batch_size = getattr(self.config, 'batch_size', 10)

            for i in range(0, len(files), batch_size):
                batch = files[i:i + batch_size]
                batch_items = await self._process_file_batch(batch, context)
                content_items.extend(batch_items)

            # Cache result if cache service available
            if self._cache_service and cache_key:
                await self._cache_service.set(
                    cache_key,
                    content_items,
                    ttl=getattr(self.config, 'cache_ttl', 3600)
                )

            logger.info("Successfully indexed %d files, created %d content items",
                       len(files), len(content_items))

            return content_items

        except Exception as e:
            logger.exception("Failed to index content at %s", path)
            self._stats["errors"] += 1
            raise IndexingError(path, str(e)) from e

    async def get_content(
        self,
        identifier: str,
        context: dict[str, Any] = None
    ) -> ContentItem | None:
        """Get specific content by identifier."""
        try:
            file_path = Path(identifier)

            if not file_path.exists() or not file_path.is_file():
                return None

            # Check if file should be included
            if not self._filtering_service.should_include_file(
                file_path,
                self.config.include_patterns,
                self.config.exclude_patterns
            ):
                logger.debug("File %s excluded by filtering rules", file_path)
                return None

            # Read and process single file
            content_items = await self._process_single_file(file_path, context)

            # Return first item (single file should produce one ContentItem)
            return content_items[0] if content_items else None

        except Exception as e:
            logger.exception("Failed to get content for %s", identifier)
            return None

    async def validate_configuration(self) -> None:
        """Validate source configuration."""
        logger.debug("Validating filesystem source configuration")

        # Validate base path if specified
        if hasattr(self.config, 'base_path') and self.config.base_path:
            base_path = Path(self.config.base_path)
            if not base_path.exists():
                raise SourceConfigurationError(
                    f"Base path does not exist: {base_path}"
                )
            if not base_path.is_dir():
                raise SourceConfigurationError(
                    f"Base path is not a directory: {base_path}"
                )

        # Validate patterns
        if self.config.include_patterns and self.config.exclude_patterns:
            # Check for conflicting patterns
            for include_pattern in self.config.include_patterns:
                if include_pattern in self.config.exclude_patterns:
                    raise SourceConfigurationError(
                        f"Pattern appears in both include and exclude: {include_pattern}"
                    )

        logger.debug("Filesystem source configuration validated successfully")

    async def _process_file_batch(
        self,
        file_paths: List[Path],
        context: dict[str, Any] = None
    ) -> List[ContentItem]:
        """Process a batch of files."""
        content_items = []

        for file_path in file_paths:
            try:
                items = await self._process_single_file(file_path, context)
                content_items.extend(items)
                self._stats["files_processed"] += 1

            except Exception as e:
                logger.warning("Failed to process file %s: %s", file_path, e)
                self._stats["errors"] += 1
                # Continue with other files

        return content_items

    async def _process_single_file(
        self,
        file_path: Path,
        context: dict[str, Any] = None
    ) -> List[ContentItem]:
        """Process a single file into content items."""
        logger.debug("Processing file %s", file_path)

        try:
            # Read file content
            content = await self._read_file_content(file_path)
            if not content.strip():
                logger.debug("Skipping empty file %s", file_path)
                return []

            # Chunk content using chunking service
            chunks = await self._chunking_service.chunk_content(
                content,
                file_path,
                metadata=context
            )

            if not chunks:
                logger.debug("No chunks created for file %s", file_path)
                return []

            # Validate chunks if validation service available
            if self._validation_service:
                validated_chunks = []
                for chunk in chunks:
                    validation_result = await self._validation_service.validate_chunk(chunk)
                    if validation_result.is_valid:
                        validated_chunks.append(chunk)
                    else:
                        logger.debug("Chunk validation failed for %s: %s",
                                   file_path, validation_result.errors)
                chunks = validated_chunks

            # Convert chunks to content items
            content_items = []
            for chunk in chunks:
                content_item = ContentItem(
                    id=f"{file_path}:{chunk.chunk_index}",
                    content=chunk.content,
                    metadata={
                        "file_path": str(file_path),
                        "chunk_index": chunk.chunk_index,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "language": chunk.language,
                        "source_type": self.source_type.value,
                        **chunk.metadata
                    }
                )
                content_items.append(content_item)

            self._stats["chunks_created"] += len(content_items)
            logger.debug("Created %d content items from %s", len(content_items), file_path)

            return content_items

        except Exception as e:
            logger.exception("Error processing file %s", file_path)
            raise IndexingError(file_path, f"File processing failed: {e}") from e

    async def _read_file_content(self, file_path: Path) -> str:
        """Read file content with encoding detection."""
        try:
            # Try UTF-8 first
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try other encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue

            # If all encodings fail, read as binary and decode with errors
            with open(file_path, 'rb') as f:
                return f.read().decode('utf-8', errors='replace')

    def _generate_cache_key(self, path: Path, context: dict[str, Any] = None) -> str:
        """Generate cache key for path and context."""
        import hashlib

        key_parts = [
            str(path),
            str(self.config.include_patterns),
            str(self.config.exclude_patterns),
            str(context) if context else ""
        ]

        key_string = "|".join(key_parts)
        return f"fs_index:{hashlib.md5(key_string.encode()).hexdigest()}"

    def get_stats(self) -> dict[str, Any]:
        """Get source statistics."""
        return self._stats.copy()
```

### 4.2 Updated Source Registry

**File:** `src/codeweaver/factories/source_registry.py` (UPDATE for service injection)

```python
import logging
from typing import TYPE_CHECKING

from codeweaver.factories.base_registry import BaseRegistry
from codeweaver.types import SourceType
from codeweaver.types import SourceConfig
from codeweaver.sources.base import AbstractDataSource
from codeweaver.types import SourceCreationError, SourceNotFoundError

if TYPE_CHECKING:
    from codeweaver.factories.codeweaver_factory import CodeWeaverFactory

logger = logging.getLogger(__name__)

class SourceRegistry(BaseRegistry):
    """Registry for data source providers with service injection."""

    def __init__(self, factory: 'CodeWeaverFactory'):
        """Initialize source registry with factory reference."""
        super().__init__()
        self._factory = factory
        self._register_default_sources()

    def create_source(
        self,
        source_type: SourceType,
        config: SourceConfig
    ) -> AbstractDataSource:
        """Create data source with injected services."""
        logger.debug("Creating source %s", source_type.value)

        try:
            # Get services manager
            services_manager = self._factory.get_services_manager()

            # Resolve required services
            required_services = self._get_required_services(source_type)
            services = services_manager.resolve_dependencies(required_services)

            # Create source with service injection
            if source_type == SourceType.FILESYSTEM:
                from codeweaver.sources.filesystem import FileSystemSource
                return FileSystemSource(
                    config=config,
                    chunking_service=services[ServiceType.CHUNKING],
                    filtering_service=services[ServiceType.FILTERING],
                    validation_service=services.get(ServiceType.VALIDATION),
                    cache_service=services.get(ServiceType.CACHE)
                )

            elif source_type == SourceType.GIT:
                from codeweaver.sources.git import GitSource
                return GitSource(
                    config=config,
                    chunking_service=services[ServiceType.CHUNKING],
                    filtering_service=services[ServiceType.FILTERING],
                    validation_service=services.get(ServiceType.VALIDATION),
                    cache_service=services.get(ServiceType.CACHE)
                )

            else:
                raise SourceNotFoundError(source_type)

        except Exception as e:
            logger.exception("Failed to create source %s", source_type.value)
            raise SourceCreationError(source_type, str(e)) from e

    def _get_required_services(self, source_type: SourceType) -> list[ServiceType]:
        """Get required services for source type."""
        # All sources require chunking and filtering
        required = [ServiceType.CHUNKING, ServiceType.FILTERING]

        # Optional services that sources can use if available
        optional = [ServiceType.VALIDATION, ServiceType.CACHE]

        return required + optional

    def _register_default_sources(self) -> None:
        """Register default source implementations."""
        # Sources are created on-demand through create_source method
        # No need to register classes here since we use factory pattern
        pass
```

---

## Testing Strategy

### Test Structure for Services

**File:** `tests/unit/services/test_services_manager.py`

```python
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from codeweaver.services.services_manager import ServicesManager
from codeweaver.factories.service_registry import ServiceRegistry
from codeweaver.types import ServiceType
from codeweaver.types import ServicesConfig, ChunkingServiceConfig
from codeweaver.types import ServiceInitializationError

class TestServicesManager:
    """Test suite for ServicesManager."""

    @pytest.fixture
    def mock_registry(self):
        """Mock service registry."""
        registry = Mock(spec=ServiceRegistry)
        registry.create_service = Mock()
        return registry

    @pytest.fixture
    def services_config(self):
        """Test services configuration."""
        return ServicesConfig(
            chunking=ChunkingServiceConfig(enabled=True),
            filtering=FilteringServiceConfig(enabled=True)
        )

    @pytest.fixture
    def services_manager(self, mock_registry, services_config):
        """Services manager instance."""
        return ServicesManager(mock_registry, services_config)

    @pytest.mark.asyncio
    async def test_initialize_services_success(self, services_manager, mock_registry):
        """Test successful service initialization."""
        # Setup mocks
        mock_service = AsyncMock()
        mock_service.initialize = AsyncMock()
        mock_registry.create_service.return_value = mock_service

        # Initialize services
        await services_manager.initialize_services()

        # Verify services were created and initialized
        assert mock_registry.create_service.call_count >= 2  # chunking + filtering
        mock_service.initialize.assert_called()

    @pytest.mark.asyncio
    async def test_initialize_services_failure(self, services_manager, mock_registry):
        """Test service initialization failure."""
        # Setup mock to fail
        mock_registry.create_service.side_effect = Exception("Service creation failed")

        # Should raise ServiceInitializationError
        with pytest.raises(ServiceInitializationError):
            await services_manager.initialize_services()

    @pytest.mark.asyncio
    async def test_get_service_success(self, services_manager, mock_registry):
        """Test successful service retrieval."""
        # Setup and initialize
        mock_service = AsyncMock()
        mock_service.initialize = AsyncMock()
        mock_registry.create_service.return_value = mock_service

        await services_manager.initialize_services()

        # Get service
        service = services_manager.get_service(ServiceType.CHUNKING)
        assert service == mock_service

    def test_get_service_not_found(self, services_manager):
        """Test service not found error."""
        with pytest.raises(ServiceNotFoundError):
            services_manager.get_service(ServiceType.CHUNKING)

    @pytest.mark.asyncio
    async def test_health_check(self, services_manager, mock_registry):
        """Test service health checking."""
        # Setup mock service with health check
        mock_service = AsyncMock()
        mock_service.initialize = AsyncMock()
        mock_service.health_check = AsyncMock(return_value=Mock(status=HealthStatus.HEALTHY))
        mock_registry.create_service.return_value = mock_service

        await services_manager.initialize_services()

        # Perform health check
        health_report = await services_manager.health_check()

        assert health_report.overall_status == HealthStatus.HEALTHY
        assert ServiceType.CHUNKING in health_report.services
```

**File:** `tests/integration/test_services_integration.py`

```python
import pytest
import tempfile
from pathlib import Path

from codeweaver.factories.codeweaver_factory import CodeWeaverFactory
from codeweaver.types import ServiceType
from codeweaver.types import ServicesConfig
from codeweaver.sources.filesystem import FileSystemSource

class TestServicesIntegration:
    """Integration tests for services system."""

    @pytest.fixture
    async def factory_with_services(self):
        """Factory with services initialized."""
        factory = CodeWeaverFactory()

        # Configure services
        services_config = ServicesConfig()
        await factory.configure_services(services_config)
        await factory.initialize_services()

        yield factory

        # Cleanup
        await factory.shutdown_services()

    @pytest.fixture
    def temp_codebase(self):
        """Temporary codebase for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            (temp_path / "test.py").write_text("""
def hello_world():
    print("Hello, World!")
    return True

class TestClass:
    def method(self):
        pass
""")

            (temp_path / "test.js").write_text("""
function helloWorld() {
    console.log("Hello, World!");
    return true;
}

class TestClass {
    method() {
        // implementation
    }
}
""")

            yield temp_path

    @pytest.mark.asyncio
    async def test_end_to_end_indexing(self, factory_with_services, temp_codebase):
        """Test end-to-end indexing with services."""
        # Create filesystem source
        source = factory_with_services.create_source_with_services(
            SourceType.FILESYSTEM,
            FileSystemSourceConfig(
                include_patterns=["*.py", "*.js"],
                exclude_patterns=["*.pyc"]
            )
        )

        # Index content
        content_items = await source.index_content(temp_codebase)

        # Verify results
        assert len(content_items) > 0

        # Check that content was properly chunked
        python_items = [item for item in content_items if "test.py" in item.metadata["file_path"]]
        js_items = [item for item in content_items if "test.js" in item.metadata["file_path"]]

        assert len(python_items) > 0
        assert len(js_items) > 0

        # Verify metadata
        for item in content_items:
            assert "file_path" in item.metadata
            assert "chunk_index" in item.metadata
            assert "source_type" in item.metadata
            assert item.metadata["source_type"] == "filesystem"

    @pytest.mark.asyncio
    async def test_service_health_monitoring(self, factory_with_services):
        """Test service health monitoring."""
        services_manager = factory_with_services.get_services_manager()

        # Start services
        await services_manager.start_services()

        # Check health
        health_report = await services_manager.health_check()

        assert health_report.overall_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        assert ServiceType.CHUNKING in health_report.services
        assert ServiceType.FILTERING in health_report.services

    @pytest.mark.asyncio
    async def test_service_dependency_injection(self, factory_with_services):
        """Test service dependency injection."""
        services_manager = factory_with_services.get_services_manager()

        # Get services
        chunking_service = services_manager.get_service(ServiceType.CHUNKING)
        filtering_service = services_manager.get_service(ServiceType.FILTERING)

        # Verify services implement protocols
        assert hasattr(chunking_service, 'chunk_content')
        assert hasattr(chunking_service, 'detect_language')
        assert hasattr(filtering_service, 'discover_files')
        assert hasattr(filtering_service, 'should_include_file')

    @pytest.mark.asyncio
    async def test_service_reconfiguration(self, factory_with_services):
        """Test runtime service reconfiguration."""
        services_manager = factory_with_services.get_services_manager()

        # Get original configuration
        original_config = services_manager.get_service_configuration(ServiceType.CHUNKING)

        # Create new configuration
        new_config = ChunkingServiceConfig(
            max_chunk_size=2000,  # Different from default
            min_chunk_size=100
        )

        # Reconfigure service
        await services_manager.reconfigure_service(ServiceType.CHUNKING, new_config)

        # Verify configuration changed
        updated_config = services_manager.get_service_configuration(ServiceType.CHUNKING)
        assert updated_config.max_chunk_size == 2000
        assert updated_config.min_chunk_size == 100
```

---

## Conclusion

This implementation specification provides a complete roadmap for implementing CodeWeaver's services registry and abstraction layer. The approach ensures:

**✅ Clean Architecture**: No legacy dependencies, pure factory pattern integration
**✅ Type Safety**: Full protocol-based interfaces with runtime validation
**✅ Extensibility**: Easy addition of new services and providers
**✅ Testability**: Comprehensive test coverage with mocking and integration tests
**✅ Observability**: Built-in health monitoring and performance metrics
**✅ Configuration**: Flexible, hierarchical configuration system

The phased implementation approach allows for incremental development while maintaining system stability and clear separation of concerns.
