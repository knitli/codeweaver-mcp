<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Services Clean Implementation Guide

**CodeWeaver MCP Server - Services Architecture Implementation Guide**

*Document Version: 1.0*
*Implementation Type: Clean (No Legacy Migration)*
*Guide Date: 2025-07-26*
*Target Audience: Developers implementing the services layer*

---

## Implementation Overview

This guide provides step-by-step instructions for implementing CodeWeaver's services registry and abstraction layer as a clean implementation. Since CodeWeaver is pre-release, there's no legacy code to preserve - this allows for optimal architecture without compromise.

**Implementation Benefits:**
- 🚀 **Zero Technical Debt**: Clean implementation from ground up
- 🏗️ **Optimal Architecture**: Factory pattern with dependency injection
- 🧪 **Test-Driven**: Tests define contracts and drive implementation
- 📊 **Observable**: Built-in monitoring and health checking
- ⚡ **Performance**: Optimized for efficiency and scalability

---

## Prerequisites & Setup

### Development Environment

```bash
# Ensure you have the latest CodeWeaver development environment
cd /path/to/codeweaver-mcp

# Install development dependencies
uv sync --group dev

# Activate virtual environment
source .venv/bin/activate

# Verify environment
uv run python -c "import codeweaver; print('Environment ready')"
```

### Code Quality Setup

```bash
# Run initial code quality checks
uv run ruff check src/
uv run ruff format src/

# Run existing tests to ensure baseline
uv run pytest tests/ -v
```

---

## Phase 1: Core Type System (Day 1)

### Step 1.1: Extend ComponentType Enum

**File:** `src/codeweaver/_types/config.py`

```python
# Add to existing ComponentType enum
class ComponentType(BaseEnum):
    BACKEND = "backend"
    PROVIDER = "provider"
    SOURCE = "source"
    SERVICE = "service"      # ADD THIS
    MIDDLEWARE = "middleware" # ADD THIS
    FACTORY = "factory"
    PLUGIN = "plugin"

# ADD new ServiceType enum
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
        """Core services required for basic operation."""
        return [cls.CHUNKING, cls.FILTERING]

    @classmethod
    def get_optional_services(cls) -> list['ServiceType']:
        """Optional services for enhanced functionality."""
        return [cls.VALIDATION, cls.CACHE, cls.MONITORING, cls.METRICS]
```

**Test Implementation:**

```bash
# Create test file
touch tests/unit/test_service_types.py
```

**File:** `tests/unit/test_service_types.py`

```python
import pytest
from codeweaver.types import ServiceType, ComponentType

class TestServiceTypes:
    """Test service type definitions."""

    def test_service_type_values(self):
        """Test service type enum values."""
        assert ServiceType.CHUNKING.value == "chunking"
        assert ServiceType.FILTERING.value == "filtering"
        assert ServiceType.VALIDATION.value == "validation"
        assert ServiceType.CACHE.value == "cache"

    def test_core_services(self):
        """Test core services classification."""
        core = ServiceType.get_core_services()
        assert ServiceType.CHUNKING in core
        assert ServiceType.FILTERING in core
        assert len(core) == 2

    def test_optional_services(self):
        """Test optional services classification."""
        optional = ServiceType.get_optional_services()
        assert ServiceType.VALIDATION in optional
        assert ServiceType.CACHE in optional
        assert ServiceType.MONITORING in optional
        assert ServiceType.METRICS in optional

    def test_component_type_extended(self):
        """Test extended component types."""
        assert ComponentType.SERVICE.value == "service"
        assert ComponentType.MIDDLEWARE.value == "middleware"

# Run test
# uv run pytest tests/unit/test_service_types.py -v
```

### Step 1.2: Create Service Data Structures

**File:** `src/codeweaver/_types/service_data.py` (NEW)

```python
from typing import Any, Annotated
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path

from codeweaver.types import BaseEnum

# Health monitoring types
class HealthStatus(BaseEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class ServiceHealth(BaseModel):
    """Health status of a service."""

    service_type: Annotated['ServiceType', Field(description="Type of service")]
    status: Annotated[HealthStatus, Field(description="Current health status")]
    last_check: Annotated[datetime, Field(description="Last health check timestamp")]
    response_time: Annotated[float, Field(ge=0, description="Last response time in seconds")]
    error_count: Annotated[int, Field(ge=0, description="Number of recent errors")] = 0
    success_rate: Annotated[float, Field(ge=0, le=1, description="Success rate (0-1)")] = 1.0
    last_error: Annotated[str | None, Field(description="Last error message")] = None
    uptime: Annotated[float, Field(ge=0, description="Service uptime in seconds")] = 0.0
    memory_usage: Annotated[int, Field(ge=0, description="Memory usage in bytes")] = 0

# Statistics types
class ChunkingStats(BaseModel):
    """Statistics for chunking operations."""

    total_files_processed: Annotated[int, Field(ge=0, description="Total files processed")] = 0
    total_chunks_created: Annotated[int, Field(ge=0, description="Total chunks created")] = 0
    average_chunk_size: Annotated[float, Field(ge=0, description="Average chunk size")] = 0.0
    total_processing_time: Annotated[float, Field(ge=0, description="Total processing time")] = 0.0
    languages_processed: Annotated[dict[str, int], Field(description="Files by language")] = Field(default_factory=dict)
    error_count: Annotated[int, Field(ge=0, description="Number of errors")] = 0
    success_rate: Annotated[float, Field(ge=0, le=1, description="Success rate")] = 1.0

class FilteringStats(BaseModel):
    """Statistics for filtering operations."""

    total_files_scanned: Annotated[int, Field(ge=0, description="Total files scanned")] = 0
    total_files_included: Annotated[int, Field(ge=0, description="Total files included")] = 0
    total_files_excluded: Annotated[int, Field(ge=0, description="Total files excluded")] = 0
    total_directories_scanned: Annotated[int, Field(ge=0, description="Total directories scanned")] = 0
    total_scan_time: Annotated[float, Field(ge=0, description="Total scan time")] = 0.0
    patterns_matched: Annotated[dict[str, int], Field(description="Files matched by pattern")] = Field(default_factory=dict)
    error_count: Annotated[int, Field(ge=0, description="Number of errors")] = 0

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
```

**Test Implementation:**

```bash
# Create test file
touch tests/unit/test_service_data.py
```

**File:** `tests/unit/test_service_data.py`

```python
import pytest
from datetime import datetime
from pathlib import Path

from codeweaver.types import (
    ServiceHealth, ChunkingStats, FilteringStats, FileMetadata, HealthStatus
)
from codeweaver.types import ServiceType

class TestServiceData:
    """Test service data structures."""

    def test_service_health_creation(self):
        """Test ServiceHealth model creation."""
        health = ServiceHealth(
            service_type=ServiceType.CHUNKING,
            status=HealthStatus.HEALTHY,
            last_check=UTC,
            response_time=0.5,
            error_count=0,
            success_rate=1.0
        )

        assert health.service_type == ServiceType.CHUNKING
        assert health.status == HealthStatus.HEALTHY
        assert health.response_time == 0.5
        assert health.error_count == 0
        assert health.success_rate == 1.0

    def test_chunking_stats_defaults(self):
        """Test ChunkingStats default values."""
        stats = ChunkingStats()

        assert stats.total_files_processed == 0
        assert stats.total_chunks_created == 0
        assert stats.average_chunk_size == 0.0
        assert stats.success_rate == 1.0
        assert stats.error_count == 0

    def test_file_metadata_creation(self):
        """Test FileMetadata model creation."""
        metadata = FileMetadata(
            path=Path("/test/file.py"),
            size=1024,
            file_type="python",
            is_binary=False
        )

        assert metadata.path == Path("/test/file.py")
        assert metadata.size == 1024
        assert metadata.file_type == "python"
        assert metadata.is_binary is False

# Run test
# uv run pytest tests/unit/test_service_data.py -v
```

### Step 1.3: Create Service Exceptions

**File:** `src/codeweaver/_types/service_exceptions.py` (NEW)

```python
from pathlib import Path
from codeweaver.types import CodeWeaverError
from codeweaver.types import ServiceType

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

class ServiceInitializationError(ServiceError):
    """Exception raised when service initialization fails."""

    def __init__(self, message: str):
        super().__init__(f"Service initialization failed: {message}")

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

class UnsupportedLanguageError(ChunkingError):
    """Exception raised for unsupported programming languages."""

    def __init__(self, file_path: Path, language: str):
        super().__init__(file_path, f"Unsupported language: {language}")
        self.language = language
```

**Validation:**

```bash
# Test imports work correctly
uv run python -c "
from codeweaver.types import ServiceError, ServiceNotFoundError
from codeweaver.types import ServiceType
print('Service exceptions imported successfully')
"
```

---

## Phase 2: Service Protocols (Day 2)

### Step 2.1: Create Service Protocol Interfaces

**File:** `src/codeweaver/_types/services.py` (NEW)

```python
from typing import Protocol, runtime_checkable, AsyncGenerator, Any
from pathlib import Path
from abc import abstractmethod

from codeweaver.types import CodeChunk, ContentItem
from codeweaver.types import ChunkingStats, FilteringStats, FileMetadata
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
```

**Test Protocol Compliance:**

**File:** `tests/unit/test_service_protocols.py` (NEW)

```python
import pytest
from typing import get_type_hints

from codeweaver.types import ServiceProvider, ChunkingService, FilteringService

class TestServiceProtocols:
    """Test service protocol definitions."""

    def test_service_provider_protocol(self):
        """Test ServiceProvider protocol methods."""
        # Get protocol methods
        hints = get_type_hints(ServiceProvider)

        # Check required properties exist
        assert hasattr(ServiceProvider, 'name')
        assert hasattr(ServiceProvider, 'version')
        assert hasattr(ServiceProvider, 'initialize')
        assert hasattr(ServiceProvider, 'shutdown')
        assert hasattr(ServiceProvider, 'health_check')

    def test_chunking_service_protocol(self):
        """Test ChunkingService protocol methods."""
        assert hasattr(ChunkingService, 'chunk_content')
        assert hasattr(ChunkingService, 'detect_language')
        assert hasattr(ChunkingService, 'get_supported_languages')
        assert hasattr(ChunkingService, 'get_chunking_stats')

    def test_filtering_service_protocol(self):
        """Test FilteringService protocol methods."""
        assert hasattr(FilteringService, 'discover_files')
        assert hasattr(FilteringService, 'should_include_file')
        assert hasattr(FilteringService, 'get_file_metadata')
        assert hasattr(FilteringService, 'get_filtering_stats')

    def test_runtime_checkable(self):
        """Test protocols are runtime checkable."""
        # Create mock implementations
        class MockServiceProvider:
            @property
            def name(self) -> str:
                return "mock"

            @property
            def version(self) -> str:
                return "1.0.0"

            async def initialize(self) -> None:
                pass

            async def shutdown(self) -> None:
                pass

            async def health_check(self) -> bool:
                return True

        mock = MockServiceProvider()

        # Test isinstance works with protocol
        assert isinstance(mock, ServiceProvider)

# Run test
# uv run pytest tests/unit/test_service_protocols.py -v
```

---

## Phase 3: Service Configuration (Day 3)

### Step 3.1: Create Service Configuration Types

**File:** `src/codeweaver/_types/service_config.py` (NEW)

```python
from typing import Annotated, Any
from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path

from codeweaver.types import ChunkingStrategy, PerformanceMode

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

    global_timeout: Annotated[float, Field(gt=0, description="Global timeout")] = 300.0
    health_check_enabled: Annotated[bool, Field(description="Enable health checks")] = True
    metrics_enabled: Annotated[bool, Field(description="Enable metrics")] = True
    auto_recovery: Annotated[bool, Field(description="Enable auto recovery")] = True
```

**Test Configuration:**

**File:** `tests/unit/test_service_config.py` (NEW)

```python
import pytest
from pydantic import ValidationError

from codeweaver.types import (
    ServiceConfig, ChunkingServiceConfig, FilteringServiceConfig, ServicesConfig
)
from codeweaver.types import ChunkingStrategy, PerformanceMode

class TestServiceConfig:
    """Test service configuration models."""

    def test_base_service_config(self):
        """Test base ServiceConfig creation."""
        config = ServiceConfig(provider="test_provider")

        assert config.enabled is True
        assert config.provider == "test_provider"
        assert config.priority == 50
        assert config.timeout == 30.0
        assert config.max_retries == 3

    def test_chunking_service_config(self):
        """Test ChunkingServiceConfig creation."""
        config = ChunkingServiceConfig()

        assert config.provider == "fastmcp_chunking"
        assert config.max_chunk_size == 1500
        assert config.min_chunk_size == 50
        assert config.ast_grep_enabled is True
        assert config.fallback_strategy == ChunkingStrategy.SIMPLE
        assert config.performance_mode == PerformanceMode.BALANCED

    def test_chunking_config_validation(self):
        """Test ChunkingServiceConfig validation."""
        # Test valid config
        config = ChunkingServiceConfig(
            max_chunk_size=2000,
            min_chunk_size=100
        )
        assert config.max_chunk_size == 2000
        assert config.min_chunk_size == 100

        # Test invalid values
        with pytest.raises(ValidationError):
            ChunkingServiceConfig(max_chunk_size=0)  # Must be > 0

        with pytest.raises(ValidationError):
            ChunkingServiceConfig(max_chunk_size=20000)  # Must be <= 10000

    def test_filtering_service_config(self):
        """Test FilteringServiceConfig creation."""
        config = FilteringServiceConfig(
            include_patterns=["*.py", "*.js"],
            exclude_patterns=["*.pyc", "*.log"],
            max_file_size=2 * 1024 * 1024  # 2MB
        )

        assert config.provider == "fastmcp_filtering"
        assert config.include_patterns == ["*.py", "*.js"]
        assert config.exclude_patterns == ["*.pyc", "*.log"]
        assert config.max_file_size == 2 * 1024 * 1024
        assert config.follow_symlinks is False
        assert config.ignore_hidden is True

    def test_services_config(self):
        """Test ServicesConfig creation."""
        config = ServicesConfig()

        assert isinstance(config.chunking, ChunkingServiceConfig)
        assert isinstance(config.filtering, FilteringServiceConfig)
        assert config.global_timeout == 300.0
        assert config.health_check_enabled is True
        assert config.metrics_enabled is True
        assert config.auto_recovery is True

    def test_services_config_custom(self):
        """Test ServicesConfig with custom settings."""
        chunking_config = ChunkingServiceConfig(max_chunk_size=2000)
        filtering_config = FilteringServiceConfig(max_file_size=5 * 1024 * 1024)

        config = ServicesConfig(
            chunking=chunking_config,
            filtering=filtering_config,
            health_check_enabled=False
        )

        assert config.chunking.max_chunk_size == 2000
        assert config.filtering.max_file_size == 5 * 1024 * 1024
        assert config.health_check_enabled is False

# Run test
# uv run pytest tests/unit/test_service_config.py -v
```

### Step 3.2: Update Main Configuration

**File:** `src/codeweaver/config.py` (UPDATE)

```python
# Add to existing CodeWeaverConfig class

from codeweaver.types import ServicesConfig

class CodeWeaverConfig(BaseModel):
    """Extended root configuration with services."""

    # Existing configurations
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_backend: VectorBackendConfig = Field(default_factory=VectorBackendConfig)

    # NEW: Services configuration
    services: ServicesConfig = Field(default_factory=ServicesConfig)

    # Factory configuration
    factory: FactoryConfig = Field(default_factory=FactoryConfig)
```

---

## Phase 4: Service Registry Implementation (Day 4-5)

### Step 4.1: Create Service Registry

**File:** `src/codeweaver/factories/service_registry.py` (NEW)

```python
import logging
from typing import Any, Dict, Type, TYPE_CHECKING
from collections import defaultdict
from datetime import datetime

from codeweaver.types import ServiceType
from codeweaver.types import ServiceProvider
from codeweaver.types import ServiceConfig
from codeweaver.types import (
    ServiceNotFoundError, ServiceCreationError, ProviderRegistrationError
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
        self._instances: dict[ServiceType, Any] = {}
        self._configs: dict[ServiceType, ServiceConfig] = {}

        logger.debug("ServiceRegistry initialized")

    def register_provider(
        self,
        service_type: ServiceType,
        provider_name: str,
        provider_class: Type[ServiceProvider]
    ) -> None:
        """Register a service provider."""
        logger.debug("Registering provider %s for service %s", provider_name, service_type.value)

        # Store provider
        self._providers[service_type][provider_name] = provider_class

        logger.info("Successfully registered provider %s for service %s", provider_name, service_type.value)

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
            raise ServiceCreationError(service_type, f"Provider not found: {provider_name}")

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

    def _get_default_provider(self, service_type: ServiceType, config: ServiceConfig = None) -> str:
        """Get default provider for service type."""
        providers = self._providers[service_type]
        if not providers:
            raise ServiceCreationError(service_type, "No providers registered")

        # If config specifies provider, use it
        if config and hasattr(config, 'provider'):
            return config.provider

        # Return first available provider
        return next(iter(providers.keys()))

    def _create_default_config(self, service_type: ServiceType) -> ServiceConfig | None:
        """Create default configuration for service type."""
        from codeweaver.types import ChunkingServiceConfig, FilteringServiceConfig

        config_map = {
            ServiceType.CHUNKING: ChunkingServiceConfig,
            ServiceType.FILTERING: FilteringServiceConfig,
        }

        config_class = config_map.get(service_type)
        return config_class() if config_class else None
```

**Test Service Registry:**

**File:** `tests/unit/test_service_registry.py` (NEW)

```python
import pytest
from unittest.mock import Mock

from codeweaver.factories.service_registry import ServiceRegistry
from codeweaver.types import ServiceType
from codeweaver.types import ChunkingServiceConfig
from codeweaver.types import ServiceCreationError, ServiceNotFoundError

class MockServiceProvider:
    """Mock service provider for testing."""

    def __init__(self, config=None):
        self.config = config

    @property
    def name(self) -> str:
        return "mock_provider"

    @property
    def version(self) -> str:
        return "1.0.0"

class TestServiceRegistry:
    """Test service registry functionality."""

    @pytest.fixture
    def mock_factory(self):
        """Mock factory for testing."""
        return Mock()

    @pytest.fixture
    def service_registry(self, mock_factory):
        """Service registry instance."""
        return ServiceRegistry(mock_factory)

    def test_register_provider(self, service_registry):
        """Test provider registration."""
        service_registry.register_provider(
            ServiceType.CHUNKING,
            "mock_provider",
            MockServiceProvider
        )

        # Verify provider is registered
        assert ServiceType.CHUNKING in service_registry._providers
        assert "mock_provider" in service_registry._providers[ServiceType.CHUNKING]
        assert service_registry._providers[ServiceType.CHUNKING]["mock_provider"] == MockServiceProvider

    def test_create_service(self, service_registry):
        """Test service creation."""
        # Register provider
        service_registry.register_provider(
            ServiceType.CHUNKING,
            "mock_provider",
            MockServiceProvider
        )

        # Create service
        service = service_registry.create_service(ServiceType.CHUNKING)

        assert isinstance(service, MockServiceProvider)
        assert service.name == "mock_provider"

    def test_create_service_with_config(self, service_registry):
        """Test service creation with configuration."""
        # Register provider
        service_registry.register_provider(
            ServiceType.CHUNKING,
            "mock_provider",
            MockServiceProvider
        )

        # Create service with config
        config = ChunkingServiceConfig(max_chunk_size=2000)
        service = service_registry.create_service(ServiceType.CHUNKING, config)

        assert isinstance(service, MockServiceProvider)
        assert service.config == config
        assert service.config.max_chunk_size == 2000

    def test_get_service_cached(self, service_registry):
        """Test cached service retrieval."""
        # Register provider
        service_registry.register_provider(
            ServiceType.CHUNKING,
            "mock_provider",
            MockServiceProvider
        )

        # Get service (should create and cache)
        service1 = service_registry.get_service(ServiceType.CHUNKING)
        service2 = service_registry.get_service(ServiceType.CHUNKING)

        # Should return same instance
        assert service1 is service2

    def test_service_not_found(self, service_registry):
        """Test service not found error."""
        with pytest.raises(ServiceNotFoundError):
            service_registry.get_service(ServiceType.CHUNKING, create_if_missing=False)

    def test_no_provider_registered(self, service_registry):
        """Test error when no provider registered."""
        with pytest.raises(ServiceCreationError):
            service_registry.create_service(ServiceType.CHUNKING)

# Run test
# uv run pytest tests/unit/test_service_registry.py -v
```

### Step 4.2: Create Base Service Provider

**Directory Setup:**

```bash
# Create services directory structure
mkdir -p src/codeweaver/services
mkdir -p src/codeweaver/services/providers
mkdir -p src/codeweaver/services/providers/chunking
mkdir -p src/codeweaver/services/providers/filtering

# Create __init__.py files
touch src/codeweaver/services/__init__.py
touch src/codeweaver/services/providers/__init__.py
touch src/codeweaver/services/providers/chunking/__init__.py
touch src/codeweaver/services/providers/filtering/__init__.py
```

**File:** `src/codeweaver/services/providers/base_provider.py` (NEW)

```python
import logging
from abc import ABC, abstractmethod
from typing import Any
from datetime import datetime

from codeweaver.types import ServiceConfig
from codeweaver.types import ServiceHealth, HealthStatus
from codeweaver.types import ServiceType

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

        logger.debug("BaseServiceProvider initialized for %s", self.name)

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

    async def health_check(self) -> ServiceHealth:
        """Check service health."""
        try:
            status = HealthStatus.HEALTHY if self._running else HealthStatus.DEGRADED

            health = ServiceHealth(
                service_type=self._get_service_type(),
                status=status,
                last_check=UTC,
                response_time=0.0,
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

    # Abstract methods for implementation
    @abstractmethod
    async def _initialize_impl(self) -> None:
        """Implementation-specific initialization."""
        ...

    @abstractmethod
    async def _shutdown_impl(self) -> None:
        """Implementation-specific shutdown."""
        ...

    @abstractmethod
    def _get_service_type(self) -> ServiceType:
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

---

## Phase 5: Implement Service Providers (Day 6-8)

### Step 5.1: FastMCP Chunking Provider

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
```

**Test Chunking Provider:**

**File:** `tests/unit/services/providers/test_chunking_providers.py` (NEW)

```python
import pytest
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from codeweaver.services.providers.chunking.fastmcp_provider import ChunkingService
from codeweaver.types import ChunkingServiceConfig
from codeweaver.types import ChunkingStrategy, Language
from codeweaver.types import ChunkingError

class TestChunkingService:
    """Test FastMCP chunking provider."""

    @pytest.fixture
    def chunking_config(self):
        """Chunking service configuration."""
        return ChunkingServiceConfig(
            max_chunk_size=1500,
            min_chunk_size=50,
            ast_grep_enabled=True
        )

    @pytest.fixture
    def mock_middleware(self):
        """Mock ChunkingMiddleware."""
        middleware = AsyncMock()
        middleware.chunk_file_content.return_value = [
            {
                "content": "def hello():\n    pass",
                "start_line": 1,
                "end_line": 2,
                "language": "python",
                "metadata": {}
            }
        ]
        middleware.detect_language.return_value = "python"
        middleware.get_supported_languages.return_value = {
            "python": {"ast_support": True, "tree_sitter": True}
        }
        return middleware

    @pytest.fixture
    def chunking_provider(self, chunking_config):
        """Chunking provider instance."""
        return ChunkingService(chunking_config)

    def test_provider_properties(self, chunking_provider):
        """Test provider properties."""
        assert chunking_provider.name == "fastmcp_chunking"
        assert chunking_provider.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_initialize_provider(self, chunking_provider, mock_middleware):
        """Test provider initialization."""
        with patch('codeweaver.services.providers.chunking.fastmcp_provider.ChunkingMiddleware') as mock_class:
            mock_class.return_value = mock_middleware

            await chunking_provider.initialize()

            assert chunking_provider._initialized
            mock_middleware.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_chunk_content(self, chunking_provider, mock_middleware):
        """Test content chunking."""
        # Setup
        chunking_provider._middleware = mock_middleware
        chunking_provider._initialized = True

        # Test chunking
        content = "def hello():\n    pass"
        file_path = Path("test.py")

        chunks = await chunking_provider.chunk_content(content, file_path)

        assert len(chunks) == 1
        assert chunks[0].content == "def hello():\n    pass"
        assert chunks[0].file_path == file_path
        assert chunks[0].language == "python"

        # Verify middleware was called
        mock_middleware.chunk_file_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_chunk_content_without_initialization(self, chunking_provider):
        """Test chunking without initialization raises error."""
        content = "def hello():\n    pass"
        file_path = Path("test.py")

        with pytest.raises(ChunkingError):
            await chunking_provider.chunk_content(content, file_path)

    def test_detect_language(self, chunking_provider, mock_middleware):
        """Test language detection."""
        chunking_provider._middleware = mock_middleware

        language = chunking_provider.detect_language(Path("test.py"))

        assert language == Language.PYTHON
        mock_middleware.detect_language.assert_called_once()

    def test_get_supported_languages(self, chunking_provider, mock_middleware):
        """Test getting supported languages."""
        chunking_provider._middleware = mock_middleware

        languages = chunking_provider.get_supported_languages()

        assert Language.PYTHON in languages
        assert languages[Language.PYTHON]["ast_support"] is True

# Run test
# uv run pytest tests/unit/services/providers/test_chunking_providers.py -v
```

**Register the Provider:**

**File:** `src/codeweaver/services/providers/chunking/__init__.py`

```python
"""Chunking service providers."""

from .fastmcp_provider import ChunkingService

__all__ = ['ChunkingService']
```

---

## Phase 6: Services Manager (Day 9-10)

### Step 6.1: Create Services Manager

**File:** `src/codeweaver/services/services_manager.py` (NEW)

```python
import logging
import asyncio
from typing import Any, Dict, List
from datetime import datetime

from codeweaver.types import ServiceType
from codeweaver.types import ServicesConfig, ServiceConfig
from codeweaver.types import ServicesHealthReport, ServiceHealth, HealthStatus
from codeweaver.types import (
    ServiceInitializationError, ServiceNotFoundError, ServiceNotReadyError
)
from codeweaver.factories.service_registry import ServiceRegistry

logger = logging.getLogger(__name__)

class ServicesManager:
    """Central manager for service lifecycle and dependencies."""

    def __init__(self, registry: ServiceRegistry, config: ServicesConfig):
        """Initialize services manager."""
        self._registry = registry
        self._config = config

        self._services: Dict[ServiceType, Any] = {}
        self._service_states: Dict[ServiceType, str] = {}
        self._initialization_order: List[ServiceType] = []
        self._startup_time = None

        logger.debug("ServicesManager initialized")

    async def initialize_services(self) -> None:
        """Initialize all configured services."""
        logger.info("Initializing services...")

        try:
            # Register default providers
            self._register_default_providers()

            # Determine initialization order
            self._initialization_order = self._get_enabled_services()

            logger.debug("Service initialization order: %s", [s.value for s in self._initialization_order])

            # Initialize services in order
            for service_type in self._initialization_order:
                await self._initialize_service(service_type)

            logger.info("All services initialized successfully")

        except Exception as e:
            logger.exception("Failed to initialize services")
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

            logger.info("All services started successfully")

        except Exception as e:
            logger.exception("Failed to start services")
            raise ServiceInitializationError(f"Service startup failed: {e}") from e

    async def stop_services(self) -> None:
        """Stop all running services."""
        logger.info("Stopping services...")

        try:
            # Stop services in reverse order
            for service_type in reversed(self._initialization_order):
                await self._stop_service(service_type)

            logger.info("All services stopped successfully")

        except Exception as e:
            logger.exception("Failed to stop services gracefully")

    def get_service(self, service_type: ServiceType) -> Any:
        """Get a service instance."""
        if service_type not in self._services:
            raise ServiceNotFoundError(service_type)

        service = self._services[service_type]
        state = self._service_states.get(service_type, "unknown")

        if state not in ["initialized", "running"]:
            raise ServiceNotReadyError(service_type, state)

        return service

    def resolve_dependencies(self, dependencies: List[ServiceType]) -> Dict[ServiceType, Any]:
        """Resolve multiple service dependencies."""
        try:
            resolved = {}
            for service_type in dependencies:
                resolved[service_type] = self.get_service(service_type)
            return resolved

        except Exception as e:
            logger.exception("Failed to resolve dependencies: %s", dependencies)
            raise

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

    def _register_default_providers(self) -> None:
        """Register default service providers."""
        # Register chunking providers
        from codeweaver.services.providers.chunking import ChunkingService
        self._registry.register_provider(
            ServiceType.CHUNKING,
            "fastmcp_chunking",
            ChunkingService
        )

        # Register filtering providers
        from codeweaver.services.providers.filtering import FilteringService
        self._registry.register_provider(
            ServiceType.FILTERING,
            "fastmcp_filtering",
            FilteringService
        )

        logger.debug("Default service providers registered")

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

    def _get_enabled_services(self) -> List[ServiceType]:
        """Get list of enabled services."""
        enabled = []

        for service_type in ServiceType.get_core_services():
            config = self._get_service_config(service_type)
            if config and config.enabled:
                enabled.append(service_type)

        return enabled

    def _get_service_config(self, service_type: ServiceType) -> ServiceConfig | None:
        """Get configuration for a service type."""
        config_mapping = {
            ServiceType.CHUNKING: self._config.chunking,
            ServiceType.FILTERING: self._config.filtering,
        }

        return config_mapping.get(service_type)

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
```

---

## Phase 7: Factory Integration (Day 11)

### Step 7.1: Update CodeWeaver Factory

**File:** `src/codeweaver/factories/codeweaver_factory.py` (UPDATE)

```python
# Add to existing CodeWeaverFactory class

from codeweaver.types import ServiceType
from codeweaver.types import ServicesConfig, ServiceConfig
from codeweaver.factories.service_registry import ServiceRegistry
from codeweaver.services.services_manager import ServicesManager

class CodeWeaverFactory:
    """Extended factory with services integration."""

    def __init__(self, config: CodeWeaverConfig = None):
        """Initialize factory with services support."""
        # Existing initialization code...

        # NEW: Service registry and manager
        self._service_registry = ServiceRegistry(self)
        self._services_manager = None

        logger.debug("CodeWeaverFactory initialized with services support")

    def create_service(
        self,
        service_type: ServiceType,
        config: ServiceConfig = None,
        provider_name: str = None
    ) -> Any:
        """Create a service instance through the factory."""
        return self._service_registry.create_service(service_type, config, provider_name)

    def get_services_manager(self) -> ServicesManager:
        """Get the services manager instance."""
        if not self._services_manager:
            self._services_manager = ServicesManager(
                self._service_registry,
                self._config.services
            )
        return self._services_manager

    def create_source_with_services(
        self,
        source_type: SourceType,
        config: SourceConfig
    ) -> AbstractDataSource:
        """Create a data source with service dependencies injected."""
        return self._source_registry.create_source(source_type, config)

    async def configure_services(self, services_config: ServicesConfig) -> None:
        """Configure all services."""
        self._config.services = services_config

        # Update existing services manager if it exists
        if self._services_manager:
            self._services_manager._config = services_config

    async def initialize_services(self) -> None:
        """Initialize all configured services."""
        manager = self.get_services_manager()
        await manager.initialize_services()

    async def start_services(self) -> None:
        """Start all services."""
        manager = self.get_services_manager()
        await manager.start_services()

    async def shutdown_services(self) -> None:
        """Shutdown all services gracefully."""
        if self._services_manager:
            await self._services_manager.stop_services()
```

---

## Phase 8: Testing & Validation (Day 12-13)

### Step 8.1: Integration Tests

**File:** `tests/integration/test_services_integration.py` (NEW)

```python
import pytest
import tempfile
from pathlib import Path

from codeweaver.factories.codeweaver_factory import CodeWeaverFactory
from codeweaver.types import ServiceType, SourceType
from codeweaver.types import ServicesConfig

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
        await factory.start_services()

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

            yield temp_path

    @pytest.mark.asyncio
    async def test_services_initialization(self, factory_with_services):
        """Test services are properly initialized."""
        services_manager = factory_with_services.get_services_manager()

        # Check that core services are available
        chunking_service = services_manager.get_service(ServiceType.CHUNKING)
        filtering_service = services_manager.get_service(ServiceType.FILTERING)

        assert chunking_service is not None
        assert filtering_service is not None

        # Verify they implement protocols
        assert hasattr(chunking_service, 'chunk_content')
        assert hasattr(filtering_service, 'discover_files')

    @pytest.mark.asyncio
    async def test_end_to_end_service_usage(self, factory_with_services, temp_codebase):
        """Test end-to-end service usage through filesystem source."""
        # Get services manager
        services_manager = factory_with_services.get_services_manager()

        # Get individual services
        chunking_service = services_manager.get_service(ServiceType.CHUNKING)
        filtering_service = services_manager.get_service(ServiceType.FILTERING)

        # Test filtering service
        files = await filtering_service.discover_files(
            temp_codebase,
            include_patterns=["*.py"],
            exclude_patterns=[]
        )

        assert len(files) == 1
        assert files[0].name == "test.py"

        # Test chunking service
        content = (temp_codebase / "test.py").read_text()
        chunks = await chunking_service.chunk_content(content, files[0])

        assert len(chunks) > 0
        assert all(chunk.content.strip() for chunk in chunks)

    @pytest.mark.asyncio
    async def test_service_health_monitoring(self, factory_with_services):
        """Test service health monitoring."""
        services_manager = factory_with_services.get_services_manager()

        # Check health
        health_report = await services_manager.health_check()

        assert health_report.overall_status in ["healthy", "degraded"]
        assert ServiceType.CHUNKING in health_report.services
        assert ServiceType.FILTERING in health_report.services

# Run test
# uv run pytest tests/integration/test_services_integration.py -v
```

### Step 8.2: Run Complete Test Suite

```bash
# Run all tests to ensure no regressions
uv run pytest tests/ -v

# Run specific service tests
uv run pytest tests/unit/services/ -v
uv run pytest tests/integration/test_services_integration.py -v

# Check code coverage
uv run pytest --cov=codeweaver tests/

# Check code quality
uv run ruff check src/
uv run ruff format src/
```

---

## Phase 9: Server Integration (Day 14)

### Step 9.1: Update Main Server

**File:** `src/codeweaver/server.py` (UPDATE)

```python
# Add to existing CodeWeaverServer class

async def initialize(self) -> None:
    """Initialize server with services."""

    # Existing initialization...

    # NEW: Initialize services BEFORE plugins
    logger.info("Initializing services...")
    await self._factory.initialize_services()
    await self._factory.start_services()
    logger.info("Services initialized successfully")

    # Initialize plugins with service injection
    await self._initialize_plugins()

    # Setup FastMCP middleware (updated to work with services)
    await self._setup_middleware()

async def shutdown(self) -> None:
    """Shutdown server gracefully."""

    # Existing shutdown...

    # NEW: Shutdown services
    logger.info("Shutting down services...")
    await self._factory.shutdown_services()
    logger.info("Services shut down successfully")
```

### Step 9.2: Verify Everything Works

```bash
# Test server startup
uv run python src/codeweaver/main.py --help

# Test with a small codebase
mkdir -p /tmp/test_codebase
echo "def hello(): pass" > /tmp/test_codebase/test.py

# Start server and test indexing (in separate terminal)
# uv run codeweaver
# (Test with Claude Desktop or MCP client)
```

---

## Completion Validation

### Final Verification Checklist

- [ ] **Type System**: ServiceType enum, data structures, exceptions work
- [ ] **Service Protocols**: ChunkingService, FilteringService protocols defined
- [ ] **Configuration**: Service configuration classes created and tested
- [ ] **Service Registry**: Provider registration and instance management works
- [ ] **Service Providers**: FastMCP chunking and filtering providers implemented
- [ ] **Services Manager**: Lifecycle management and dependency injection works
- [ ] **Factory Integration**: CodeWeaverFactory extended with services support
- [ ] **Server Integration**: Main server initializes and uses services
- [ ] **Testing**: Unit and integration tests pass
- [ ] **Documentation**: Code is documented and follows conventions

### Performance Validation

```bash
# Run benchmarks to ensure no performance regression
uv run python tests/integration/test_benchmarks.py

# Test with large codebase
git clone https://github.com/python/cpython /tmp/large_codebase
# Test indexing performance
```

### Code Quality Validation

```bash
# Final code quality check
uv run ruff check src/ --fix
uv run ruff format src/

# Type checking
uv run mypy src/codeweaver --ignore-missing-imports

# Security check
uv run bandit -r src/
```

---

## Conclusion

This clean implementation guide provides a complete step-by-step approach to implementing CodeWeaver's services registry and abstraction layer. The implementation:

**✅ Eliminates Direct Dependencies**: Plugins no longer directly import middleware
**✅ Implements Factory Pattern**: All services created through factory system
**✅ Provides Dependency Injection**: Clean service injection into plugins
**✅ Enables Testing**: Comprehensive test coverage with mocking support
**✅ Supports Monitoring**: Built-in health checking and performance metrics
**✅ Maintains Performance**: Optimized implementation with minimal overhead

The result is a clean, extensible, and maintainable services architecture that provides a solid foundation for CodeWeaver's continued development.
