# Services Configuration

CodeWeaver's service layer provides clean abstraction between FastMCP middleware and the factory pattern system. This enables dependency injection, health monitoring, and extensible service providers for enhanced functionality.

## Service Architecture Overview

The service layer consists of:

- **Core Services** (required): Chunking and Filtering
- **Optional Services** (enhanced functionality): Validation, Cache, Monitoring, Metrics
- **Service Manager**: Coordinates all services with health monitoring
- **Middleware Bridge**: Integrates services with FastMCP middleware

## Core Services

### Chunking Service

**Purpose:** Intelligent code segmentation using ast-grep with fallback parsing

#### Configuration

=== "Environment Variables"

    ```bash
    # Basic configuration
    export CW_SERVICES__CHUNKING__PROVIDER="fastmcp_chunking"
    export CW_SERVICES__CHUNKING__MAX_CHUNK_SIZE=1500
    export CW_SERVICES__CHUNKING__MIN_CHUNK_SIZE=50
    
    # Advanced settings
    export CW_SERVICES__CHUNKING__AST_GREP_ENABLED=true
    export CW_SERVICES__CHUNKING__PERFORMANCE_MODE="balanced"
    ```

=== "TOML Configuration"

    ```toml
    [services.chunking]
    provider = "fastmcp_chunking"
    max_chunk_size = 1500
    min_chunk_size = 50
    ast_grep_enabled = true
    performance_mode = "balanced"  # fast, balanced, quality
    fallback_enabled = true
    
    # Advanced chunking settings
    overlap_size = 100
    respect_boundaries = true
    language_detection = true
    ```

#### Performance Modes

| Mode | Speed | Quality | Use Case |
|------|-------|---------|----------|
| `fast` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Large codebases, CI/CD |
| `balanced` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | General use (default) |
| `quality` | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Precision-critical tasks |

#### Language-Specific Settings

```toml
[services.chunking.languages]
# Python-specific chunking
python.function_priority = true
python.class_grouping = true

# JavaScript-specific chunking  
javascript.module_awareness = true
javascript.async_handling = true

# Generic settings
generic.line_based_fallback = true
generic.comment_preservation = true
```

---

### Filtering Service

**Purpose:** File discovery and filtering with gitignore support

#### Configuration

```toml
[services.filtering]
provider = "fastmcp_filtering"
use_gitignore = true
max_file_size = 1048576  # 1MB
parallel_scanning = true
follow_symlinks = false

# File type filtering
include_extensions = [".py", ".js", ".ts", ".java", ".cpp"]
exclude_patterns = ["**/node_modules/**", "**/build/**"]

# Performance settings
batch_size = 100
max_depth = 10
scan_timeout = 30
```

#### Built-in Filter Patterns

```toml
[services.filtering.patterns]
# Common exclusions (automatically applied)
build_dirs = ["build/", "dist/", "target/", ".next/"]
cache_dirs = [".cache/", "__pycache__/", "node_modules/"]
version_control = [".git/", ".svn/", ".hg/"]

# Language-specific exclusions
python = ["*.pyc", "*.pyo", "*.pyd", "__pycache__/"]
javascript = ["node_modules/", "*.min.js", "bundle.js"]
java = ["target/", "*.class", "*.jar"]
```

## Optional Services

### Validation Service

**Purpose:** Content validation with configurable rules

#### Configuration

```toml
[services.validation]
provider = "fastmcp_validation"
enabled = true
strict_mode = false

# Validation rules
max_content_size = 10485760  # 10MB
encoding_validation = true
syntax_checking = true
security_scanning = false

# Custom validation rules
[services.validation.rules]
# File size limits by type
python_max_size = 1048576
javascript_max_size = 524288

# Content rules
no_secrets = true
no_binary_content = true
valid_encoding = ["utf-8", "ascii"]
```

#### Validation Rules Engine

```toml
[services.validation.custom_rules]
# Define custom validation patterns
no_hardcoded_passwords = '''
rule = "regex"
pattern = "(password|passwd|pwd)\\s*=\\s*['\"][^'\"]{8,}['\"]"
severity = "high"
message = "Potential hardcoded password detected"
'''

file_size_limits = '''
rule = "size"
max_size = 5242880  # 5MB
applies_to = ["*.md", "*.txt"]
severity = "medium"
'''
```

---

### Cache Service

**Purpose:** Performance optimization through intelligent caching

#### Configuration

```toml
[services.cache]
provider = "fastmcp_cache"
enabled = true
backend = "memory"  # memory, redis, disk

# Memory cache settings
max_memory_mb = 512
eviction_policy = "lru"  # lru, lfu, fifo
ttl_seconds = 3600

# Disk cache settings (when backend = "disk")
cache_dir = "~/.cache/codeweaver"
max_disk_mb = 2048
compression = true

# Redis cache settings (when backend = "redis")
redis_url = "redis://localhost:6379"
redis_db = 0
```

#### Cache Strategies

```toml
[services.cache.strategies]
# Embedding cache
embeddings.enabled = true
embeddings.ttl = 86400  # 24 hours
embeddings.key_pattern = "emb:{hash}"

# Chunking cache
chunks.enabled = true
chunks.ttl = 3600  # 1 hour
chunks.invalidate_on_change = true

# Search results cache
search.enabled = false  # Disable for dynamic results
search.ttl = 300  # 5 minutes
```

---

### Monitoring Service

**Purpose:** Health monitoring and auto-recovery

#### Configuration

```toml
[services.monitoring]
provider = "fastmcp_monitoring"
enabled = true
health_check_interval = 60  # seconds
auto_recovery = true
max_recovery_attempts = 3

# Health check endpoints
[services.monitoring.endpoints]
chunking_service = "http://localhost:8080/health"
backend_service = "http://localhost:6333/health"

# Alert configuration
[services.monitoring.alerts]
email_enabled = false
webhook_url = "https://your-webhook.com/alerts"
severity_threshold = "warning"  # info, warning, error, critical
```

#### Auto-Recovery Patterns

```toml
[services.monitoring.recovery]
# Service restart patterns
restart_on_failure = true
restart_delay = 5  # seconds
max_restart_attempts = 3

# Circuit breaker pattern
circuit_breaker_enabled = true
failure_threshold = 5
recovery_timeout = 60

# Degraded mode handling
enable_degraded_mode = true
degraded_timeout = 300
```

---

### Metrics Service

**Purpose:** Performance metrics collection and analysis

#### Configuration

```toml
[services.metrics]
provider = "fastmcp_metrics"
enabled = true
collection_interval = 60  # seconds
export_format = "prometheus"  # prometheus, json, csv

# Metrics storage
storage_backend = "memory"  # memory, file, database
retention_days = 30

# Custom metrics
[services.metrics.custom]
request_duration = true
memory_usage = true
cache_hit_rate = true
error_rate = true
```

## Service Manager Configuration

### Global Service Settings

```toml
[services]
# Global service configuration
enable_health_monitoring = true
health_check_interval = 60
service_timeout = 30

# Dependency injection
auto_wire_dependencies = true
lazy_initialization = false

# Error handling
fail_fast = false
retry_failed_services = true
max_retry_attempts = 3
```

### Service Discovery

```toml
[services.discovery]
# Service provider discovery
auto_discovery = true
scan_paths = ["src/", "plugins/"]
plugin_patterns = ["*_service.py", "*_provider.py"]

# Manual service registration
[services.providers]
chunking = "codeweaver.services.providers.chunking:FastMCPChunkingProvider"
filtering = "codeweaver.services.providers.filtering:FastMCPFilteringProvider"
```

## Service Provider Development

### Creating Custom Services

#### 1. Implement Service Protocol

```python
from codeweaver.cw_types import ChunkingService
from codeweaver.services.providers.base_provider import BaseServiceProvider

class CustomChunkingProvider(BaseServiceProvider, ChunkingService):
    async def _initialize_provider(self) -> None:
        # Initialize your service
        self.config = self.get_config()
        await self._setup_resources()
    
    async def chunk_content(
        self, 
        content: str, 
        file_path: str | None = None
    ) -> list[str]:
        # Implement chunking logic
        chunks = self._custom_chunking_algorithm(content, file_path)
        return chunks
    
    async def _health_check_implementation(self) -> bool:
        # Implement health check
        return True
```

#### 2. Register with Factory

```python
from codeweaver.factories import codeweaver_factory
from codeweaver.cw_types import ServiceType

# Register service provider
codeweaver_factory.register_service_provider(
    ServiceType.CHUNKING,
    "custom_chunking",
    CustomChunkingProvider
)
```

#### 3. Configure Service

```toml
[services.chunking]
provider = "custom_chunking"
# Your custom configuration options
custom_option = "value"
```

### Service Plugin Development

#### Plugin Interface

```python
from codeweaver.cw_types import ServiceProvider

class ServicePlugin:
    def get_service_class(self) -> type[ServiceProvider]:
        return CustomValidationProvider
    
    @property
    def service_type(self) -> ServiceType:
        return ServiceType.VALIDATION
    
    def get_config_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "custom_option": {"type": "string"}
            }
        }
```

#### Plugin Registration

```python
# Entry point in setup.py/pyproject.toml
[project.entry-points."codeweaver.services"]
my_service = "my_package.plugin:ServicePlugin"
```

## Service Integration Patterns

### Dependency Injection

Services can depend on other services:

```python
class ValidationProvider(BaseServiceProvider, ValidationService):
    def __init__(self, config: dict, services_manager):
        super().__init__(config)
        self.chunking_service = services_manager.get_chunking_service()
        self.filtering_service = services_manager.get_filtering_service()
    
    async def validate_content(self, content: str) -> bool:
        # Use other services
        chunks = await self.chunking_service.chunk_content(content)
        # Validation logic using chunks
        return True
```

### Middleware Integration

Services integrate automatically with FastMCP middleware:

```python
from codeweaver.services.middleware_bridge import ServiceBridge

# Services are injected into tool contexts
async def search_tool(context: MCP_Context):
    chunking_service = context.get_service("chunking")
    filtering_service = context.get_service("filtering")
    
    # Use services in tool implementation
    chunks = await chunking_service.chunk_content(content)
    files = await filtering_service.discover_files(path)
```

## Performance Optimization

### Service-Level Caching

```toml
[services.chunking]
enable_caching = true
cache_size_mb = 256
cache_ttl = 3600

[services.filtering]
enable_caching = true
cache_directory_scans = true
cache_ttl = 1800
```

### Parallel Processing

```toml
[services]
# Global parallel processing settings
max_workers = 8
enable_async_processing = true
batch_processing = true

[services.chunking]
parallel_chunks = true
chunk_workers = 4

[services.filtering]
parallel_scanning = true
scan_workers = 6
```

### Resource Management

```toml
[services.resources]
# Memory limits per service
max_memory_per_service = 512  # MB
memory_monitoring = true

# CPU limits
max_cpu_per_service = 50  # percentage
cpu_monitoring = true

# Timeout settings
service_timeout = 30
operation_timeout = 10
```

## Monitoring and Observability

### Health Monitoring

```toml
[services.health]
# Global health monitoring
enable_monitoring = true
check_interval = 60
auto_recovery = true

# Service-specific health checks
[services.health.checks]
chunking.endpoint = "/health"
chunking.timeout = 5
chunking.expected_status = 200

filtering.custom_check = "file_system_access"
filtering.timeout = 3
```

### Metrics Collection

```toml
[services.metrics]
# Performance metrics
collect_performance_metrics = true
metrics_interval = 60

# Custom metrics per service
[services.metrics.chunking]
track_chunk_count = true
track_processing_time = true
track_cache_hits = true

[services.metrics.filtering]
track_files_scanned = true
track_filter_performance = true
track_exclusion_rates = true
```

### Logging Configuration

```toml
[services.logging]
# Service-specific logging
level = "INFO"
format = "json"
include_context = true

# Per-service log levels
[services.logging.levels]
chunking = "DEBUG"
filtering = "INFO"
validation = "WARNING"
```

## Troubleshooting Services

### Service Initialization Issues

```python
# Check service health
from codeweaver.services.manager import ServicesManager

services_manager = ServicesManager(config)
await services_manager.initialize()

# Get health report
health_report = await services_manager.get_health_report()
print(f"Overall status: {health_report.overall_status}")

for service_name, health in health_report.services.items():
    print(f"{service_name}: {health.status} - {health.message}")
```

### Common Service Errors

#### Service Not Found

```plaintext
Error: Service provider 'custom_chunking' not found
```

**Solutions:**
1. Check provider registration
2. Verify plugin installation
3. Ensure correct service name

#### Service Initialization Failed

```plaintext
Error: Failed to initialize chunking service
```

**Solutions:**
1. Check service dependencies
2. Verify configuration
3. Review service logs

#### Health Check Failures

```plaintext
Warning: Chunking service health check failed
```

**Solutions:**
1. Check service resources
2. Review error logs
3. Restart service if needed

### Debug Mode

Enable debug mode for detailed service information:

```toml
[services.debug]
enabled = true
log_level = "DEBUG"
trace_requests = true
profile_performance = true
```

## Next Steps

- **Advanced configuration**: [Advanced Configuration](./advanced.md)
- **Custom provider development**: [Extension Development](../extension-dev/)
- **Performance optimization**: [Performance Guide](../user-guide/performance.md)
- **Monitoring setup**: [Monitoring Guide](../user-guide/monitoring.md)