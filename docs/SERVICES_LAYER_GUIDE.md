# Services Layer Usage Guide

**Date:** January 27, 2025
**Author:** CodeWeaver Development Team
**Version:** 1.0

## Overview

The CodeWeaver services layer provides a clean abstraction between the FastMCP middleware and the factory-based plugin system. It's designed to handle cross-cutting concerns like caching, rate limiting, health monitoring, and request processing while maintaining loose coupling between components.

## Architecture Overview

### Core Components

```
src/codeweaver/services/
├── manager.py              # ServicesManager - coordinates all services
├── middleware_bridge.py    # Bridge between FastMCP and services
└── providers/             # Service provider implementations
    ├── base_provider.py   # Base service provider interface
    ├── chunking.py        # Content chunking service
    ├── file_filtering.py  # File filtering service
    └── middleware.py      # Middleware service integration
```

### Key Classes

- **`ServicesManager`**: Central coordinator for all services with health monitoring
- **`MiddlewareBridge`**: Bridge between FastMCP middleware and service layer
- **`BaseServiceProvider`**: Base class for all service implementations
- **Service Providers**: Specific implementations (chunking, filtering, etc.)

## Getting Started

### Basic Service Integration

Services are injected into plugin operations through the context parameter:

```python
from codeweaver.services.manager import ServicesManager
from codeweaver._types import ServiceHealth, ServiceStatus

class ExamplePlugin:
    async def process_content(self, content: str, context: dict) -> ProcessedContent:
        """Process content using services layer."""

        # Get service from context
        chunking_service = context.get("chunking_service")

        # Use service if available with fallback
        if chunking_service and await chunking_service.is_healthy():
            chunks = await chunking_service.chunk_content(content)
        else:
            # Fallback logic without service dependency
            chunks = self._simple_chunk_fallback(content)

        return ProcessedContent(chunks=chunks)

    def _simple_chunk_fallback(self, content: str) -> list[str]:
        """Simple chunking fallback without external dependencies."""
        chunk_size = 1500
        return [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
```

### Service Configuration

Services use hierarchical configuration through TOML files:

```toml
# config.toml
[services.chunking]
max_chunk_size = 1500
min_chunk_size = 50
overlap_size = 100
enabled = true

[services.filtering]
max_file_size = 1048576  # 1MB
ignore_patterns = [".git", "node_modules", "__pycache__"]
enabled = true

[services.validation]
strict_mode = false
enabled = true
```

### Service Manager Initialization

```python
from codeweaver.services.manager import ServicesManager
from codeweaver.config import CodeWeaverConfig

# Initialize services manager
config = CodeWeaverConfig.from_file("config.toml")
services_manager = ServicesManager(config.services)

# Start all services
await services_manager.start_all_services()

# Create context for plugins
context = await services_manager.create_service_context()
```

## Service Usage Patterns

### 1. Service-Aware Plugin Implementation

Always design plugins to work with or without services:

```python
class FileSystemSourceProvider:
    """Example of proper services layer integration."""

    async def discover_content(self, context: dict[str, Any]) -> AsyncIterator[ContentItem]:
        """Discover content using services layer."""

        # Get services from context
        filtering_service = context.get("filtering_service")
        chunking_service = context.get("chunking_service")
        validation_service = context.get("validation_service")

        for file_path in self._scan_files():
            # Use filtering service if available
            if filtering_service:
                if not await filtering_service.should_process_file(file_path):
                    continue
            else:
                # Fallback logic without service dependency
                if self._should_skip_file(file_path):
                    continue

            # Read and validate content
            content = await self._read_file(file_path)
            if validation_service:
                content = await validation_service.validate_content(content)

            # Chunk content using service
            if chunking_service:
                chunks = await chunking_service.chunk_content(content, str(file_path))
            else:
                chunks = self._simple_chunk_fallback(content, file_path)

            for chunk in chunks:
                yield ContentItem.from_chunk(chunk, file_path)
```

### 2. Service Health Monitoring

Monitor service health and handle degraded states:

```python
async def process_with_health_check(self, data: Any, context: dict) -> Any:
    """Process data with service health monitoring."""

    services_manager = context.get("services_manager")
    if services_manager:
        # Check service health
        health_status = await services_manager.get_service_health("chunking")

        if health_status.status == ServiceStatus.HEALTHY:
            chunking_service = context.get("chunking_service")
            return await chunking_service.process(data)
        elif health_status.status == ServiceStatus.DEGRADED:
            logger.warning(f"Service degraded: {health_status.message}")
            # Use limited functionality
            return await self._limited_processing(data)
        else:
            logger.error(f"Service unhealthy: {health_status.message}")
            # Use fallback
            return await self._fallback_processing(data)

    # No services manager available
    return await self._fallback_processing(data)
```

### 3. Service Capabilities Checking

Check service capabilities before use:

```python
async def smart_processing(self, content: str, language: str, context: dict) -> ProcessedContent:
    """Use service capabilities intelligently."""

    chunking_service = context.get("chunking_service")

    if chunking_service:
        # Check if service supports the specific language
        if hasattr(chunking_service, 'supports_language') and \
           chunking_service.supports_language(language):
            return await chunking_service.chunk_code(content, language)
        elif hasattr(chunking_service, 'chunk_content'):
            return await chunking_service.chunk_content(content)

    # Fallback to generic processing
    return self._generic_processing(content)
```

## Advanced Usage

### Custom Service Implementation

Create custom services by extending the base provider:

```python
from codeweaver.services.providers.base_provider import BaseServiceProvider
from codeweaver._types import ServiceHealth, ServiceStatus

class CustomCacheService(BaseServiceProvider):
    """Custom caching service implementation."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._cache = {}
        self._max_size = config.get("max_size", 1000)

    async def start(self) -> None:
        """Start the cache service."""
        self._cache.clear()
        logger.info("Cache service started")

    async def stop(self) -> None:
        """Stop the cache service."""
        self._cache.clear()
        logger.info("Cache service stopped")

    async def health_check(self) -> ServiceHealth:
        """Check cache service health."""
        try:
            cache_size = len(self._cache)
            if cache_size > self._max_size * 0.9:
                return ServiceHealth(
                    status=ServiceStatus.DEGRADED,
                    message=f"Cache nearly full: {cache_size}/{self._max_size}",
                    last_check=datetime.now(UTC)
                )
            else:
                return ServiceHealth(
                    status=ServiceStatus.HEALTHY,
                    message=f"Cache healthy: {cache_size}/{self._max_size}",
                    last_check=datetime.now(UTC)
                )
        except Exception as e:
            return ServiceHealth(
                status=ServiceStatus.UNHEALTHY,
                message=f"Cache error: {e}",
                last_check=datetime.now(UTC)
            )

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        return self._cache.get(key)

    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set value in cache with TTL."""
        if len(self._cache) >= self._max_size:
            # Simple LRU eviction
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = {
            "value": value,
            "expires": time.time() + ttl
        }
```

### Service Registration

Register custom services with the services manager:

```python
from codeweaver.services.manager import ServicesManager

# Register custom service
services_manager.register_service("cache", CustomCacheService)

# Configure in TOML
[services.cache]
max_size = 2000
enabled = true
```

## Best Practices

### 1. Always Provide Fallbacks

```python
# ✅ GOOD: Graceful degradation
async def process_content(self, content: str, context: dict) -> ProcessedContent:
    service = context.get("processing_service")
    if service and await service.is_healthy():
        return await service.process(content)
    else:
        return self._fallback_processing(content)

# ❌ BAD: Hard dependency on service
async def process_content(self, content: str, context: dict) -> ProcessedContent:
    service = context["processing_service"]  # Will fail if not available
    return await service.process(content)
```

### 2. Use Service Capabilities

```python
# ✅ GOOD: Check service capabilities before use
chunking_service = context.get("chunking_service")
if chunking_service and hasattr(chunking_service, 'supports_language'):
    if chunking_service.supports_language("python"):
        return await chunking_service.chunk_code(content, "python")

# ❌ BAD: Assume service capabilities
chunking_service = context.get("chunking_service")
return await chunking_service.chunk_code(content, "python")  # May not support language-specific chunking
```

### 3. Handle Service Errors Gracefully

```python
# ✅ GOOD: Proper error handling
try:
    result = await service.process(data)
except ServiceUnavailableError:
    logger.warning("Service unavailable, using fallback")
    result = self._fallback_process(data)
except ServiceError as e:
    logger.error(f"Service error: {e}")
    raise ProcessingError("Failed to process data") from e

# ❌ BAD: Let service errors propagate
result = await service.process(data)  # Service errors will break the plugin
```

### 4. Implement Service-Aware Testing

```python
# ✅ GOOD: Test with and without services
async def test_with_service(self):
    context = {"chunking_service": MockChunkingService()}
    result = await self.provider.process(content, context)
    assert result.chunks_count > 0

async def test_without_service(self):
    context = {}  # No services available
    result = await self.provider.process(content, context)
    assert result.chunks_count > 0  # Should still work

# ❌ BAD: Only test with services
async def test_processing(self):
    context = {"chunking_service": MockChunkingService()}
    result = await self.provider.process(content, context)
    assert result.chunks_count > 0
    # Missing test for fallback behavior
```

## Common Patterns

### Rate Limiting Integration

```python
async def embed_documents(self, texts: list[str], context: dict) -> list[list[float]]:
    """Generate embeddings with rate limiting."""

    # Check rate limiting service
    rate_limiter = context.get("rate_limiting_service")
    if rate_limiter:
        await rate_limiter.acquire("embedding_provider", len(texts))

    try:
        result = await self._generate_embeddings(texts)
    except Exception:
        logger.exception("Error generating embeddings")
        raise
    else:
        return result
```

### Caching Integration

```python
async def embed_documents(self, texts: list[str], context: dict) -> list[list[float]]:
    """Generate embeddings with caching."""

    # Check cache service
    cache_service = context.get("caching_service")
    if cache_service:
        cache_key = self._generate_cache_key(texts)
        cached_result = await cache_service.get(cache_key)
        if cached_result:
            return cached_result

    # Generate embeddings
    result = await self._generate_embeddings(texts)

    # Cache result
    if cache_service:
        await cache_service.set(cache_key, result, ttl=3600)

    return result
```

### Health Monitoring Integration

```python
class QdrantBackend:
    async def health_check(self) -> ServiceHealth:
        """Check backend health."""
        try:
            await self.client.get_collections()
            return ServiceHealth(
                status=ServiceStatus.HEALTHY,
                message="Qdrant connection healthy",
                last_check=datetime.now(UTC)
            )
        except Exception as e:
            return ServiceHealth(
                status=ServiceStatus.UNHEALTHY,
                message=f"Qdrant connection failed: {e}",
                last_check=datetime.now(UTC)
            )

# Register with services manager
services_manager.register_health_check("qdrant_backend", backend.health_check)
```

## Troubleshooting

### Common Issues

1. **Service Not Available**
   ```python
   # Problem: Service not found in context
   service = context.get("missing_service")  # Returns None

   # Solution: Always check and provide fallback
   service = context.get("missing_service")
   if service:
       result = await service.process(data)
   else:
       result = self._fallback_process(data)
   ```

2. **Service Health Issues**
   ```python
   # Problem: Service is unhealthy but still being used
   service = context.get("service")
   result = await service.process(data)  # May fail

   # Solution: Check health before use
   service = context.get("service")
   if service and await service.is_healthy():
       result = await service.process(data)
   else:
       result = self._fallback_process(data)
   ```

3. **Configuration Issues**
   ```toml
   # Problem: Service not enabled
   [services.chunking]
   enabled = false  # Service won't be available

   # Solution: Enable required services
   [services.chunking]
   enabled = true
   max_chunk_size = 1500
   ```

### Debugging Tips

1. **Enable Service Logging**
   ```python
   import logging
   logging.getLogger("codeweaver.services").setLevel(logging.DEBUG)
   ```

2. **Check Service Status**
   ```python
   services_manager = context.get("services_manager")
   if services_manager:
       status = await services_manager.get_all_service_health()
       for service_name, health in status.items():
           print(f"{service_name}: {health.status.value} - {health.message}")
   ```

3. **Validate Service Configuration**
   ```python
   from codeweaver.config import CodeWeaverConfig

   config = CodeWeaverConfig.from_file("config.toml")
   print(f"Services config: {config.services}")
   ```

## Migration from Direct Dependencies

If you have existing code that directly imports middleware or other dependencies, follow this pattern:

```python
# Before: Direct middleware dependency
from codeweaver.middleware.chunking import ChunkingMiddleware

class OldPlugin:
    async def process(self, content: str) -> ProcessedContent:
        chunker = ChunkingMiddleware()  # Direct dependency
        chunks = await chunker.chunk_content(content)
        return ProcessedContent(chunks=chunks)

# After: Services layer integration
class NewPlugin:
    async def process(self, content: str, context: dict) -> ProcessedContent:
        # Use service from context
        chunking_service = context.get("chunking_service")
        if chunking_service:
            chunks = await chunking_service.chunk_content(content)
        else:
            # Fallback without external dependency
            chunks = self._simple_chunk_fallback(content)
        return ProcessedContent(chunks=chunks)

    def _simple_chunk_fallback(self, content: str) -> list[str]:
        """Simple chunking without middleware dependency."""
        chunk_size = 1500
        return [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
```

## Conclusion

The services layer provides a powerful abstraction for managing cross-cutting concerns while maintaining loose coupling between components. By following the patterns and best practices outlined in this guide, you can create robust, testable, and maintainable plugins that work seamlessly with the CodeWeaver architecture.

For more information, see:
- [Development Patterns Guide](DEVELOPMENT_PATTERNS.md)
- [Migration Guide](MIGRATION_GUIDE.md)
- [Factory System Documentation](FACTORY_SYSTEM.md)
