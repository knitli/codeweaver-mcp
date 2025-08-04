<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Extension Development Guidelines

This guide provides comprehensive guidelines for creating extensions to CodeWeaver. Learn how to build providers, backends, sources, and services that integrate seamlessly with the plugin architecture.

## 🏗️ Architecture Overview

CodeWeaver's extensible architecture allows you to add new functionality through four main extension points:

```
┌─────────────────────────────────────────────────────────────┐
│                    CodeWeaver Core                          │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Providers     │    Backends     │       Sources           │
│                 │                 │                         │
│ • Embeddings    │ • Vector DBs    │ • Content Discovery     │
│ • Reranking     │ • Storage       │ • Processing            │
│ • Custom APIs   │ • Search        │ • Filtering             │
└─────────────────┴─────────────────┴─────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │     Services      │
                    │                   │
                    │ • Caching         │
                    │ • Rate Limiting   │
                    │ • Health Checks   │
                    │ • Metrics         │
                    └───────────────────┘
```

## 🎯 Extension Types

### Providers
Integrate AI services for embeddings and reranking:
- **Embedding Providers**: OpenAI, Anthropic, Cohere, etc.
- **Reranking Providers**: Cohere Rerank, Cross-encoder models
- **Combined Providers**: Services that offer both capabilities

### Backends
Add vector database support:
- **Vector Databases**: Pinecone, Weaviate, Milvus, LanceDB
- **Traditional Databases**: PostgreSQL with pgvector, SQLite-VSS
- **Cloud Services**: AWS OpenSearch, Azure Cognitive Search

### Sources
Enable new content discovery methods:
- **Version Control**: GitHub, GitLab, Bitbucket APIs
- **Communication**: Slack, Discord, Microsoft Teams
- **Documentation**: Confluence, Notion, Wiki systems
- **Databases**: PostgreSQL, MongoDB, Elasticsearch

### Services
Add cross-cutting functionality:
- **Caching**: Redis, Memcached, local cache
- **Rate Limiting**: Token bucket, sliding window
- **Monitoring**: Metrics collection, health checks
- **Security**: Authentication, authorization, encryption

## 🔧 Provider Development

Providers handle AI service integration for embeddings and reranking.

### Basic Provider Structure

```python
from codeweaver.providers.base import BaseProvider
from codeweaver.cw_types import (
    EmbeddingProvider,
    ProviderCapability,
    EmbeddingProviderInfo,
    ProviderError
)

class MyProvider(BaseProvider):
    """Custom provider for MyService API."""

    def __init__(self, config: MyProviderConfig | dict[str, Any]):
        super().__init__(config)
        self._validate_config()
        self._initialize_client()

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "my_provider"

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities."""
        return self._capabilities

    @classmethod
    def check_availability(cls, capability: ProviderCapability) -> tuple[bool, str | None]:
        """Check if provider is available for the given capability."""
        if not MY_SERVICE_AVAILABLE:
            return False, "my-service package not installed (install with: uv add my-service)"

        if capability == ProviderCapability.EMBEDDING:
            return True, None

        return False, f"Capability {capability.value} not supported"

    @classmethod
    def get_static_provider_info(cls) -> EmbeddingProviderInfo:
        """Get static information about this provider."""
        return EmbeddingProviderInfo(
            name="my_provider",
            capabilities=cls._get_static_capabilities(),
            supported_models=["model-v1", "model-v2"],
            default_model="model-v1",
            native_dimensions={"model-v1": 768, "model-v2": 1536},
            description="Custom provider for MyService embeddings"
        )

    async def embed_documents(
        self,
        texts: list[str],
        context: dict | None = None
    ) -> list[list[float]]:
        """Generate embeddings for documents."""
        if context is None:
            context = {}

        # Input validation
        if not texts:
            raise ValueError("texts cannot be empty")

        # Rate limiting integration
        rate_limiter = context.get("rate_limiting_service")
        if rate_limiter:
            await rate_limiter.acquire("my_provider", len(texts))

        # Caching integration
        cache_service = context.get("caching_service")
        cache_key = None
        if cache_service:
            cache_key = self._generate_cache_key(texts)
            cached_result = await cache_service.get(cache_key)
            if cached_result:
                return cached_result

        try:
            # Call your API
            embeddings = await self._call_api(texts)

            # Cache results
            if cache_service and cache_key:
                await cache_service.set(cache_key, embeddings, ttl=3600)

            # Record metrics
            metrics_service = context.get("metrics_service")
            if metrics_service:
                await metrics_service.record_request(
                    provider="my_provider",
                    operation="embed_documents",
                    count=len(texts),
                    success=True
                )

            return embeddings

        except Exception as e:
            # Record failure metrics
            metrics_service = context.get("metrics_service")
            if metrics_service:
                await metrics_service.record_request(
                    provider="my_provider",
                    operation="embed_documents",
                    count=len(texts),
                    success=False,
                    error=str(e)
                )

            # Convert to appropriate exception type
            if isinstance(e, ConnectionError):
                raise ServiceUnavailableError(f"MyService API unavailable: {e}") from e
            else:
                raise ProviderError(f"MyService provider error: {e}") from e
```

### Provider Configuration

```python
from pydantic import BaseModel, Field, field_validator
from typing import Annotated

class MyProviderConfig(BaseModel):
    """Configuration for MyProvider."""

    api_key: Annotated[str, Field(description="MyService API key")]
    model: Annotated[str, Field(default="model-v1", description="Model to use")]
    max_batch_size: Annotated[int, Field(default=128, description="Maximum batch size")]
    timeout: Annotated[int, Field(default=30, description="Request timeout in seconds")]

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("API key cannot be empty")
        return v.strip()

    @field_validator("max_batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_batch_size must be positive")
        if v > 1000:
            raise ValueError("max_batch_size cannot exceed 1000")
        return v
```

### Provider Testing

```python
import pytest
from unittest.mock import AsyncMock
from codeweaver.providers.my_provider import MyProvider, MyProviderConfig

class TestMyProvider:
    @pytest.fixture
    def config(self):
        return MyProviderConfig(
            api_key="test-key",
            model="model-v1",
            max_batch_size=10
        )

    @pytest.fixture
    def provider(self, config):
        return MyProvider(config)

    @pytest.fixture
    def mock_context(self):
        return {
            "rate_limiting_service": AsyncMock(),
            "caching_service": AsyncMock(),
            "metrics_service": AsyncMock(),
        }

    async def test_embed_documents_success(self, provider, mock_context):
        """Test successful embedding generation."""
        # Mock the API call
        provider._call_api = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        result = await provider.embed_documents(["test text"], mock_context)

        assert len(result) == 1
        assert len(result[0]) == 3

        # Verify service integrations
        mock_context["rate_limiting_service"].acquire.assert_called_once()
        mock_context["metrics_service"].record_request.assert_called_once()

    async def test_embed_documents_validation(self, provider):
        """Test input validation."""
        with pytest.raises(ValueError, match="texts cannot be empty"):
            await provider.embed_documents([])

    def test_check_availability(self):
        """Test availability checking."""
        available, error = MyProvider.check_availability(ProviderCapability.EMBEDDING)
        assert available is True
        assert error is None
```

## 🗄️ Backend Development

Backends handle vector database integration for storing and searching embeddings.

### Basic Backend Structure

```python
from codeweaver.backends.base import BaseBackend
from codeweaver.cw_types import (
    VectorBackend,
    VectorPoint,
    SearchResult,
    BackendError
)

class MyBackend(BaseBackend):
    """Custom backend for MyVectorDB."""

    def __init__(self, config: MyBackendConfig | dict[str, Any]):
        super().__init__(config)
        self._validate_config()
        self._client = None

    @property
    def backend_name(self) -> str:
        """Get the backend name."""
        return "my_backend"

    async def connect(self) -> None:
        """Connect to the vector database."""
        try:
            self._client = MyVectorDBClient(
                url=self.config["url"],
                api_key=self.config.get("api_key"),
                timeout=self.config.get("timeout", 30)
            )
            await self._client.connect()

        except Exception as e:
            raise BackendError(f"Failed to connect to MyVectorDB: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from the vector database."""
        if self._client:
            await self._client.close()
            self._client = None

    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        distance_metric: str = "cosine"
    ) -> None:
        """Create a new collection."""
        if not self._client:
            await self.connect()

        try:
            await self._client.create_collection(
                name=collection_name,
                dimension=dimension,
                metric=distance_metric
            )
        except Exception as e:
            raise BackendError(f"Failed to create collection: {e}") from e

    async def store_vectors(self, vectors: list[VectorPoint]) -> None:
        """Store vectors in the database."""
        if not self._client:
            await self.connect()

        if not vectors:
            return

        try:
            # Convert to backend format
            points = [
                {
                    "id": point.id,
                    "vector": point.vector,
                    "metadata": point.metadata or {}
                }
                for point in vectors
            ]

            await self._client.upsert_points(
                collection_name=self.collection_name,
                points=points
            )

        except Exception as e:
            raise BackendError(f"Failed to store vectors: {e}") from e

    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: dict | None = None
    ) -> list[SearchResult]:
        """Search for similar vectors."""
        if not self._client:
            await self.connect()

        try:
            results = await self._client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                filters=filters or {}
            )

            return [
                SearchResult(
                    id=result["id"],
                    score=result["score"],
                    metadata=result.get("metadata", {})
                )
                for result in results
            ]

        except Exception as e:
            raise BackendError(f"Search failed: {e}") from e

    async def health_check(self) -> dict[str, Any]:
        """Check backend health."""
        try:
            if not self._client:
                return {"status": "disconnected", "healthy": False}

            # Perform a simple operation to check health
            info = await self._client.get_info()

            return {
                "status": "connected",
                "healthy": True,
                "version": info.get("version"),
                "collections": info.get("collections", 0)
            }

        except Exception as e:
            return {
                "status": "error",
                "healthy": False,
                "error": str(e)
            }
```

### Backend Configuration

```python
class MyBackendConfig(BaseModel):
    """Configuration for MyBackend."""

    url: Annotated[str, Field(description="MyVectorDB server URL")]
    api_key: Annotated[str | None, Field(default=None, description="API key if required")]
    collection_name: Annotated[str, Field(default="codeweaver", description="Collection name")]
    timeout: Annotated[int, Field(default=30, description="Connection timeout")]
    max_connections: Annotated[int, Field(default=10, description="Max connection pool size")]

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v
```

## 📁 Source Development

Sources handle content discovery and processing from various data repositories.

### Basic Source Structure

```python
from codeweaver.sources.base import BaseSource
from codeweaver.cw_types import (
    DataSource,
    ContentItem,
    SourceError
)

class MySource(BaseSource):
    """Custom source for MyDataService."""

    def __init__(self, config: MySourceConfig | dict[str, Any]):
        super().__init__(config)
        self._validate_config()
        self._client = None

    @property
    def source_name(self) -> str:
        """Get the source name."""
        return "my_source"

    async def connect(self) -> None:
        """Connect to the data source."""
        try:
            self._client = MyDataServiceClient(
                api_key=self.config["api_key"],
                base_url=self.config.get("base_url")
            )
            await self._client.authenticate()

        except Exception as e:
            raise SourceError(f"Failed to connect to MyDataService: {e}") from e

    async def discover_content(
        self,
        path: str | None = None,
        filters: dict | None = None
    ) -> list[ContentItem]:
        """Discover content from the source."""
        if not self._client:
            await self.connect()

        try:
            # Apply filters
            query_filters = self._build_filters(filters or {})

            # Fetch content
            items = await self._client.list_items(
                path=path,
                filters=query_filters
            )

            # Convert to ContentItem format
            content_items = []
            for item in items:
                content_item = ContentItem(
                    id=item["id"],
                    path=item["path"],
                    content=item.get("content", ""),
                    content_type=self._detect_content_type(item["path"]),
                    size=item.get("size", 0),
                    last_modified=item.get("modified_at"),
                    metadata={
                        "source": "my_source",
                        "author": item.get("author"),
                        "tags": item.get("tags", []),
                        **item.get("custom_metadata", {})
                    }
                )
                content_items.append(content_item)

            return content_items

        except Exception as e:
            raise SourceError(f"Failed to discover content: {e}") from e

    async def get_content(self, item_id: str) -> ContentItem:
        """Get specific content by ID."""
        if not self._client:
            await self.connect()

        try:
            item = await self._client.get_item(item_id)

            return ContentItem(
                id=item["id"],
                path=item["path"],
                content=item["content"],
                content_type=self._detect_content_type(item["path"]),
                size=len(item["content"]),
                last_modified=item.get("modified_at"),
                metadata={
                    "source": "my_source",
                    "author": item.get("author"),
                    "version": item.get("version")
                }
            )

        except Exception as e:
            raise SourceError(f"Failed to get content {item_id}: {e}") from e

    def _build_filters(self, filters: dict) -> dict:
        """Convert generic filters to source-specific format."""
        source_filters = {}

        if "file_types" in filters:
            source_filters["extensions"] = filters["file_types"]

        if "modified_after" in filters:
            source_filters["updated_since"] = filters["modified_after"]

        if "author" in filters:
            source_filters["created_by"] = filters["author"]

        return source_filters

    def _detect_content_type(self, path: str) -> str:
        """Detect content type from file path."""
        extension = path.split(".")[-1].lower()

        type_mapping = {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "md": "markdown",
            "txt": "text",
            "json": "json",
            "yaml": "yaml",
            "yml": "yaml"
        }

        return type_mapping.get(extension, "unknown")
```

## ⚙️ Service Development

Services provide cross-cutting functionality like caching, rate limiting, and monitoring.

### Basic Service Structure

```python
from codeweaver.services.providers.base_provider import BaseServiceProvider
from codeweaver.cw_types import CachingService, ServiceHealth, ServiceStatus

class MyCachingService(BaseServiceProvider, CachingService):
    """Custom caching service implementation."""

    def __init__(self, config: MyCachingConfig):
        super().__init__(config)
        self._cache_client = None

    async def _initialize_provider(self) -> None:
        """Initialize the caching service."""
        try:
            self._cache_client = MyCacheClient(
                host=self.config.get("host", "localhost"),
                port=self.config.get("port", 6379),
                password=self.config.get("password")
            )
            await self._cache_client.connect()

        except Exception as e:
            raise ServiceError(f"Failed to initialize cache: {e}") from e

    async def _cleanup_provider(self) -> None:
        """Cleanup resources."""
        if self._cache_client:
            await self._cache_client.close()
            self._cache_client = None

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        if not self._cache_client:
            return None

        try:
            value = await self._cache_client.get(key)
            return self._deserialize(value) if value else None
        except Exception as e:
            logger.warning("Cache get failed for key %s: %s", key, e)
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache."""
        if not self._cache_client:
            return

        try:
            serialized_value = self._serialize(value)
            await self._cache_client.set(key, serialized_value, ttl=ttl)
        except Exception as e:
            logger.warning("Cache set failed for key %s: %s", key, e)

    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        if not self._cache_client:
            return

        try:
            await self._cache_client.delete(key)
        except Exception as e:
            logger.warning("Cache delete failed for key %s: %s", key, e)

    async def health_check(self) -> ServiceHealth:
        """Check service health."""
        try:
            if not self._cache_client:
                return ServiceHealth(
                    status=ServiceStatus.UNHEALTHY,
                    message="Cache client not initialized"
                )

            # Test basic operation
            test_key = "health_check_test"
            await self._cache_client.set(test_key, "test", ttl=1)
            result = await self._cache_client.get(test_key)

            if result == "test":
                return ServiceHealth(
                    status=ServiceStatus.HEALTHY,
                    message="Cache is operational"
                )
            else:
                return ServiceHealth(
                    status=ServiceStatus.DEGRADED,
                    message="Cache operations not working correctly"
                )

        except Exception as e:
            return ServiceHealth(
                status=ServiceStatus.UNHEALTHY,
                message=f"Cache health check failed: {e}"
            )

    def _serialize(self, value: Any) -> str:
        """Serialize value for storage."""
        import json
        try:
            return json.dumps(value)
        except TypeError:
            import pickle
            import base64
            return base64.b64encode(pickle.dumps(value)).decode()

    def _deserialize(self, value: str) -> Any:
        """Deserialize value from storage."""
        import json
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            import pickle
            import base64
            return pickle.loads(base64.b64decode(value.encode()))
```

## 📋 Extension Checklist

### Before You Start
- [ ] Review existing implementations for patterns
- [ ] Check if similar functionality already exists
- [ ] Understand the target integration (API docs, SDK, etc.)
- [ ] Plan configuration requirements
- [ ] Consider error handling and edge cases

### During Development
- [ ] Follow established naming conventions
- [ ] Implement all required protocol methods
- [ ] Add comprehensive input validation
- [ ] Include services layer integration
- [ ] Handle errors gracefully with appropriate exception types
- [ ] Add logging for debugging and monitoring
- [ ] Write docstrings with examples

### Testing Requirements
- [ ] Unit tests with mocked dependencies
- [ ] Integration tests with real services (when possible)
- [ ] Error condition testing
- [ ] Performance testing for high-volume operations
- [ ] Configuration validation testing

### Documentation
- [ ] Class and method docstrings
- [ ] Configuration examples
- [ ] Integration examples
- [ ] Error handling documentation
- [ ] Performance characteristics

### Quality Assurance
- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] No linting errors
- [ ] Type hints are complete
- [ ] No security vulnerabilities

## 🚀 Advanced Patterns

### Async Context Managers

```python
class MyAsyncBackend(BaseBackend):
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

# Usage
async with MyAsyncBackend(config) as backend:
    await backend.store_vectors(vectors)
    results = await backend.search(query_vector)
```

### Batch Processing

```python
class MyProvider(BaseProvider):
    async def embed_documents_batch(
        self,
        texts: list[str],
        batch_size: int | None = None
    ) -> list[list[float]]:
        """Process documents in batches."""
        batch_size = batch_size or self.max_batch_size
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await self.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)

            # Rate limiting between batches
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)

        return all_embeddings
```

### Health Monitoring

```python
class MyService(BaseServiceProvider):
    async def health_check(self) -> ServiceHealth:
        """Comprehensive health check."""
        checks = {}
        overall_status = ServiceStatus.HEALTHY

        # Check connections
        try:
            await self._client.ping()
            checks["connection"] = "healthy"
        except Exception as e:
            checks["connection"] = f"unhealthy: {e}"
            overall_status = ServiceStatus.UNHEALTHY

        # Check performance
        start_time = time.time()
        try:
            await self._client.test_operation()
            response_time = time.time() - start_time
            if response_time > 5.0:
                checks["performance"] = f"degraded: {response_time:.2f}s response"
                overall_status = ServiceStatus.DEGRADED
            else:
                checks["performance"] = f"healthy: {response_time:.2f}s response"
        except Exception as e:
            checks["performance"] = f"unhealthy: {e}"
            overall_status = ServiceStatus.UNHEALTHY

        # Check resources
        resource_usage = await self._get_resource_usage()
        if resource_usage > 0.9:
            checks["resources"] = f"degraded: {resource_usage:.1%} usage"
            overall_status = ServiceStatus.DEGRADED
        else:
            checks["resources"] = f"healthy: {resource_usage:.1%} usage"

        return ServiceHealth(
            status=overall_status,
            message=f"Health check completed with {len(checks)} checks",
            details=checks
        )
```

## 📚 Additional Resources

### Code Examples
- **[Provider Examples](../extension-development/providers.md)** - Detailed provider implementations
- **[Backend Examples](../extension-development/backends.md)** - Vector database integrations
- **[Service Examples](../extension-development/services.md)** - Cross-cutting services

### API Documentation
- **[Protocols Reference](../reference/protocols.md)** - Interface definitions
- **[Type System](../reference/types.md)** - Type definitions and data structures
- **[Error Handling](../reference/error-codes.md)** - Exception types and patterns

### Development Resources
- **[Development Patterns](development_patterns.md)** - Coding standards
- **[Testing Guide](../extension-development/testing.md)** - Testing best practices
- **[Performance Guide](../extension-development/performance.md)** - Optimization techniques

---

## 🎯 Ready to Build?

1. **Choose your extension type** - Provider, Backend, Source, or Service
2. **Study existing implementations** in the codebase
3. **Follow the development patterns** from this guide
4. **Test thoroughly** with both unit and integration tests
5. **Submit a pull request** following our [contribution guidelines](contributing.md)

*Need help getting started? Ask in [GitHub Discussions](https://github.com/knitli/codeweaver-mcp/discussions) or [email us](mailto:adam@knit.li).*
