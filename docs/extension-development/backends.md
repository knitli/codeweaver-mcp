# Building Custom Vector Backends

This guide covers building custom vector backends for CodeWeaver. Vector backends handle storage, indexing, and search operations for semantic code search.

## 🎯 Overview

Vector backends provide the storage and search infrastructure for CodeWeaver's semantic search capabilities. CodeWeaver supports multiple backend types:

- **Vector Backends**: Basic vector storage and similarity search
- **Hybrid Search Backends**: Combined dense + sparse vector search
- **Streaming Backends**: High-throughput streaming operations
- **Transactional Backends**: ACID transaction support

## 🏗️ Backend Architecture

### Core VectorBackend Protocol

```python
from typing import Protocol, runtime_checkable, Any
from codeweaver.cw_types import (
    VectorPoint, SearchResult, SearchFilter,
    CollectionInfo, DistanceMetric
)

@runtime_checkable
class VectorBackend(Protocol):
    # Collection Management
    async def create_collection(
        self,
        name: str,
        dimension: int,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs: Any
    ) -> None: ...

    async def list_collections(self) -> list[str]: ...
    async def delete_collection(self, name: str) -> None: ...
    async def get_collection_info(self, name: str) -> CollectionInfo: ...

    # Vector Operations
    async def upsert_vectors(
        self,
        collection_name: str,
        vectors: list[VectorPoint]
    ) -> None: ...

    async def search_vectors(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        search_filter: SearchFilter | None = None,
        score_threshold: float | None = None,
        **kwargs: Any
    ) -> list[SearchResult]: ...

    async def delete_vectors(
        self,
        collection_name: str,
        ids: list[str | int]
    ) -> None: ...

    # Health Check
    async def health_check(self) -> bool: ...
```

### Extended Backend Protocols

#### HybridSearchBackend
```python
from codeweaver.cw_types import HybridStrategy

@runtime_checkable
class HybridSearchBackend(VectorBackend, Protocol):
    # Sparse Index Management
    async def create_sparse_index(
        self,
        collection_name: str,
        fields: list[str],
        index_type: Literal["keyword", "text", "bm25"] = "bm25",
        **kwargs: Any
    ) -> None: ...

    # Hybrid Search
    async def hybrid_search(
        self,
        collection_name: str,
        dense_vector: list[float],
        sparse_query: dict[str, float] | str,
        limit: int = 10,
        hybrid_strategy: HybridStrategy = HybridStrategy.RRF,
        alpha: float = 0.5,
        search_filter: SearchFilter | None = None,
        **kwargs: Any
    ) -> list[SearchResult]: ...

    # Sparse Vector Updates
    async def update_sparse_vectors(
        self,
        collection_name: str,
        vectors: list[VectorPoint]
    ) -> None: ...
```

#### StreamingBackend
```python
from typing import AsyncIterator

@runtime_checkable
class StreamingBackend(VectorBackend, Protocol):
    async def stream_upsert_vectors(
        self,
        collection_name: str,
        vector_stream: AsyncIterator[list[VectorPoint]],
        batch_size: int = 100
    ) -> None: ...

    async def stream_search_vectors(
        self,
        collection_name: str,
        query_stream: AsyncIterator[list[float]],
        limit: int = 10
    ) -> AsyncIterator[list[SearchResult]]: ...
```

## 🚀 Implementation Guide

### Step 1: Define Configuration

Create a Pydantic configuration model for your backend:

```python
from pydantic import BaseModel, Field
from typing import Annotated

class MyBackendConfig(BaseModel):
    """Configuration for MyVectorBackend."""

    host: Annotated[str, Field(description="Backend host address")]
    port: Annotated[int, Field(default=6333, ge=1, le=65535)]
    api_key: Annotated[str | None, Field(default=None, description="API key")]
    timeout: Annotated[int, Field(default=30, ge=1, le=300)]
    max_connections: Annotated[int, Field(default=10, ge=1, le=100)]
    ssl_enabled: Annotated[bool, Field(default=False)]
    collection_config: Annotated[dict[str, Any], Field(default_factory=dict)]
```

### Step 2: Implement Backend Class

```python
from codeweaver.backends.base import VectorBackend
from codeweaver.cw_types import (
    VectorPoint, SearchResult, SearchFilter, CollectionInfo,
    DistanceMetric, BackendConnectionError
)
import aiohttp
import asyncio
from typing import Any

class MyVectorBackend:
    """Custom vector backend implementation."""

    def __init__(self, config: MyBackendConfig):
        self.config = config
        self.client = None
        self._collections: set[str] = set()

    async def _initialize_client(self) -> None:
        """Initialize backend client connection."""
        if self.client is None:
            try:
                timeout = aiohttp.ClientTimeout(total=self.config.timeout)
                connector = aiohttp.TCPConnector(
                    limit=self.config.max_connections,
                    ssl=self.config.ssl_enabled
                )

                headers = {}
                if self.config.api_key:
                    headers["Authorization"] = f"Bearer {self.config.api_key}"

                self.client = aiohttp.ClientSession(
                    timeout=timeout,
                    connector=connector,
                    headers=headers
                )

                # Test connection
                await self._test_connection()

            except Exception as e:
                raise BackendConnectionError(f"Failed to connect: {e}") from e

    async def _cleanup_client(self) -> None:
        """Cleanup client connection."""
        if self.client:
            await self.client.close()
            self.client = None

    async def _test_connection(self) -> None:
        """Test backend connectivity."""
        base_url = f"{'https' if self.config.ssl_enabled else 'http'}://{self.config.host}:{self.config.port}"

        async with self.client.get(f"{base_url}/health") as response:
            response.raise_for_status()

    # Collection Management
    async def create_collection(
        self,
        name: str,
        dimension: int,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs: Any
    ) -> None:
        """Create a new vector collection."""
        await self._initialize_client()

        # Convert distance metric to backend format
        backend_metric = self._convert_distance_metric(distance_metric)

        collection_config = {
            "vectors": {
                "size": dimension,
                "distance": backend_metric,
                **self.config.collection_config,
                **kwargs
            }
        }

        base_url = f"{'https' if self.config.ssl_enabled else 'http'}://{self.config.host}:{self.config.port}"

        async with self.client.put(
            f"{base_url}/collections/{name}",
            json=collection_config
        ) as response:
            response.raise_for_status()
            self._collections.add(name)

    async def list_collections(self) -> list[str]:
        """List all collections."""
        await self._initialize_client()

        base_url = f"{'https' if self.config.ssl_enabled else 'http'}://{self.config.host}:{self.config.port}"

        async with self.client.get(f"{base_url}/collections") as response:
            response.raise_for_status()
            data = await response.json()

            collections = []
            for collection in data.get("result", {}).get("collections", []):
                collections.append(collection["name"])

            self._collections.update(collections)
            return collections

    async def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        await self._initialize_client()

        base_url = f"{'https' if self.config.ssl_enabled else 'http'}://{self.config.host}:{self.config.port}"

        async with self.client.delete(f"{base_url}/collections/{name}") as response:
            response.raise_for_status()
            self._collections.discard(name)

    async def get_collection_info(self, name: str) -> CollectionInfo:
        """Get collection information."""
        await self._initialize_client()

        base_url = f"{'https' if self.config.ssl_enabled else 'http'}://{self.config.host}:{self.config.port}"

        async with self.client.get(f"{base_url}/collections/{name}") as response:
            response.raise_for_status()
            data = await response.json()

            result = data["result"]
            config = result["config"]

            return CollectionInfo(
                name=name,
                dimension=config["params"]["vectors"]["size"],
                distance_metric=self._convert_from_backend_metric(
                    config["params"]["vectors"]["distance"]
                ),
                vector_count=result["vectors_count"],
                indexed_vector_count=result.get("indexed_vectors_count", 0)
            )

    # Vector Operations
    async def upsert_vectors(
        self,
        collection_name: str,
        vectors: list[VectorPoint]
    ) -> None:
        """Insert or update vectors."""
        await self._initialize_client()

        # Convert VectorPoint objects to backend format
        points = []
        for vector in vectors:
            point = {
                "id": vector.id,
                "vector": vector.vector,
                "payload": vector.payload or {}
            }
            points.append(point)

        base_url = f"{'https' if self.config.ssl_enabled else 'http'}://{self.config.host}:{self.config.port}"

        # Process in batches for large datasets
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]

            payload = {
                "points": batch,
                "wait": True  # Wait for operation to complete
            }

            async with self.client.put(
                f"{base_url}/collections/{collection_name}/points",
                json=payload
            ) as response:
                response.raise_for_status()

    async def search_vectors(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        search_filter: SearchFilter | None = None,
        score_threshold: float | None = None,
        **kwargs: Any
    ) -> list[SearchResult]:
        """Search for similar vectors."""
        await self._initialize_client()

        query = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": True,
            "with_vectors": False  # Don't return vectors by default
        }

        # Add filter if provided
        if search_filter:
            query["filter"] = self._convert_search_filter(search_filter)

        # Add score threshold
        if score_threshold is not None:
            query["score_threshold"] = score_threshold

        # Add any additional parameters
        query.update(kwargs)

        base_url = f"{'https' if self.config.ssl_enabled else 'http'}://{self.config.host}:{self.config.port}"

        async with self.client.post(
            f"{base_url}/collections/{collection_name}/points/search",
            json=query
        ) as response:
            response.raise_for_status()
            data = await response.json()

            results = []
            for hit in data["result"]:
                result = SearchResult(
                    id=hit["id"],
                    score=hit["score"],
                    payload=hit.get("payload", {}),
                    vector=hit.get("vector")  # Only if requested
                )
                results.append(result)

            return results

    async def delete_vectors(
        self,
        collection_name: str,
        ids: list[str | int]
    ) -> None:
        """Delete vectors by IDs."""
        await self._initialize_client()

        payload = {
            "points": ids,
            "wait": True
        }

        base_url = f"{'https' if self.config.ssl_enabled else 'http'}://{self.config.host}:{self.config.port}"

        async with self.client.post(
            f"{base_url}/collections/{collection_name}/points/delete",
            json=payload
        ) as response:
            response.raise_for_status()

    async def health_check(self) -> bool:
        """Check backend health."""
        try:
            await self._initialize_client()
            await self._test_connection()
            return True
        except Exception:
            return False
        finally:
            await self._cleanup_client()

    # Helper Methods
    def _convert_distance_metric(self, metric: DistanceMetric) -> str:
        """Convert DistanceMetric to backend format."""
        mapping = {
            DistanceMetric.COSINE: "Cosine",
            DistanceMetric.EUCLIDEAN: "Euclid",
            DistanceMetric.DOT_PRODUCT: "Dot"
        }
        return mapping.get(metric, "Cosine")

    def _convert_from_backend_metric(self, backend_metric: str) -> DistanceMetric:
        """Convert backend metric to DistanceMetric."""
        mapping = {
            "Cosine": DistanceMetric.COSINE,
            "Euclid": DistanceMetric.EUCLIDEAN,
            "Dot": DistanceMetric.DOT_PRODUCT
        }
        return mapping.get(backend_metric, DistanceMetric.COSINE)

    def _convert_search_filter(self, search_filter: SearchFilter) -> dict[str, Any]:
        """Convert SearchFilter to backend format."""
        # Convert SearchFilter object to your backend's filter format
        backend_filter = {}

        if search_filter.must:
            backend_filter["must"] = [
                self._convert_filter_condition(condition)
                for condition in search_filter.must
            ]

        if search_filter.must_not:
            backend_filter["must_not"] = [
                self._convert_filter_condition(condition)
                for condition in search_filter.must_not
            ]

        return backend_filter

    def _convert_filter_condition(self, condition: dict[str, Any]) -> dict[str, Any]:
        """Convert individual filter condition."""
        # Convert condition to your backend's format
        return condition  # Simplified - implement based on your backend
```

### Step 3: Create Plugin Interface

```python
from codeweaver.factories.plugin_protocols import BackendPlugin
from codeweaver.cw_types import (
    ComponentType, BaseCapabilities, BaseComponentInfo,
    ValidationResult, BackendCapabilities
)

class MyBackendPlugin(BackendPlugin):
    """Plugin interface for MyVectorBackend."""

    @classmethod
    def get_plugin_name(cls) -> str:
        return "my_vector_backend"

    @classmethod
    def get_component_type(cls) -> ComponentType:
        return ComponentType.BACKEND

    @classmethod
    def get_capabilities(cls) -> BaseCapabilities:
        return BackendCapabilities(
            supports_filtering=True,
            supports_hybrid_search=False,
            supports_streaming=False,
            supports_transactions=False,
            max_vector_dimension=1536,
            supported_distance_metrics=[
                DistanceMetric.COSINE,
                DistanceMetric.EUCLIDEAN,
                DistanceMetric.DOT_PRODUCT
            ]
        )

    @classmethod
    def get_component_info(cls) -> BaseComponentInfo:
        return BaseComponentInfo(
            name="my_vector_backend",
            display_name="My Vector Backend",
            description="Custom vector backend implementation",
            component_type=ComponentType.BACKEND,
            version="1.0.0",
            author="Your Name",
            homepage="https://github.com/yourname/my-backend"
        )

    @classmethod
    def validate_config(cls, config: MyBackendConfig) -> ValidationResult:
        """Validate backend configuration."""
        errors = []
        warnings = []

        if not config.host:
            errors.append("Host is required")

        if config.port < 1 or config.port > 65535:
            errors.append("Port must be between 1 and 65535")

        if config.timeout < 5:
            warnings.append("Timeout below 5 seconds may cause failures")

        if config.max_connections > 50:
            warnings.append("High connection count may impact performance")

        return ValidationResult(
            is_valid=not errors,
            errors=errors,
            warnings=warnings
        )

    @classmethod
    def get_dependencies(cls) -> list[str]:
        """Get required dependencies."""
        return ["aiohttp", "pydantic"]

    @classmethod
    def get_backend_class(cls) -> type[VectorBackend]:
        return MyVectorBackend
```

### Step 4: Register the Backend

```python
from codeweaver.factories.codeweaver_factory import CodeWeaverFactory

# Create factory instance
factory = CodeWeaverFactory()

# Register backend
factory.register_backend(
    "my_vector_backend",
    MyVectorBackend,
    MyBackendPlugin.get_capabilities(),
    MyBackendPlugin.get_component_info()
)
```

## 🔧 Advanced Features

### Hybrid Search Implementation

```python
from codeweaver.backends.base import HybridSearchBackend
from codeweaver.cw_types import HybridStrategy

class MyHybridBackend(MyVectorBackend, HybridSearchBackend):
    """Backend with hybrid search capabilities."""

    async def create_sparse_index(
        self,
        collection_name: str,
        fields: list[str],
        index_type: Literal["keyword", "text", "bm25"] = "bm25",
        **kwargs: Any
    ) -> None:
        """Create sparse index for hybrid search."""
        await self._initialize_client()

        index_config = {
            "field_name": fields[0],  # Simplified for example
            "field_schema": {
                "type": index_type,
                **kwargs
            }
        }

        base_url = f"{'https' if self.config.ssl_enabled else 'http'}://{self.config.host}:{self.config.port}"

        async with self.client.put(
            f"{base_url}/collections/{collection_name}/index",
            json=index_config
        ) as response:
            response.raise_for_status()

    async def hybrid_search(
        self,
        collection_name: str,
        dense_vector: list[float],
        sparse_query: dict[str, float] | str,
        limit: int = 10,
        hybrid_strategy: HybridStrategy = HybridStrategy.RRF,
        alpha: float = 0.5,
        search_filter: SearchFilter | None = None,
        **kwargs: Any
    ) -> list[SearchResult]:
        """Perform hybrid search."""
        await self._initialize_client()

        query = {
            "prefetch": [
                {
                    "query": {
                        "vector": dense_vector
                    },
                    "limit": limit * 2  # Get more for reranking
                },
                {
                    "query": {
                        "sparse": self._convert_sparse_query(sparse_query)
                    },
                    "limit": limit * 2
                }
            ],
            "query": {
                "fusion": self._convert_fusion_strategy(hybrid_strategy, alpha)
            },
            "limit": limit,
            "with_payload": True
        }

        if search_filter:
            query["filter"] = self._convert_search_filter(search_filter)

        base_url = f"{'https' if self.config.ssl_enabled else 'http'}://{self.config.host}:{self.config.port}"

        async with self.client.post(
            f"{base_url}/collections/{collection_name}/points/query",
            json=query
        ) as response:
            response.raise_for_status()
            data = await response.json()

            results = []
            for hit in data["result"]:
                result = SearchResult(
                    id=hit["id"],
                    score=hit["score"],
                    payload=hit.get("payload", {})
                )
                results.append(result)

            return results

    def _convert_sparse_query(self, sparse_query: dict[str, float] | str) -> dict[str, Any]:
        """Convert sparse query to backend format."""
        if isinstance(sparse_query, str):
            # Simple text query
            return {"text": sparse_query}
        else:
            # Term weights
            return {"indices": list(sparse_query.keys()), "values": list(sparse_query.values())}

    def _convert_fusion_strategy(self, strategy: HybridStrategy, alpha: float) -> dict[str, Any]:
        """Convert fusion strategy to backend format."""
        if strategy == HybridStrategy.RRF:
            return {"fusion": "rrf"}
        elif strategy == HybridStrategy.WEIGHTED:
            return {"fusion": "weighted", "alpha": alpha}
        else:
            return {"fusion": "rrf"}  # Default
```

### Streaming Implementation

```python
from typing import AsyncIterator

class MyStreamingBackend(MyVectorBackend, StreamingBackend):
    """Backend with streaming capabilities."""

    async def stream_upsert_vectors(
        self,
        collection_name: str,
        vector_stream: AsyncIterator[list[VectorPoint]],
        batch_size: int = 100
    ) -> None:
        """Stream upsert vectors in batches."""
        await self._initialize_client()

        batch = []
        async for vectors in vector_stream:
            batch.extend(vectors)

            if len(batch) >= batch_size:
                await self.upsert_vectors(collection_name, batch)
                batch = []

        # Process remaining vectors
        if batch:
            await self.upsert_vectors(collection_name, batch)

    async def stream_search_vectors(
        self,
        collection_name: str,
        query_stream: AsyncIterator[list[float]],
        limit: int = 10
    ) -> AsyncIterator[list[SearchResult]]:
        """Stream search multiple query vectors."""
        await self._initialize_client()

        async for query_vectors in query_stream:
            results = []
            for query_vector in query_vectors:
                search_results = await self.search_vectors(
                    collection_name, query_vector, limit
                )
                results.append(search_results)

            yield results
```

## 🧪 Testing Your Backend

### Unit Tests

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from my_package.backend import MyVectorBackend, MyBackendConfig
from codeweaver.cw_types import VectorPoint, DistanceMetric

@pytest.fixture
def backend_config():
    return MyBackendConfig(
        host="localhost",
        port=6333,
        api_key="test-key"
    )

@pytest.fixture
def backend(backend_config):
    return MyVectorBackend(backend_config)

@pytest.fixture
def mock_client():
    return AsyncMock()

class TestMyVectorBackend:
    """Test suite for MyVectorBackend."""

    async def test_create_collection(self, backend, mock_client):
        """Test collection creation."""
        backend.client = mock_client

        await backend.create_collection("test_collection", 128)

        mock_client.put.assert_called_once()
        call_args = mock_client.put.call_args
        assert "test_collection" in call_args[0][0]

    async def test_upsert_vectors(self, backend, mock_client):
        """Test vector upsert."""
        backend.client = mock_client

        vectors = [
            VectorPoint(id="1", vector=[0.1, 0.2], payload={"text": "test"}),
            VectorPoint(id="2", vector=[0.3, 0.4], payload={"text": "test2"})
        ]

        await backend.upsert_vectors("test_collection", vectors)

        mock_client.put.assert_called_once()
        call_args = mock_client.put.call_args
        json_data = call_args[1]["json"]
        assert len(json_data["points"]) == 2

    async def test_search_vectors(self, backend, mock_client):
        """Test vector search."""
        backend.client = mock_client

        # Mock search response
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "result": [
                {"id": "1", "score": 0.95, "payload": {"text": "test"}},
                {"id": "2", "score": 0.85, "payload": {"text": "test2"}}
            ]
        }
        mock_client.post.return_value.__aenter__.return_value = mock_response

        results = await backend.search_vectors(
            "test_collection",
            [0.1, 0.2],
            limit=10
        )

        assert len(results) == 2
        assert results[0].id == "1"
        assert results[0].score == 0.95

    async def test_health_check(self, backend):
        """Test health check."""
        with patch.object(backend, '_test_connection') as mock_test:
            with patch.object(backend, '_initialize_client'):
                with patch.object(backend, '_cleanup_client'):
                    mock_test.return_value = None

                    healthy = await backend.health_check()

                    assert healthy is True
```

### Integration Tests

```python
@pytest.mark.integration
class TestMyBackendIntegration:
    """Integration tests with real backend."""

    @pytest.fixture
    def real_backend(self):
        config = MyBackendConfig(
            host=os.getenv("TEST_BACKEND_HOST", "localhost"),
            port=int(os.getenv("TEST_BACKEND_PORT", "6333")),
            api_key=os.getenv("TEST_BACKEND_API_KEY")
        )
        return MyVectorBackend(config)

    async def test_full_workflow(self, real_backend):
        """Test complete workflow with real backend."""
        collection_name = f"test_collection_{int(time.time())}"

        try:
            # Create collection
            await real_backend.create_collection(collection_name, 128)

            # Verify collection exists
            collections = await real_backend.list_collections()
            assert collection_name in collections

            # Add vectors
            vectors = [
                VectorPoint(
                    id=f"vec_{i}",
                    vector=[random.random() for _ in range(128)],
                    payload={"text": f"document {i}"}
                )
                for i in range(10)
            ]
            await real_backend.upsert_vectors(collection_name, vectors)

            # Search vectors
            query_vector = [random.random() for _ in range(128)]
            results = await real_backend.search_vectors(
                collection_name,
                query_vector,
                limit=5
            )

            assert len(results) <= 5
            assert all(result.score > 0 for result in results)

        finally:
            # Cleanup
            await real_backend.delete_collection(collection_name)
```

## 📊 Performance Guidelines

### Connection Management
- Use connection pooling for better performance
- Implement proper connection cleanup
- Monitor connection health and retry logic

### Batch Operations
- Process vectors in optimal batch sizes
- Implement streaming for large datasets
- Use async operations for concurrent processing

### Memory Optimization
- Avoid loading all vectors into memory
- Use lazy loading and streaming where possible
- Monitor memory usage during operations

### Error Handling
- Implement comprehensive error handling
- Provide meaningful error messages
- Support graceful degradation

## 🚀 Next Steps

- **[Source Development :material-arrow-right-circle:](./sources.md)**: Learn about data source development
- **[Service Development :material-arrow-right-circle:](./services.md)**: Build middleware services
- **[Testing Framework :material-arrow-right-circle:](./testing.md)**: Comprehensive testing strategies
- **[Performance Guidelines :material-arrow-right-circle:](./performance.md)**: Optimization best practices
