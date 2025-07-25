# Vector Database Backend Abstraction

Comprehensive vector database backend protocols supporting 15+ vector databases with hybrid search capabilities and extensible design.

## Overview

The backend abstraction provides a unified interface for vector databases, enabling CodeWeaver to support multiple providers while maintaining backward compatibility with existing Qdrant deployments.

### Supported Backends

**Currently Implemented:**
- ✅ **Qdrant** - Full support including hybrid search with sparse vectors

**Planned Implementations:**
- 🔄 **Pinecone** - Cloud-native vector database
- 🔄 **Chroma** - Open-source embedding database
- 🔄 **Weaviate** - Vector database with hybrid search
- 🔄 **pgvector** - PostgreSQL extension for vectors
- 🔄 **Redis** - With RediSearch vector capabilities
- 🔄 **Elasticsearch** - With dense_vector field type
- 🔄 **OpenSearch** - With vector search capabilities
- 🔄 **Milvus** - Open-source vector database
- 🔄 **Vespa** - Search engine with vector support
- 🔄 **FAISS** - In-memory similarity search
- 🔄 **Annoy** - Approximate nearest neighbors
- 🔄 **ScaNN** - Google's vector search library
- 🔄 **LanceDB** - Vector database for AI applications
- 🔄 **Marqo** - Multi-modal vector search

## Architecture

### Core Protocols

```python
from codeweaver.backends import VectorBackend, HybridSearchBackend

# Basic vector operations
backend: VectorBackend = create_backend(config)
await backend.create_collection("my-collection", dimension=1024)
await backend.upsert_vectors("my-collection", vectors)
results = await backend.search_vectors("my-collection", query_vector)

# Advanced hybrid search (if supported)
if isinstance(backend, HybridSearchBackend):
    await backend.create_sparse_index("my-collection", ["content"])
    results = await backend.hybrid_search(
        "my-collection",
        dense_vector=dense_vector,
        sparse_query={"keyword": 1.0},
        hybrid_strategy="rrf"
    )
```

### Universal Data Structures

All backends use standardized data structures for interoperability:

```python
# Vector points with optional sparse vectors
vector_point = VectorPoint(
    id="unique-id",
    vector=[0.1, 0.2, ...],  # Dense embedding
    sparse_vector={123: 0.8, 456: 0.6},  # Optional sparse vector
    payload={"metadata": "value"}
)

# Search results with consistent format
search_result = SearchResult(
    id="result-id",
    score=0.95,
    payload={"metadata": "value"},
    backend_metadata={"provider_specific": "info"}
)

# Collection information
collection_info = CollectionInfo(
    name="collection-name",
    dimension=1024,
    points_count=10000,
    supports_hybrid_search=True,
    supports_sparse_vectors=True
)
```

### Advanced Filtering

Universal filtering system that translates to backend-specific queries:

```python
from codeweaver.backends import SearchFilter, FilterCondition

# Complex filter with nested logic
search_filter = SearchFilter(
    must=[  # AND conditions
        SearchFilter(conditions=[
            FilterCondition(field="language", operator="eq", value="python"),
            FilterCondition(field="file_size", operator="gt", value=1000),
        ]),
    ],
    should=[  # OR conditions
        SearchFilter(conditions=[
            FilterCondition(field="chunk_type", operator="eq", value="function"),
        ]),
    ],
    must_not=[  # NOT conditions
        SearchFilter(conditions=[
            FilterCondition(field="deprecated", operator="eq", value=True),
        ]),
    ]
)

results = await backend.search_vectors(
    collection_name="my-collection",
    query_vector=query_vector,
    search_filter=search_filter
)
```

## Configuration

### TOML Configuration

```toml
[backend]
provider = "qdrant"  # or pinecone, chroma, weaviate, etc.
url = "https://your-cluster.qdrant.io"
api_key = "your-api-key"
collection_name = "code-embeddings"

# Feature capabilities
enable_hybrid_search = true
enable_sparse_vectors = true
enable_streaming = false

# Performance settings
batch_size = 100
connection_timeout = 30.0
request_timeout = 60.0
max_connections = 10

# Hybrid search configuration
[backend.hybrid_search]
sparse_index_fields = ["content", "chunk_type"]
sparse_index_type = "bm25"
fusion_strategy = "rrf"
alpha = 0.7  # Balance between dense (1.0) and sparse (0.0)

# Provider-specific options
[backend.provider_options]
# Qdrant-specific
sparse_on_disk = false
prefer_grpc = true

# Pinecone-specific (when implemented)
# environment = "us-west1-gcp"
# index_type = "serverless"
```

### Environment Variables

```bash
# Backend selection
export VECTOR_BACKEND_PROVIDER=qdrant
export VECTOR_BACKEND_URL=https://your-cluster.qdrant.io
export VECTOR_BACKEND_API_KEY=your-api-key

export VECTOR_BACKEND_COLLECTION=code-embeddings

# Feature flags
export ENABLE_HYBRID_SEARCH=true
export ENABLE_SPARSE_VECTORS=true

# Performance tuning
export BACKEND_BATCH_SIZE=100
export BACKEND_CONNECTION_TIMEOUT=30.0
export BACKEND_REQUEST_TIMEOUT=60.0
```

### Programmatic Configuration

```python
from codeweaver.backends import BackendFactory, BackendConfigExtended

# Create configuration
config = BackendConfigExtended(
    provider="qdrant",
    url="https://your-cluster.qdrant.io",
    api_key="your-api-key",
    collection_name="code-embeddings",
    enable_hybrid_search=True,
    enable_sparse_vectors=True,
    batch_size=100,
)

# Create backend
backend = BackendFactory.create_backend(config)

# Or create from URL
backend = BackendFactory.create_from_url(
    "qdrant://api-key@cluster-url:6333/collection-name"
)
```

## Migration Guide

### From Current Qdrant Implementation

The backend abstraction maintains full backward compatibility:

```python
# OLD: Direct Qdrant client usage
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams

qdrant = QdrantClient(url=url, api_key=api_key)
qdrant.create_collection(
    collection_name="code-embeddings",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
)

# NEW: Backend abstraction (same functionality)
from codeweaver.backends import BackendFactory, BackendConfigExtended

config = BackendConfigExtended(
    provider="qdrant",
    url=url,
    api_key=api_key,
    collection_name="code-embeddings"
)
backend = BackendFactory.create_backend(config)
await backend.create_collection("code-embeddings", dimension=1024)
```

### Backend Setup

The backend system uses a unified factory pattern for all vector databases:

```python
from codeweaver.backends.factory import BackendConfig, BackendFactory
from codeweaver.providers.base import ProviderKind

# Create backend configuration
config = BackendConfig(
    provider="qdrant",
    kind=ProviderKind.COMBINED,
    url="https://your-cluster.qdrant.io",
    api_key="your-api-key",
    collection_name="code-embeddings"
)

# Create backend instance
backend = BackendFactory.create_backend(config)

# Use universal backend operations
await backend.create_collection("code-embeddings", dimension=1024)
await backend.upsert_vectors(collection_name="code-embeddings", vectors=vector_points)
```

## Hybrid Search

### Qdrant Implementation

Leverages Qdrant's native sparse vector support (v1.10+):

```python
from codeweaver.backends import BackendFactory

# Enable hybrid search
config = BackendConfigExtended(
    provider="qdrant",
    enable_hybrid_search=True,
    enable_sparse_vectors=True
)

backend = BackendFactory.create_backend(config)

# Create collection with sparse vector support
await backend.create_collection("hybrid-collection", dimension=1024)
await backend.create_sparse_index("hybrid-collection", ["content", "chunk_type"])

# Perform hybrid search
results = await backend.hybrid_search(
    collection_name="hybrid-collection",
    dense_vector=code_embedding,  # From Voyage AI
    sparse_query={"function": 1.0, "python": 0.8},  # BM25-style
    hybrid_strategy="rrf",  # Reciprocal Rank Fusion
    alpha=0.7  # Favor dense search
)
```

### Fusion Strategies

- **RRF (Reciprocal Rank Fusion)**: `1 / (k + rank)` scoring
- **DBSF (Distribution-Based Score Fusion)**: Score distribution normalization
- **Linear**: Weighted combination of scores
- **Convex**: Convex combination with alpha parameter

## Error Handling

Comprehensive error handling with backend-specific context:

```python
from codeweaver.backends import (
    BackendError,
    BackendConnectionError,
    BackendAuthError,
    BackendCollectionNotFoundError,
    BackendVectorDimensionMismatchError,
    BackendUnsupportedOperationError
)

try:
    await backend.search_vectors(collection_name, query_vector)
except BackendConnectionError as e:
    logger.error("Backend connection failed: %s", e)
    # Handle connection issues
except BackendCollectionNotFoundError as e:
    logger.error("Collection not found: %s", e)
    # Handle missing collection
except BackendUnsupportedOperationError as e:
    logger.error("Operation not supported: %s", e)
    # Fallback to alternative approach
```

## Extending the System

### Adding New Backends

1. **Implement the Protocol**
   ```python
   from codeweaver.backends.base import VectorBackend

   class CustomBackend:
       async def create_collection(self, name: str, dimension: int, **kwargs):
           # Implementation specific to your backend
           pass

       # Implement other required methods...
   ```

2. **Register with Factory**
   ```python
   from codeweaver.backends import BackendFactory

   BackendFactory.register_backend(
       provider="custom",
       backend_class=CustomBackend,
       supports_hybrid=False
   )
   ```

3. **Add Configuration Support**
   ```python
   # Add to backend factory's provider-specific config
   def _build_backend_args(cls, config):
       if config.provider == "custom":
           return {
               "endpoint": config.url,
               "auth_token": config.api_key,
               # Custom backend options
           }
   ```

### Backend Capabilities Matrix

| Backend | Vector Search | Hybrid Search | Streaming | Transactions | Filtering |
|---------|---------------|---------------|-----------|--------------|-----------|
| Qdrant | ✅ | ✅ | ⏳ | ⏳ | ✅ |
| Pinecone | 🔄 | ❌ | 🔄 | ❌ | 🔄 |
| Chroma | 🔄 | ❌ | 🔄 | ❌ | 🔄 |
| Weaviate | 🔄 | 🔄 | 🔄 | ❌ | 🔄 |
| pgvector | 🔄 | ❌ | 🔄 | ✅ | 🔄 |

Legend: ✅ Implemented, 🔄 Planned, ⏳ In Progress, ❌ Not Supported

## Performance Considerations

### Batch Operations
```python
# Efficient batch processing
vectors = [VectorPoint(...) for _ in range(1000)]
await backend.upsert_vectors(collection_name, vectors)
```

### Connection Pooling
```python
config = BackendConfigExtended(
    provider="qdrant",
    enable_connection_pooling=True,
    connection_pool_size=10,
    max_connections=50
)
```

### Caching
```python
config = BackendConfigExtended(
    enable_result_caching=True,
    cache_ttl_seconds=300
)
```

## Best Practices

1. **Use Type Hints**: All protocols are properly typed for IDE support
2. **Handle Capabilities**: Check backend capabilities before using advanced features
3. **Error Recovery**: Implement proper error handling and fallbacks
4. **Resource Management**: Use connection pooling for production deployments
5. **Monitoring**: Enable metrics and tracing for observability

## Testing

```python
# Test backend compatibility
from codeweaver.backends.base import VectorBackend

def test_backend_compatibility(backend: VectorBackend):
    """Test that a backend implements the protocol correctly."""
    assert hasattr(backend, 'create_collection')
    assert hasattr(backend, 'upsert_vectors')
    assert hasattr(backend, 'search_vectors')
    # ... more protocol checks
```

This backend abstraction provides a robust foundation for CodeWeaver's extensibility while maintaining the performance and functionality users expect.
