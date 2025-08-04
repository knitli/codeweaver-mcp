# Backend Configuration

Vector database backends store and search your code embeddings. CodeWeaver supports multiple backends with different capabilities, performance characteristics, and deployment options.

## Available Backends

### Qdrant (Recommended)

**High-performance vector database** with excellent scalability and hybrid search capabilities.

#### Quick Setup

=== "Qdrant Cloud"

    ```bash
    # Sign up at https://qdrant.tech/
    export CW_VECTOR_BACKEND_URL="https://xyz-abc123.qdrant.io"
    export CW_VECTOR_BACKEND_API_KEY="your-qdrant-api-key"
    ```

=== "Local Docker"

    ```bash
    # Start Qdrant locally
    docker run -p 6333:6333 qdrant/qdrant
    
    # Configure CodeWeaver
    export CW_VECTOR_BACKEND_URL="http://localhost:6333"
    ```

=== "TOML Configuration"

    ```toml
    [backend]
    provider = "qdrant"
    url = "https://xyz-abc123.qdrant.io"
    api_key = "your-qdrant-api-key"
    collection = "codeweaver-embeddings"
    timeout = 30
    ```

#### Advanced Configuration

```toml
[backend]
provider = "qdrant"
url = "https://xyz-abc123.qdrant.io"
api_key = "your-qdrant-api-key"

# Collection settings
collection = "codeweaver-embeddings"
vector_size = 1024  # Match your embedding model
distance_metric = "cosine"  # cosine, euclidean, dot

# Performance settings
timeout = 30
max_retries = 3
batch_size = 100
prefer_grpc = true  # Use gRPC for better performance

# Hybrid search (requires Qdrant 1.1+)
enable_hybrid_search = true
sparse_vector_name = "sparse"
text_field_name = "text"
```

#### Hybrid Search Setup

Enable both vector and keyword search:

```toml
[backend]
provider = "qdrant"
enable_hybrid_search = true

# Environment variable alternative
# CW_ENABLE_HYBRID_SEARCH=true
```

#### Qdrant Cloud Regions

Choose the region closest to your users:

```toml
[backend]
provider = "qdrant"
# EU regions
url = "https://xyz.eu-central-1.qdrant.io"   # Frankfurt
url = "https://xyz.eu-west-1.qdrant.io"     # Ireland

# US regions  
url = "https://xyz.us-east-1.qdrant.io"     # N. Virginia
url = "https://xyz.us-west-1.qdrant.io"     # Oregon

# Asia regions
url = "https://xyz.ap-southeast-1.qdrant.io" # Singapore
```

---

### DocArray

**Unified interface** for multiple vector databases with a consistent API.

#### Configuration

```toml
[backend]
provider = "docarray"
backend_type = "qdrant"  # qdrant, memory, weaviate

# Qdrant via DocArray
qdrant_url = "https://xyz.qdrant.io"
qdrant_api_key = "your-api-key"
collection_name = "codeweaver"
```

#### Supported DocArray Backends

| Backend | Use Case | Configuration |
|---------|----------|---------------|
| `qdrant` | Production | Requires Qdrant URL and API key |
| `memory` | Testing | No external dependencies |
| `weaviate` | Alternative | Requires Weaviate instance |

#### Memory Backend (Testing)

```toml
[backend]
provider = "docarray"
backend_type = "memory"
# No additional configuration needed
```

---

### Custom Backends

Implement custom backends using the `VectorBackend` protocol.

#### Backend Interface

```python
from codeweaver.cw_types import VectorBackend, VectorPoint, SearchResult

class MyCustomBackend(VectorBackend):
    async def upsert_vectors(self, vectors: list[VectorPoint]) -> None:
        # Store vectors in your database
        pass
    
    async def search_vectors(
        self, 
        query_vector: list[float], 
        limit: int = 10
    ) -> list[SearchResult]:
        # Search and return results
        pass
    
    async def delete_collection(self) -> None:
        # Clean up collection
        pass
```

#### Registration

```python
from codeweaver.factories import codeweaver_factory

codeweaver_factory.register_vector_backend(
    "my_backend",
    MyCustomBackend
)
```

## Backend Comparison

### Performance Comparison

| Backend | Speed | Scalability | Hybrid Search | Local Option |
|---------|-------|-------------|---------------|--------------|
| **Qdrant** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | ✅ |
| **DocArray + Qdrant** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | ✅ |
| **DocArray + Memory** | ⭐⭐⭐ | ⭐⭐ | ❌ | ✅ |
| **DocArray + Weaviate** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ✅ |

### Feature Matrix

| Feature | Qdrant | DocArray Memory | DocArray Weaviate |
|---------|--------|-----------------|-------------------|
| **Persistence** | ✅ | ❌ | ✅ |
| **Clustering** | ✅ | ❌ | ✅ |
| **Filtering** | ✅ | ✅ | ✅ |
| **Hybrid Search** | ✅ | ❌ | ✅ |
| **CRUD Operations** | ✅ | ✅ | ✅ |
| **Backup/Restore** | ✅ | ❌ | ✅ |

## Deployment Patterns

### Development Setup

For local development and testing:

```toml
[backend]
provider = "docarray"
backend_type = "memory"
# Fast startup, no external dependencies
```

### Production Setup

For production workloads:

```toml
[backend]
provider = "qdrant"
url = "https://your-cluster.qdrant.io"
api_key = "your-api-key"
enable_hybrid_search = true
prefer_grpc = true
```

### Multi-Environment Configuration

Use environment-specific configuration:

=== "Development"

    ```toml
    # .codeweaver.dev.toml
    [backend]
    provider = "docarray"
    backend_type = "memory"
    ```

=== "Staging"

    ```toml
    # .codeweaver.staging.toml
    [backend]
    provider = "qdrant"
    url = "https://staging.qdrant.io"
    collection = "codeweaver-staging"
    ```

=== "Production"

    ```toml
    # .codeweaver.prod.toml
    [backend]
    provider = "qdrant"
    url = "https://prod.qdrant.io"
    collection = "codeweaver-production"
    enable_hybrid_search = true
    prefer_grpc = true
    ```

## Performance Optimization

### Qdrant Optimization

#### Connection Settings

```toml
[backend]
provider = "qdrant"
# Enable gRPC for better performance
prefer_grpc = true
# Connection pooling
max_connections = 10
# Timeouts
timeout = 30
```

#### Indexing Performance

```toml
[backend]
# Batch operations for better throughput
batch_size = 100
# Parallel uploads
max_concurrent_uploads = 5

# Index configuration
[backend.index_config]
# HNSW parameters for speed vs accuracy tradeoff
m = 16              # Number of bi-directional links
ef_construct = 200  # Size of dynamic candidate list
```

#### Memory Management

```toml
[backend]
# Optimize memory usage
payload_storage_type = "on_disk"  # Store payloads on disk
vector_storage_type = "memmap"    # Memory-mapped vectors
```

### Search Optimization

#### Vector Search Tuning

```toml
[backend]
# Search parameters
ef = 128              # Size of search candidate list
search_batch_size = 32 # Batch multiple searches

# Result filtering
enable_payload_index = true  # Index payload fields for filtering
```

#### Hybrid Search Tuning

```toml
[backend]
enable_hybrid_search = true

# Balance between vector and keyword search
hybrid_alpha = 0.7  # 0.0 = pure keyword, 1.0 = pure vector

# Sparse vector settings
sparse_dimension = 30000
sparse_compression = true
```

## Monitoring and Maintenance

### Health Monitoring

```toml
[backend]
# Health check settings
health_check_interval = 60  # seconds
enable_monitoring = true

# Metrics collection
collect_metrics = true
metrics_interval = 300  # 5 minutes
```

### Backup Configuration

```toml
[backend]
# Automatic backups (Qdrant Cloud)
enable_backups = true
backup_retention_days = 30

# Manual backup location
backup_path = "/path/to/backups"
```

### Maintenance Windows

```toml
[backend]
# Maintenance settings
maintenance_window = "02:00-04:00"  # UTC time
auto_optimize = true
optimization_interval = 86400  # daily
```

## Security Configuration

### Authentication

```toml
[backend]
# API key authentication
api_key = "your-secure-api-key"

# Certificate-based authentication (for self-hosted)
tls_cert_path = "/path/to/cert.pem"
tls_key_path = "/path/to/key.pem"
verify_ssl = true
```

### Network Security

```toml
[backend]
# Network configuration
allowed_hosts = ["your-app-domain.com"]
enable_cors = false

# VPC/Private network
use_private_network = true
private_endpoint = "https://private.qdrant.internal"
```

### Data Encryption

```toml
[backend]
# Encryption at rest (Qdrant Cloud)
encryption_enabled = true

# Client-side encryption
encrypt_payloads = true
encryption_key = "your-encryption-key"
```

## Troubleshooting Backends

### Connection Issues

#### Qdrant Connection Failed

```bash
# Test connection
curl -X GET "https://your-cluster.qdrant.io/collections"
```

**Common solutions:**
- Verify URL format includes `https://`
- Check API key permissions
- Confirm network connectivity

#### Timeout Errors

```toml
[backend]
# Increase timeouts
timeout = 60
max_retries = 5
retry_delay = 2.0
```

### Performance Issues

#### Slow Search Performance

1. **Enable gRPC**: `prefer_grpc = true`
2. **Optimize HNSW parameters**: Increase `ef` for accuracy vs speed
3. **Use filtering**: Add payload indexes for common filters
4. **Batch searches**: Use `search_batch_size`

#### High Memory Usage

1. **Use disk storage**: `payload_storage_type = "on_disk"`
2. **Memory-mapped vectors**: `vector_storage_type = "memmap"`
3. **Reduce batch sizes**: Lower `batch_size`

### Common Errors

#### Collection Not Found

```
Error: Collection 'codeweaver-embeddings' not found
```

**Solution:** Collection is created automatically on first use, or create manually:

```bash
curl -X PUT "https://your-cluster.qdrant.io/collections/codeweaver-embeddings" \
  -H "api-key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"vectors": {"size": 1024, "distance": "Cosine"}}'
```

#### Dimension Mismatch

```
Error: Vector dimension mismatch: expected 1024, got 1536
```

**Solution:** Ensure backend `vector_size` matches embedding model dimensions:

```toml
[backend]
vector_size = 1536  # Match your embedding model

[providers.openai]
dimensions = 1536   # Same dimension
```

## Migration Between Backends

### Backup Current Data

```python
from codeweaver.client import CodeWeaverClient

# Export current embeddings
client = CodeWeaverClient()
embeddings = await client.export_embeddings()

# Save to file
import json
with open('embeddings_backup.json', 'w') as f:
    json.dump(embeddings, f)
```

### Restore to New Backend

```python
# Configure new backend
new_client = CodeWeaverClient(new_config)

# Import embeddings
with open('embeddings_backup.json', 'r') as f:
    embeddings = json.load(f)

await new_client.import_embeddings(embeddings)
```

## Next Steps

- **Service configuration**: [Services Configuration](./services.md)
- **Provider setup**: [Provider Configuration](./providers.md)
- **Performance tuning**: [Performance Guide](../user-guide/performance.md)
- **Monitoring setup**: [Monitoring Guide](../user-guide/monitoring.md)