<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Configuration Reference

**Complete guide to configuring CodeWeaver for your environment and use case**

This comprehensive reference covers all configuration options for CodeWeaver, from basic environment variables to advanced performance tuning and multi-provider setups.

## Configuration Methods

### 1. Environment Variables (Recommended)

```bash
# Core configuration
export CW_EMBEDDING_API_KEY="your-api-key"
export CW_VECTOR_BACKEND_URL="http://localhost:6333"
export CW_VECTOR_BACKEND_COLLECTION="my-project"
```

### 2. Configuration File

```toml
# codeweaver.toml
[embedding]
provider = "voyage"
api_key = "your-api-key"
model = "voyage-code-2"

[vector_backend]
type = "qdrant"
url = "http://localhost:6333"
collection = "my-project"

[chunking]
max_size = 1500
min_size = 50
overlap = 100
```

### 3. Runtime Configuration

```python
from codeweaver import CodeWeaver

config = {
    "embedding": {
        "provider": "voyage",
        "api_key": "your-key"
    },
    "vector_backend": {
        "type": "qdrant",
        "url": "http://localhost:6333"
    }
}

cw = CodeWeaver(config=config)
```

## Core Configuration

### Required Settings

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `CW_EMBEDDING_API_KEY` | ✅ | API key for embedding provider | `pa-your-voyage-key` |
| `CW_VECTOR_BACKEND_URL` | ✅ | Vector database URL | `http://localhost:6333` |

### Basic Settings

| Variable | Default | Description | Example |
|----------|---------|-------------|---------|
| `CW_VECTOR_BACKEND_COLLECTION` | `codeweaver-{uuid}` | Collection name | `my-project-codebase` |
| `CW_EMBEDDING_PROVIDER` | `voyage` | Embedding provider | `voyage`, `openai`, `cohere` |
| `CW_VECTOR_BACKEND_TYPE` | `qdrant` | Vector database type | `qdrant`, `pinecone`, `weaviate` |
| `CW_LOG_LEVEL` | `INFO` | Logging verbosity | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

## Embedding Provider Configuration

### Voyage AI (Recommended)

```bash
# Basic configuration
export CW_EMBEDDING_PROVIDER=voyage
export CW_EMBEDDING_API_KEY=pa-your-voyage-key
export CW_EMBEDDING_MODEL=voyage-code-2

# Advanced settings
export CW_VOYAGE_BASE_URL=https://api.voyageai.com/v1
export CW_VOYAGE_TIMEOUT=30
export CW_VOYAGE_MAX_RETRIES=3
export CW_VOYAGE_BATCH_SIZE=16
```

**Available Models:**
- `voyage-code-2` - Latest code-optimized model (recommended)
- `voyage-large-2` - General purpose, high quality
- `voyage-2` - Balanced performance and quality

**Configuration Options:**
```toml
[embedding.voyage]
model = "voyage-code-2"
base_url = "https://api.voyageai.com/v1"
timeout = 30
max_retries = 3
batch_size = 16
truncation = true
```

### OpenAI

```bash
# Basic configuration
export CW_EMBEDDING_PROVIDER=openai
export CW_EMBEDDING_API_KEY=sk-your-openai-key
export CW_EMBEDDING_MODEL=text-embedding-3-small

# Advanced settings
export CW_OPENAI_BASE_URL=https://api.openai.com/v1
export CW_OPENAI_TIMEOUT=30
export CW_OPENAI_MAX_RETRIES=3
export CW_OPENAI_BATCH_SIZE=100
```

**Available Models:**
- `text-embedding-3-small` - Fast and cost-effective (1536 dimensions)
- `text-embedding-3-large` - Higher quality (3072 dimensions)
- `text-embedding-ada-002` - Legacy model (1536 dimensions)

**Configuration Options:**
```toml
[embedding.openai]
model = "text-embedding-3-small"
base_url = "https://api.openai.com/v1"
timeout = 30
max_retries = 3
batch_size = 100
dimensions = 1536  # Optional for 3rd generation models
```

### Cohere

```bash
# Basic configuration
export CW_EMBEDDING_PROVIDER=cohere
export CW_EMBEDDING_API_KEY=your-cohere-key
export CW_EMBEDDING_MODEL=embed-english-v3.0

# Advanced settings
export CW_COHERE_BASE_URL=https://api.cohere.ai/v1
export CW_COHERE_TIMEOUT=30
export CW_COHERE_BATCH_SIZE=96
export CW_COHERE_INPUT_TYPE=search_document
```

**Available Models:**
- `embed-english-v3.0` - Latest English model
- `embed-multilingual-v3.0` - Multilingual support
- `embed-english-light-v3.0` - Lightweight version

**Configuration Options:**
```toml
[embedding.cohere]
model = "embed-english-v3.0"
base_url = "https://api.cohere.ai/v1"
timeout = 30
batch_size = 96
input_type = "search_document"
truncate = "END"
```

### HuggingFace

```bash
# Basic configuration
export CW_EMBEDDING_PROVIDER=huggingface
export CW_EMBEDDING_API_KEY=hf_your-huggingface-key
export CW_HF_MODEL_NAME=microsoft/codebert-base

# Self-hosted/local settings
export CW_HF_MODEL_PATH=/path/to/local/model
export CW_HF_DEVICE=cuda
export CW_HF_BATCH_SIZE=32
export CW_HF_MAX_LENGTH=512
```

**Popular Models:**
- `microsoft/codebert-base` - Code-specific BERT model
- `sentence-transformers/all-MiniLM-L6-v2` - General purpose, fast
- `sentence-transformers/all-mpnet-base-v2` - High quality general purpose

**Configuration Options:**
```toml
[embedding.huggingface]
model_name = "microsoft/codebert-base"
model_path = "/path/to/local/model"  # Optional for local models
device = "cuda"  # "cpu", "cuda", "mps"
batch_size = 32
max_length = 512
trust_remote_code = false
```

## Vector Database Configuration

### Qdrant

```bash
# Basic configuration
export CW_VECTOR_BACKEND_TYPE=qdrant
export CW_VECTOR_BACKEND_URL=http://localhost:6333
export CW_VECTOR_BACKEND_API_KEY=your-api-key  # Optional for local

# Advanced settings
export CW_QDRANT_COLLECTION_CONFIG='{"vectors": {"size": 1024, "distance": "Cosine"}}'
export CW_QDRANT_SHARD_NUMBER=1
export CW_QDRANT_REPLICATION_FACTOR=1
export CW_QDRANT_WRITE_CONSISTENCY_FACTOR=1
export CW_QDRANT_TIMEOUT=30
export CW_QDRANT_PREFER_GRPC=true
```

**Configuration Options:**
```toml
[vector_backend.qdrant]
url = "http://localhost:6333"
api_key = "your-api-key"
collection = "codeweaver"
timeout = 30
prefer_grpc = true
grpc_port = 6334

[vector_backend.qdrant.collection_config]
vectors.size = 1024
vectors.distance = "Cosine"
shard_number = 1
replication_factor = 1
write_consistency_factor = 1

[vector_backend.qdrant.hnsw_config]
m = 16
ef_construct = 200
full_scan_threshold = 10000
```

**Cloud Configuration:**
```bash
# Qdrant Cloud
export CW_VECTOR_BACKEND_URL=https://your-cluster.qdrant.cloud:6333
export CW_VECTOR_BACKEND_API_KEY=your-cloud-api-key
export CW_QDRANT_PREFER_GRPC=true
export CW_QDRANT_COMPRESSION=true
```

### Pinecone

```bash
# Basic configuration
export CW_VECTOR_BACKEND_TYPE=pinecone
export CW_VECTOR_BACKEND_API_KEY=your-pinecone-api-key
export CW_PINECONE_ENVIRONMENT=us-west1-gcp
export CW_PINECONE_INDEX_NAME=codeweaver

# Advanced settings
export CW_PINECONE_DIMENSION=1024
export CW_PINECONE_METRIC=cosine
export CW_PINECONE_PODS=1
export CW_PINECONE_REPLICAS=1
export CW_PINECONE_SHARDS=1
export CW_PINECONE_POD_TYPE=p1.x1
```

**Configuration Options:**
```toml
[vector_backend.pinecone]
api_key = "your-pinecone-api-key"
environment = "us-west1-gcp"
index_name = "codeweaver"
dimension = 1024
metric = "cosine"
pods = 1
replicas = 1
shards = 1
pod_type = "p1.x1"
```

### Weaviate

```bash
# Basic configuration
export CW_VECTOR_BACKEND_TYPE=weaviate
export CW_VECTOR_BACKEND_URL=http://localhost:8080
export CW_WEAVIATE_API_KEY=your-api-key  # Optional

# Advanced settings
export CW_WEAVIATE_CLASS_NAME=CodeChunk
export CW_WEAVIATE_TIMEOUT=30
export CW_WEAVIATE_BATCH_SIZE=100
export CW_WEAVIATE_DISTANCE_METRIC=cosine
```

**Configuration Options:**
```toml
[vector_backend.weaviate]
url = "http://localhost:8080"
api_key = "your-api-key"
class_name = "CodeChunk"
timeout = 30
batch_size = 100
distance_metric = "cosine"

[vector_backend.weaviate.schema]
vectorizer = "none"
vector_index_type = "hnsw"
vector_index_config.ef = 200
vector_index_config.max_connections = 64
```

### ChromaDB

```bash
# Basic configuration
export CW_VECTOR_BACKEND_TYPE=chroma
export CW_VECTOR_BACKEND_URL=http://localhost:8000
export CW_CHROMA_COLLECTION_NAME=codeweaver

# Advanced settings
export CW_CHROMA_DISTANCE_METRIC=cosine
export CW_CHROMA_BATCH_SIZE=100
export CW_CHROMA_EMBEDDING_FUNCTION=default
```

**Configuration Options:**
```toml
[vector_backend.chroma]
url = "http://localhost:8000"
collection_name = "codeweaver"
distance_metric = "cosine"
batch_size = 100
embedding_function = "default"
```

## Content Processing Configuration

### Chunking Settings

```bash
# Chunk size configuration
export CW_CHUNK_SIZE=1500              # Maximum chunk size in characters
export CW_MIN_CHUNK_SIZE=50            # Minimum chunk size
export CW_CHUNK_OVERLAP=100            # Overlap between chunks
export CW_CHUNK_SEPARATOR="\n\n"       # Preferred split points

# Advanced chunking
export CW_ENABLE_AST_CHUNKING=true     # Use ast-grep for intelligent chunking
export CW_AST_RESPECT_BOUNDARIES=true  # Respect function/class boundaries
export CW_FALLBACK_CHUNKING=true       # Fall back if ast-grep fails
```

**Configuration Options:**
```toml
[chunking]
max_size = 1500
min_size = 50
overlap = 100
separator = "\n\n"
enable_ast_chunking = true
ast_respect_boundaries = true
fallback_chunking = true

[chunking.ast_patterns]
python = ["function_definition", "class_definition"]
javascript = ["function_declaration", "class_declaration"]
rust = ["function_item", "impl_item"]
```

### File Filtering

```bash
# File type filtering
export CW_INCLUDE_LANGUAGES=python,javascript,typescript,rust,go
export CW_EXCLUDE_PATTERNS="*.min.js,*.bundle.js,*.d.ts"
export CW_MAX_FILE_SIZE=1048576        # 1MB in bytes
export CW_MIN_FILE_SIZE=10             # Minimum file size

# Directory filtering
export CW_INCLUDE_DIRS=src,lib,app,components
export CW_EXCLUDE_DIRS=node_modules,target,build,dist,.git,.venv
export CW_FOLLOW_SYMLINKS=false
export CW_RESPECT_GITIGNORE=true
```

**Configuration Options:**
```toml
[filtering]
include_languages = ["python", "javascript", "typescript", "rust", "go"]
exclude_patterns = ["*.min.js", "*.bundle.js", "*.d.ts"]
max_file_size = 1048576  # 1MB
min_file_size = 10
include_dirs = ["src", "lib", "app", "components"]
exclude_dirs = ["node_modules", "target", "build", "dist", ".git", ".venv"]
follow_symlinks = false
respect_gitignore = true

[filtering.language_extensions]
python = [".py", ".pyw"]
javascript = [".js", ".jsx", ".mjs"]
typescript = [".ts", ".tsx"]
rust = [".rs"]
go = [".go"]
```

## Performance Configuration

### Indexing Performance

```bash
# Parallel processing
export CW_PARALLEL_PROCESSING=true
export CW_INDEXING_WORKERS=4           # Number of worker processes
export CW_MAX_CONCURRENT_CHUNKS=4      # Concurrent chunk processing
export CW_BATCH_SIZE=16                # Batch size for API calls

# Memory management
export CW_MEMORY_LIMIT_MB=2048         # Memory limit in MB
export CW_CHUNK_CACHE_SIZE=1000        # Number of chunks to cache
export CW_ENABLE_CACHING=true          # Enable result caching
```

**Configuration Options:**
```toml
[performance]
parallel_processing = true
indexing_workers = 4
max_concurrent_chunks = 4
batch_size = 16
memory_limit_mb = 2048
chunk_cache_size = 1000
enable_caching = true

[performance.timeouts]
request_timeout = 30
chunk_timeout = 10
indexing_timeout = 3600
search_timeout = 30
```

### Search Performance

```bash
# Search optimization
export CW_SEARCH_TOP_K=20              # Initial candidate count
export CW_SEARCH_SCORE_THRESHOLD=0.7   # Minimum similarity score
export CW_RERANK_TOP_N=10              # Final result count
export CW_ENABLE_QUERY_CACHE=true      # Cache query results
export CW_QUERY_CACHE_SIZE=1000        # Query cache size
export CW_QUERY_CACHE_TTL=3600         # Cache TTL in seconds
```

**Configuration Options:**
```toml
[search]
top_k = 20
score_threshold = 0.7
rerank_top_n = 10
enable_query_cache = true
query_cache_size = 1000
query_cache_ttl = 3600

[search.hybrid]
enable_hybrid_search = true
semantic_weight = 0.7
keyword_weight = 0.3
enable_fuzzy_matching = true
max_edit_distance = 2
```

## Advanced Configuration

### Multi-Provider Setup

```toml
# Multiple embedding providers
[embedding]
primary = "voyage"
fallback = ["openai", "cohere"]

[embedding.voyage]
api_key = "pa-your-voyage-key"
model = "voyage-code-2"

[embedding.openai]
api_key = "sk-your-openai-key"
model = "text-embedding-3-small"

[embedding.cohere]
api_key = "your-cohere-key"
model = "embed-english-v3.0"
```

### Multi-Collection Setup

```bash
# Different collections for different projects
export CW_COLLECTIONS_CONFIG='[
  {"name": "frontend", "languages": ["javascript", "typescript"], "path": "src/frontend"},
  {"name": "backend", "languages": ["python", "go"], "path": "src/backend"},
  {"name": "docs", "languages": ["markdown"], "path": "docs"}
]'
```

### Security Configuration

```bash
# Security settings
export CW_ENABLE_AUTH=true
export CW_AUTH_TOKEN=your-secret-token
export CW_ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com
export CW_RATE_LIMIT_REQUESTS=100
export CW_RATE_LIMIT_WINDOW=3600
export CW_ENABLE_HTTPS=true
export CW_SSL_CERT_PATH=/path/to/cert.pem
export CW_SSL_KEY_PATH=/path/to/key.pem
```

**Configuration Options:**
```toml
[security]
enable_auth = true
auth_token = "your-secret-token"
allowed_origins = ["http://localhost:3000", "https://yourdomain.com"]
rate_limit_requests = 100
rate_limit_window = 3600
enable_https = true
ssl_cert_path = "/path/to/cert.pem"
ssl_key_path = "/path/to/key.pem"

[security.api_keys]
embedding_key_rotation = 86400  # 24 hours
vector_db_key_rotation = 604800  # 7 days
```

### Monitoring Configuration

```bash
# Monitoring and metrics
export CW_ENABLE_METRICS=true
export CW_METRICS_PORT=8001
export CW_PROMETHEUS_ENABLED=true
export CW_HEALTH_CHECK_INTERVAL=30
export CW_LOG_FORMAT=json
export CW_LOG_FILE=/var/log/codeweaver/app.log
export CW_LOG_ROTATION=daily
export CW_LOG_RETENTION_DAYS=7
```

**Configuration Options:**
```toml
[monitoring]
enable_metrics = true
metrics_port = 8001
prometheus_enabled = true
health_check_interval = 30

[monitoring.logging]
format = "json"
file = "/var/log/codeweaver/app.log"
rotation = "daily"
retention_days = 7
level = "INFO"

[monitoring.alerts]
enable_alerts = true
webhook_url = "https://hooks.slack.com/your-webhook"
alert_on_errors = true
alert_on_performance = true
performance_threshold_ms = 1000
```

## Environment-Specific Configurations

### Development Environment

```bash
# Development settings
export CW_LOG_LEVEL=DEBUG
export CW_ENABLE_HOT_RELOAD=true
export CW_DEVELOPMENT_MODE=true
export CW_MOCK_APIS=false
export CW_VECTOR_BACKEND_URL=http://localhost:6333
export CW_CHUNK_SIZE=800  # Smaller chunks for faster development
```

### Staging Environment

```bash
# Staging settings
export CW_LOG_LEVEL=INFO
export CW_ENABLE_METRICS=true
export CW_RATE_LIMIT_REQUESTS=500
export CW_VECTOR_BACKEND_URL=https://staging-qdrant.yourcompany.com:6333
export CW_CHUNK_SIZE=1200
export CW_BATCH_SIZE=12
```

### Production Environment

```bash
# Production settings
export CW_LOG_LEVEL=WARNING
export CW_ENABLE_METRICS=true
export CW_ENABLE_AUTH=true
export CW_RATE_LIMIT_REQUESTS=1000
export CW_VECTOR_BACKEND_URL=https://prod-qdrant.yourcompany.com:6333
export CW_CHUNK_SIZE=1500
export CW_BATCH_SIZE=16
export CW_PARALLEL_PROCESSING=true
export CW_INDEXING_WORKERS=8
export CW_MEMORY_LIMIT_MB=4096
```

## Configuration Validation

### Validation Tools

```bash
# Validate configuration
codeweaver config validate

# Show current configuration
codeweaver config show

# Test connectivity
codeweaver config test
```

### Configuration Schema

```python
# Python validation
from codeweaver.config import validate_config

config = {
    "embedding": {"provider": "voyage", "api_key": "pa-key"},
    "vector_backend": {"type": "qdrant", "url": "http://localhost:6333"}
}

errors = validate_config(config)
if errors:
    print("Configuration errors:", errors)
```

## Configuration Templates

### Small Project Template

```toml
# codeweaver-small.toml
[embedding]
provider = "voyage"
model = "voyage-code-2"

[vector_backend]
type = "qdrant"
url = "http://localhost:6333"

[chunking]
max_size = 1000
min_size = 50
overlap = 50

[performance]
batch_size = 8
indexing_workers = 2
parallel_processing = false
```

### Large Project Template

```toml
# codeweaver-large.toml
[embedding]
provider = "voyage"
model = "voyage-code-2"

[vector_backend]
type = "qdrant"
url = "https://your-cluster.qdrant.cloud:6333"

[chunking]
max_size = 1500
min_size = 100
overlap = 100

[performance]
batch_size = 24
indexing_workers = 8
parallel_processing = true
memory_limit_mb = 4096

[search]
top_k = 50
score_threshold = 0.6
enable_query_cache = true
```

### Team/Enterprise Template

```toml
# codeweaver-enterprise.toml
[embedding]
provider = "voyage"
model = "voyage-code-2"

[vector_backend]
type = "qdrant"
url = "https://enterprise-qdrant.company.com:6333"

[security]
enable_auth = true
allowed_origins = ["https://ide.company.com", "https://docs.company.com"]
rate_limit_requests = 2000

[monitoring]
enable_metrics = true
prometheus_enabled = true
health_check_interval = 15

[performance]
batch_size = 32
indexing_workers = 16
parallel_processing = true
memory_limit_mb = 8192
```

## Configuration Best Practices

### Security Best Practices

1. **Never commit API keys** to version control
2. **Use environment variables** for sensitive data
3. **Rotate API keys** regularly
4. **Enable authentication** in production
5. **Use HTTPS** for all external connections
6. **Implement rate limiting** to prevent abuse
7. **Monitor access logs** for suspicious activity

### Performance Best Practices

1. **Start with recommended settings** and tune based on usage
2. **Monitor resource usage** during indexing and search
3. **Use caching** for frequently accessed data
4. **Enable parallel processing** for large codebases
5. **Optimize chunk sizes** for your content type
6. **Use local vector databases** for development
7. **Scale horizontally** for high-load scenarios

### Maintenance Best Practices

1. **Document your configuration** choices
2. **Version control** configuration files
3. **Test configuration changes** in staging first
4. **Monitor performance metrics** after changes
5. **Keep backups** of vector databases
6. **Regular health checks** for all components
7. **Update dependencies** regularly

## Next Steps

With your configuration complete:

- [**Troubleshooting Guide**](troubleshooting.md) - Resolve configuration issues
- [**Claude Desktop Integration**](../user-guide/claude-desktop.md) - Connect with your AI assistant
- [**Performance Optimization**](../user-guide/performance.md) - Fine-tune for your use case
- [**Development Workflows**](../user-guide/workflows.md) - Learn practical usage patterns