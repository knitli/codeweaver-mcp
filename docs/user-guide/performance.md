<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Performance Optimization

**Configuration tuning and optimization strategies for production use**

This guide covers performance optimization techniques for CodeWeaver deployments, from small projects to enterprise-scale codebases, focusing on indexing speed, search performance, and resource efficiency.

## Performance Overview

### Key Performance Factors

**Indexing Performance:**
- File discovery and filtering
- Code chunking and processing
- Embedding generation
- Vector storage operations

**Search Performance:**
- Query processing and embedding
- Vector similarity search
- Result ranking and filtering
- Response formatting

**Resource Efficiency:**
- Memory usage during indexing
- Storage requirements for vectors
- Network bandwidth for API calls
- CPU utilization for processing

## Baseline Performance Characteristics

### Project Size Guidelines

| Project Size | Files | Indexing Time | Memory Usage | Storage Overhead |
|--------------|--------|---------------|--------------|------------------|
| **Small** | <1,000 | 1-3 minutes | ~50MB | 2-3x source size |
| **Medium** | 1,000-10,000 | 5-15 minutes | ~200MB | 3-4x source size |
| **Large** | 10,000-50,000 | 15-60 minutes | ~1GB | 4-5x source size |
| **Enterprise** | 50,000+ | 1+ hours | ~2GB+ | 5-6x source size |

### Search Performance Targets

| Operation | Target Response Time | Factors |
|-----------|---------------------|---------|
| **Semantic Search** | 200-500ms | Vector database performance, embedding generation |
| **Pattern Search** | 50-200ms | ast-grep performance, result filtering |
| **Hybrid Search** | 300-700ms | Combined semantic + pattern matching |
| **Language Detection** | <50ms | Cached results, file extension mapping |

## Configuration Optimization

### Environment Variables for Performance

#### Core Performance Settings
```bash
# Chunking Performance
export CW_CHUNK_SIZE=1200              # Default: 1500, Range: 50-2000
export CW_MIN_CHUNK_SIZE=100           # Default: 50, Range: 25-200
export CW_CHUNK_OVERLAP=50             # Default: 100, Range: 0-200

# Batch Processing
export CW_BATCH_SIZE=16                # Default: 8, Range: 1-32
export CW_MAX_CONCURRENT_CHUNKS=4      # Default: 2, Range: 1-8
export CW_INDEXING_WORKERS=4           # Default: 2, Range: 1-16

# File Processing
export CW_MAX_FILE_SIZE=2097152        # Default: 1MB, Range: 512KB-10MB
export CW_PARALLEL_PROCESSING=true     # Default: false
export CW_ENABLE_CACHING=true          # Default: true
```

#### Resource Management
```bash
# Memory Management
export CW_MEMORY_LIMIT_MB=2048         # Default: 1024, Range: 512-8192
export CW_CHUNK_CACHE_SIZE=1000        # Default: 500, Range: 100-5000
export CW_VECTOR_CACHE_SIZE=5000       # Default: 1000, Range: 500-20000

# Network Optimization
export CW_REQUEST_TIMEOUT=30           # Default: 20, Range: 10-120
export CW_MAX_RETRIES=3                # Default: 2, Range: 1-5
export CW_RETRY_DELAY=1.5              # Default: 1.0, Range: 0.5-5.0
```

### Project-Specific Configurations

#### Small to Medium Projects (<10k files)
```bash
# Optimized for speed and simplicity
export CW_CHUNK_SIZE=1000
export CW_BATCH_SIZE=12
export CW_MAX_CONCURRENT_CHUNKS=2
export CW_PARALLEL_PROCESSING=true
export CW_MEMORY_LIMIT_MB=512
```

#### Large Projects (10k-50k files)
```bash
# Balanced performance and resource usage
export CW_CHUNK_SIZE=1200
export CW_BATCH_SIZE=16
export CW_MAX_CONCURRENT_CHUNKS=4
export CW_INDEXING_WORKERS=4
export CW_PARALLEL_PROCESSING=true
export CW_MEMORY_LIMIT_MB=2048
export CW_CHUNK_CACHE_SIZE=2000
```

#### Enterprise Projects (50k+ files)
```bash
# Maximum performance with high resource usage
export CW_CHUNK_SIZE=1500
export CW_BATCH_SIZE=24
export CW_MAX_CONCURRENT_CHUNKS=6
export CW_INDEXING_WORKERS=8
export CW_PARALLEL_PROCESSING=true
export CW_MEMORY_LIMIT_MB=4096
export CW_CHUNK_CACHE_SIZE=5000
export CW_VECTOR_CACHE_SIZE=10000
```

## Provider-Specific Optimization

### Embedding Provider Performance

#### Voyage AI (Recommended)
```bash
export CW_EMBEDDING_PROVIDER=voyage
export CW_EMBEDDING_MODEL=voyage-code-2    # Best for code
export CW_EMBEDDING_BATCH_SIZE=16          # Optimal batch size
export CW_REQUEST_TIMEOUT=30               # Higher timeout for large batches
```

**Performance Characteristics:**
- **Best for code understanding**
- **Rate limit: 1M tokens/minute**
- **Batch size: Up to 128 texts**
- **Latency: 200-500ms per batch**

#### OpenAI
```bash
export CW_EMBEDDING_PROVIDER=openai
export CW_EMBEDDING_MODEL=text-embedding-3-small  # Faster, smaller
# export CW_EMBEDDING_MODEL=text-embedding-3-large # Better quality, slower
export CW_EMBEDDING_BATCH_SIZE=100         # Higher batch size supported
export CW_REQUEST_TIMEOUT=20
```

**Performance Characteristics:**
- **High rate limits**
- **Large batch support (up to 2048 texts)**
- **Latency: 100-300ms per batch**
- **Choose small vs large model based on needs**

#### Cohere
```bash
export CW_EMBEDDING_PROVIDER=cohere
export CW_EMBEDDING_MODEL=embed-english-v3.0
export CW_EMBEDDING_BATCH_SIZE=96          # Good batch support
export CW_REQUEST_TIMEOUT=25
```

#### HuggingFace (Local/Self-Hosted)
```bash
export CW_EMBEDDING_PROVIDER=huggingface
export CW_HF_MODEL_NAME=microsoft/codebert-base
export CW_HF_DEVICE=cuda                   # Use GPU if available
export CW_EMBEDDING_BATCH_SIZE=32          # Limited by local GPU memory
```

### Vector Database Performance

#### Qdrant (Recommended)
```bash
export CW_VECTOR_BACKEND_TYPE=qdrant
export CW_VECTOR_DIMENSION=1024            # Match embedding model
export CW_VECTOR_DISTANCE_METRIC=cosine    # Best for semantic similarity
export CW_QDRANT_PARALLEL_INDEXING=true    # Enable parallel indexing
export CW_QDRANT_SHARD_NUMBER=1            # Single shard for <1M vectors
```

**Local Qdrant Optimization:**
```bash
# Docker configuration for performance
docker run -p 6333:6333 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  -e QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=32 \
  -e QDRANT__SERVICE__GRPC_PORT=6334 \
  qdrant/qdrant
```

**Qdrant Cloud Configuration:**
```bash
export CW_VECTOR_BACKEND_URL=https://your-cluster.qdrant.cloud:6333
export CW_QDRANT_PREFER_GRPC=true          # Better performance
export CW_QDRANT_COMPRESSION=true          # Reduce bandwidth
```

#### Pinecone
```bash
export CW_VECTOR_BACKEND_TYPE=pinecone
export CW_PINECONE_ENVIRONMENT=us-west1-gcp
export CW_PINECONE_INDEX_TYPE=pod          # Better for large datasets
export CW_PINECONE_REPLICAS=1              # Start with 1, scale as needed
export CW_PINECONE_SHARDS=1
```

#### Weaviate
```bash
export CW_VECTOR_BACKEND_TYPE=weaviate
export CW_WEAVIATE_SCHEME=http
export CW_WEAVIATE_TIMEOUT=30
export CW_WEAVIATE_BATCH_SIZE=100          # Good batch performance
```

## Indexing Optimization Strategies

### Smart File Filtering

#### Built-in Optimizations
CodeWeaver automatically filters:
- Binary files and common build artifacts
- Large files (>1MB by default)
- Hidden files and directories
- Common cache and dependency directories

#### Custom Filtering Configuration
```bash
# File size limits
export CW_MAX_FILE_SIZE=2097152      # 2MB limit for large codebases
export CW_MIN_FILE_SIZE=10           # Skip very small files

# Language-specific filtering
export CW_INCLUDE_LANGUAGES=python,javascript,typescript,rust,go
export CW_EXCLUDE_PATTERNS="*.min.js,*.bundle.js,*.d.ts"

# Directory filtering
export CW_EXCLUDE_DIRS=node_modules,target,build,dist,.git,.venv
export CW_INCLUDE_DIRS=src,lib,app,components
```

### Incremental Indexing

#### Strategy 1: File Modification Tracking
```bash
# Enable file modification tracking
export CW_ENABLE_INCREMENTAL=true
export CW_MODIFICATION_CHECK=true
export CW_CACHE_METADATA=true
```

#### Strategy 2: Git-Based Incremental Updates
```python
# Example: Index only changed files
# This would be implemented as a custom script
import subprocess
import os

def get_changed_files(since_commit="HEAD~1"):
    """Get list of files changed since last commit"""
    result = subprocess.run(
        ["git", "diff", "--name-only", since_commit],
        capture_output=True, text=True
    )
    return result.stdout.strip().split('\n')

def selective_indexing(changed_files):
    """Index only changed files"""
    for file_path in changed_files:
        if os.path.exists(file_path):
            # Call CodeWeaver indexing for specific file
            subprocess.run(["codeweaver", "index-file", file_path])
```

### Parallel Processing Optimization

#### Multi-Core CPU Utilization
```bash
# Optimize for multi-core systems
export CW_INDEXING_WORKERS=$(nproc)      # Use all CPU cores
export CW_MAX_CONCURRENT_CHUNKS=4        # Limit concurrent chunks
export CW_BATCH_SIZE=16                  # Optimal batch size
export CW_PARALLEL_PROCESSING=true
```

#### Memory vs Speed Trade-offs
```bash
# High memory, maximum speed
export CW_MEMORY_LIMIT_MB=4096
export CW_CHUNK_CACHE_SIZE=5000
export CW_PRELOAD_EMBEDDINGS=true

# Low memory, moderate speed
export CW_MEMORY_LIMIT_MB=512
export CW_CHUNK_CACHE_SIZE=500
export CW_STREAM_PROCESSING=true
```

## Search Performance Optimization

### Query Optimization

#### Semantic Search Tuning
```bash
# Vector search parameters
export CW_SEARCH_TOP_K=20              # Number of initial candidates
export CW_SEARCH_SCORE_THRESHOLD=0.7   # Minimum similarity score
export CW_RERANK_TOP_N=10              # Final result count
export CW_ENABLE_QUERY_EXPANSION=true  # Improve recall
```

#### Hybrid Search Configuration
```bash
# Balance semantic and keyword search
export CW_HYBRID_SEARCH_WEIGHTS=0.7,0.3  # 70% semantic, 30% keyword
export CW_ENABLE_FUZZY_MATCHING=true      # Handle typos
export CW_MAX_EDIT_DISTANCE=2            # Fuzzy matching tolerance
```

### Caching Strategies

#### Query Result Caching
```bash
# Enable result caching
export CW_ENABLE_QUERY_CACHE=true
export CW_QUERY_CACHE_SIZE=1000        # Number of cached queries
export CW_QUERY_CACHE_TTL=3600         # 1 hour cache lifetime
```

#### Embedding Caching
```bash
# Cache embeddings for repeated queries
export CW_EMBEDDING_CACHE_SIZE=5000
export CW_EMBEDDING_CACHE_TTL=7200     # 2 hours
export CW_PERSISTENT_CACHE=true        # Survive restarts
```

### Response Optimization

#### Result Formatting
```bash
# Optimize response size and format
export CW_MAX_RESULT_LENGTH=500        # Truncate long results
export CW_INCLUDE_METADATA=false       # Reduce response size
export CW_COMPRESS_RESPONSES=true      # Enable compression
```

#### Streaming Responses
```bash
# Enable streaming for large result sets
export CW_ENABLE_STREAMING=true
export CW_STREAM_CHUNK_SIZE=10         # Results per chunk
export CW_STREAM_DELAY=50              # Milliseconds between chunks
```

## Monitoring and Diagnostics

### Performance Metrics Collection

#### Built-in Metrics
```bash
# Enable performance monitoring
export CW_ENABLE_METRICS=true
export CW_METRICS_COLLECTION_INTERVAL=30  # Seconds
export CW_METRICS_RETENTION_DAYS=7
```

#### Custom Monitoring Integration
```python
# Example: Prometheus metrics integration
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
indexing_duration = Histogram('codeweaver_indexing_duration_seconds')
search_requests = Counter('codeweaver_search_requests_total')
vector_store_size = Gauge('codeweaver_vector_store_size_bytes')

# Use in CodeWeaver configuration
export CW_PROMETHEUS_ENABLED=true
export CW_PROMETHEUS_PORT=8000
```

### Performance Debugging

#### Debug Logging Configuration
```bash
# Enable detailed performance logging
export CW_LOG_LEVEL=DEBUG
export CW_LOG_PERFORMANCE=true
export CW_LOG_API_CALLS=true
export CW_LOG_TIMING=true
```

#### Profiling Tools
```bash
# Enable Python profiling
export CW_ENABLE_PROFILING=true
export CW_PROFILE_OUTPUT_DIR=/tmp/codeweaver-profiles
export CW_PROFILE_TOP_N=20
```

## Production Deployment Optimization

### Container Optimization

#### Docker Configuration
```dockerfile
# Optimized Dockerfile for CodeWeaver
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set optimal Python settings
ENV PYTHONOPTIMIZE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Configure CodeWeaver
ENV CW_PARALLEL_PROCESSING=true
ENV CW_MEMORY_LIMIT_MB=2048
ENV CW_ENABLE_CACHING=true

# Install CodeWeaver
RUN pip install --no-cache-dir codeweaver

EXPOSE 8000
CMD ["codeweaver"]
```

#### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: codeweaver
spec:
  replicas: 3
  selector:
    matchLabels:
      app: codeweaver
  template:
    metadata:
      labels:
        app: codeweaver
    spec:
      containers:
      - name: codeweaver
        image: codeweaver:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: CW_MEMORY_LIMIT_MB
          value: "3072"
        - name: CW_INDEXING_WORKERS
          value: "4"
        - name: CW_PARALLEL_PROCESSING
          value: "true"
```

### Load Balancing and Scaling

#### Horizontal Scaling Strategy
```bash
# Multiple CodeWeaver instances
# Instance 1: Read-heavy workload
export CW_INSTANCE_ROLE=search
export CW_ENABLE_INDEXING=false
export CW_SEARCH_CACHE_SIZE=10000

# Instance 2: Indexing workload
export CW_INSTANCE_ROLE=indexing
export CW_ENABLE_SEARCH=false
export CW_INDEXING_WORKERS=8
```

#### Database Connection Pooling
```bash
# Vector database connection optimization
export CW_CONNECTION_POOL_SIZE=20
export CW_CONNECTION_POOL_MAX_OVERFLOW=10
export CW_CONNECTION_TIMEOUT=30
export CW_CONNECTION_RETRY_ATTEMPTS=3
```

## Performance Troubleshooting

### Common Performance Issues

#### Slow Indexing
**Symptoms:** Indexing takes longer than expected

**Diagnostics:**
```bash
# Check file processing stats
export CW_LOG_LEVEL=DEBUG
export CW_LOG_TIMING=true

# Monitor resource usage
htop  # Check CPU usage
free -h  # Check memory usage
iotop  # Check I/O usage
```

**Solutions:**
1. Increase batch size and workers
2. Filter out unnecessary files
3. Optimize chunk size for your content
4. Use faster embedding provider
5. Enable parallel processing

#### High Memory Usage
**Symptoms:** CodeWeaver consumes excessive memory

**Solutions:**
```bash
# Reduce memory footprint
export CW_MEMORY_LIMIT_MB=1024
export CW_CHUNK_CACHE_SIZE=500
export CW_STREAM_PROCESSING=true
export CW_BATCH_SIZE=8
```

#### Slow Search Performance
**Symptoms:** Queries take >1 second to respond

**Diagnostics:**
```bash
# Profile search operations
export CW_LOG_PERFORMANCE=true
export CW_ENABLE_PROFILING=true
```

**Solutions:**
1. Enable query result caching
2. Optimize vector database configuration
3. Reduce search result count
4. Use result streaming for large responses
5. Configure hybrid search weights

### Performance Benchmarking

#### Indexing Benchmarks
```python
# Benchmark indexing performance
import time
import subprocess

def benchmark_indexing(codebase_path):
    start_time = time.time()
    result = subprocess.run([
        "codeweaver", "index", codebase_path
    ], capture_output=True)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"Indexing completed in {duration:.2f} seconds")
    return duration
```

#### Search Benchmarks
```python
# Benchmark search performance
def benchmark_search(query, iterations=10):
    import statistics
    times = []
    
    for _ in range(iterations):
        start_time = time.time()
        # Perform search operation
        result = search_codebase(query)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = statistics.mean(times)
    print(f"Average search time: {avg_time:.3f} seconds")
    return avg_time
```

## Next Steps

With optimized performance configuration:

- [**Troubleshooting Guide**](../getting-started/troubleshooting.md) - Resolve performance issues
- [**Configuration Reference**](../getting-started/configuration.md) - Complete configuration options
- [**Extension Development**](../extension-dev/) - Build performance-optimized extensions
- [**Production Deployment**](../enterprise/) - Enterprise-scale deployment patterns