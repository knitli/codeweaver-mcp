# CodeWeaver Configuration Migration Guide

This guide helps you migrate from the legacy configuration format (v1.x) to the new extensible format (v2.0).

## Overview

CodeWeaver v2.0 introduces a new configuration schema that supports:

- **15+ Vector Database Backends**: Qdrant, Pinecone, Chroma, Weaviate, pgvector, Milvus, Elasticsearch, and more
- **5+ Embedding Providers**: VoyageAI, OpenAI, Cohere, SentenceTransformers, HuggingFace
- **Multiple Data Sources**: File system, Git repositories, APIs, databases, web crawlers
- **Hybrid Search**: Dense + sparse vector search for improved relevance
- **Enhanced Performance**: Connection pooling, caching, and batch processing

## Migration Strategies

### 1. Automatic Migration (Recommended)

The new system automatically detects and migrates legacy configurations:

```bash
# Your existing configuration will be automatically migrated
# No action required - legacy configs continue to work!
```

### 2. Manual Migration

Use the configuration CLI tool to migrate explicitly:

```bash
# Install migration dependencies
pip install tomlkit

# Migrate configuration file
python -m codeweaver.config_cli migrate .code-weaver.toml -o config-v2.toml

# Validate migrated configuration
python -m codeweaver.config_cli validate config-v2.toml
```

### 3. Side-by-Side Comparison

Create both formats to understand the differences:

```bash
# Generate examples
python -m codeweaver.config_cli generate migration_guide -o migration-example.toml
```

## Configuration Mapping

### Legacy to New Format Mapping

| Legacy Section | New Section | Notes |
|---------------|-------------|--------|
| `[embedding]` | `[provider]` | Enhanced with more providers |
| `[qdrant]` | `[backend]` | Supports multiple backends |
| `[indexing]` | `[data_sources]` | Multi-source support added |

### Detailed Mapping

#### Embedding Configuration

**Legacy Format:**
```toml
[embedding]
provider = "voyage"
api_key = "your-key"
model = "voyage-code-3"
dimension = 1024
rerank_provider = "voyage"
```

**New Format:**
```toml
[provider]
embedding_provider = "voyage"
embedding_api_key = "your-key"
embedding_model = "voyage-code-3"
embedding_dimension = 1024
rerank_provider = "voyage"
rerank_model = "voyage-rerank-2"
```

#### Backend Configuration

**Legacy Format:**
```toml
[qdrant]
url = "https://your-cluster.qdrant.io"
api_key = "your-key"
collection_name = "code-embeddings"
enable_sparse_vectors = true
```

**New Format:**
```toml
[backend]
provider = "qdrant"
url = "https://your-cluster.qdrant.io"
api_key = "your-key"
collection_name = "code-embeddings"
enable_hybrid_search = true
enable_sparse_vectors = true
```

#### Data Sources Configuration

**Legacy Format:**
```toml
[indexing]
use_gitignore = true
additional_ignore_patterns = ["node_modules", ".git"]
batch_size = 8
```

**New Format:**
```toml
[data_sources]
enabled = true

[[data_sources.sources]]
type = "filesystem"
enabled = true
priority = 1

[data_sources.sources.config]
root_path = "."
use_gitignore = true
additional_ignore_patterns = ["node_modules", ".git"]
batch_size = 8
```

## Environment Variables

### Legacy Environment Variables (Still Supported)

```bash
# Legacy variables continue to work
VOYAGE_API_KEY=your-voyage-key
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-key
COLLECTION_NAME=code-embeddings
```

### New Environment Variables (Recommended)

```bash
# New backend-agnostic variables
VECTOR_BACKEND_PROVIDER=qdrant
VECTOR_BACKEND_URL=https://your-cluster.qdrant.io
VECTOR_BACKEND_API_KEY=your-qdrant-key

# New provider-agnostic variables
EMBEDDING_PROVIDER=voyage
EMBEDDING_API_KEY=your-voyage-key
EMBEDDING_MODEL=voyage-code-3

# Feature flags
ENABLE_HYBRID_SEARCH=true
ENABLE_SPARSE_VECTORS=true
USE_LOCAL_MODELS=false
```

## Step-by-Step Migration

### Step 1: Backup Current Configuration

```bash
# Backup your current configuration
cp .code-weaver.toml .code-weaver.toml.backup
```

### Step 2: Test Automatic Migration

```bash
# Test with your current configuration
# The system will automatically migrate and log the process
LOG_LEVEL=DEBUG python -m codeweaver.main
```

### Step 3: Generate New Configuration

```bash
# Generate a new configuration based on your deployment type
python -m codeweaver.config_cli generate production_cloud -o config-v2.toml

# Or for local development
python -m codeweaver.config_cli generate local_development -o config-v2.toml
```

### Step 4: Validate and Compare

```bash
# Validate new configuration
python -m codeweaver.config_cli validate config-v2.toml

# Check compatibility
python -m codeweaver.config_cli check-compatibility qdrant voyage --enable-hybrid
```

### Step 5: Deploy New Configuration

```bash
# Replace old configuration
mv config-v2.toml .code-weaver.toml

# Test the new configuration
python -m codeweaver.main
```

## New Features Available After Migration

### 1. Multiple Backend Support

```toml
# Switch to Pinecone
[backend]
provider = "pinecone"
api_key = "your-pinecone-key"

[backend.provider_options]
environment = "us-west1-gcp"
```

### 2. Hybrid Search

```toml
[backend]
enable_hybrid_search = true
enable_sparse_vectors = true

# Configure fusion strategy
[backend.provider_options]
fusion_strategy = "rrf"  # or "dbsf"
alpha = 0.7  # Dense/sparse balance
```

### 3. Multiple Data Sources

```toml
# Add Git repository source
[[data_sources.sources]]
type = "git"
enabled = true
priority = 2

[data_sources.sources.config]
repository_url = "https://github.com/example/repo.git"
branch = "main"
```

### 4. Local Models

```toml
[provider]
embedding_provider = "sentence-transformers"
embedding_model = "all-MiniLM-L6-v2"
use_local = true
device = "cuda"  # or "cpu", "mps"
```

## Troubleshooting

### Common Migration Issues

#### 1. API Key Not Found

**Error:** `VOYAGE_API_KEY is required when using voyage embeddings`

**Solution:**
```bash
# Set the API key
export VOYAGE_API_KEY=your-key

# Or use new format
export EMBEDDING_API_KEY=your-key
```

#### 2. Backend URL Missing

**Error:** `QDRANT_URL or VECTOR_BACKEND_URL is required`

**Solution:**
```bash
# Legacy format
export QDRANT_URL=https://your-cluster.qdrant.io

# New format
export VECTOR_BACKEND_URL=https://your-cluster.qdrant.io
```

#### 3. Mixed Configuration Warning

**Warning:** `Mixed legacy and new environment variables detected`

**Solution:**
```bash
# Choose one format consistently
# Legacy:
export VOYAGE_API_KEY=key
export QDRANT_URL=url

# Or new:
export EMBEDDING_API_KEY=key
export VECTOR_BACKEND_URL=url
```

### Validation Commands

```bash
# Check environment variables
python -c "from codeweaver.config import validate_environment_variables; print(validate_environment_variables())"

# Get configuration summary
python -c "from codeweaver.config import get_effective_config_summary; print(get_effective_config_summary())"

# Get improvement suggestions
python -c "from codeweaver.config import suggest_configuration_improvements; print(suggest_configuration_improvements())"
```

## Performance Optimizations

After migration, consider these optimizations:

### 1. Enable Hybrid Search

```toml
[backend]
enable_hybrid_search = true
enable_sparse_vectors = true
```

### 2. Increase Batch Sizes

```toml
[backend]
batch_size = 100  # Up from default 50

[provider]
embedding_batch_size = 16  # Up from default 8
```

### 3. Enable Caching

```toml
[provider]
enable_caching = true
cache_ttl_seconds = 3600
```

### 4. Connection Pooling

```toml
[backend]
max_connections = 20
connection_timeout = 30.0
request_timeout = 60.0
```

## Rollback Plan

If you need to rollback to the legacy format:

```bash
# Restore backup
cp .code-weaver.toml.backup .code-weaver.toml

# Or generate legacy format
python -m codeweaver.config_cli generate legacy_format -o .code-weaver.toml
```

## Support

For migration assistance:

1. **Validation Issues**: Use the configuration CLI tool
2. **Performance Problems**: Review the performance optimization section
3. **Backend Compatibility**: Check the compatibility matrix in the CLI info command

```bash
# Get comprehensive system information
python -m codeweaver.config_cli info
```

## Next Steps

After successful migration:

1. **Explore New Backends**: Try Pinecone, Weaviate, or other supported backends
2. **Add Data Sources**: Integrate Git repositories, APIs, or databases
3. **Enable Hybrid Search**: Improve search quality with sparse vectors
4. **Optimize Performance**: Tune batch sizes and connection settings
5. **Monitor Usage**: Enable metrics and logging for production deployments