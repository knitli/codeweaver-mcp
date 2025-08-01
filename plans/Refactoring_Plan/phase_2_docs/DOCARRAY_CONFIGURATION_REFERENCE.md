# DocArray Configuration Reference

<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

**Version**: 1.0.0
**Date**: 2025-01-26
**Purpose**: Complete configuration reference for DocArray integration

## Overview

This reference provides comprehensive configuration options for DocArray backends in CodeWeaver, including environment variables, TOML configuration files, and programmatic configuration examples.

## Quick Start Examples

### Basic Configuration

```python
from codeweaver import CodeWeaverFactory
from codeweaver.backends.providers.docarray.config import QdrantDocArrayConfig

# Create factory with DocArray support
factory = CodeWeaverFactory(enable_docarray=True)

# Basic Qdrant DocArray configuration
config = QdrantDocArrayConfig(
    url="http://localhost:6333",
    api_key="your-api-key",
    collection_name="my_codebase",
    schema_config={
        "embedding_dimension": 512,
        "include_sparse_vectors": True,
        "schema_template": "code_search"
    }
)

# Create backend
backend = factory.create_backend(config)
```

### Environment Variables

```bash
# Basic DocArray backend configuration
export CW_BACKEND_PROVIDER="docarray_qdrant"
export CW_BACKEND_URL="http://localhost:6333"
export CW_BACKEND_API_KEY="your-api-key"
export CW_BACKEND_COLLECTION="codeweaver"

# DocArray-specific settings
export CW_DOCARRAY_EMBEDDING_DIMENSION="512"
export CW_DOCARRAY_ENABLE_SPARSE="true"
export CW_DOCARRAY_SCHEMA_TEMPLATE="code_search"
export CW_DOCARRAY_BATCH_SIZE="100"
export CW_DOCARRAY_ENABLE_HYBRID_SEARCH="true"
```

### TOML Configuration

```toml
# codeweaver.toml

[backend]
provider = "docarray_qdrant"
url = "http://localhost:6333"
api_key = "your-api-key"
collection_name = "codeweaver"

[backend.schema_config]
embedding_dimension = 512
include_sparse_vectors = true
schema_template = "code_search"
enable_validation = true

[backend.schema_config.metadata_fields]
file_path = "str"
language = "str"
function_name = "str"
line_number = "int"

[backend.db_config]
prefer_grpc = false
timeout = 30.0
retry_total = 3

[backend.runtime_config]
batch_size = 100
enable_async = true
enable_hybrid_search = true
```

## Configuration Structure

### Backend Configuration Hierarchy

```
DocArrayBackendConfig
├── provider                    # Backend provider name
├── url                        # Connection URL
├── api_key                    # Authentication key
├── collection_name            # Collection/index name
├── schema_config              # Document schema configuration
│   ├── embedding_dimension    # Vector dimension
│   ├── include_sparse_vectors # Enable sparse vectors
│   ├── metadata_fields        # Typed metadata fields
│   ├── custom_fields          # Custom field definitions
│   ├── schema_template        # Predefined template
│   └── enable_validation      # Pydantic validation
├── db_config                  # Backend-specific database config
├── runtime_config             # DocArray runtime configuration
├── batch_size                 # Operation batch size
├── enable_async               # Asynchronous operations
├── connection_timeout         # Connection timeout
├── retry_attempts             # Retry attempts
├── enable_hybrid_search       # Hybrid search support
├── enable_compression         # Vector compression
└── enable_caching             # Query result caching
```

## Schema Configuration

### Predefined Schema Templates

#### Code Search Schema
Optimized for semantic code search with AST-aware metadata.

```python
schema_config = {
    "schema_template": "code_search",
    "embedding_dimension": 512,
    "include_sparse_vectors": True,
}

# Includes these fields automatically:
# - id: str
# - content: str
# - embedding: NdArray[512]
# - metadata: dict[str, Any]
# - sparse_vector: dict[str, float]
# - keywords: list[str]
# - meta_file_path: str
# - meta_language: str
# - meta_function_name: str
# - meta_class_name: str
# - meta_line_number: int
```

#### Semantic Search Schema
General-purpose semantic search with document metadata.

```python
schema_config = {
    "schema_template": "semantic_search",
    "embedding_dimension": 768,
    "include_sparse_vectors": False,
}

# Includes these fields automatically:
# - id: str
# - content: str
# - embedding: NdArray[768]
# - metadata: dict[str, Any]
# - meta_title: str
# - meta_author: str
# - meta_timestamp: str
# - meta_category: str
```

#### Multimodal Schema
Support for multimodal documents with image and text embeddings.

```python
schema_config = {
    "schema_template": "multimodal",
    "embedding_dimension": 512,
    "include_sparse_vectors": True,
    "custom_fields": {
        "image_embedding": ("NdArray[512]", "Image embedding vector"),
        "text_embedding": ("NdArray[512]", "Text embedding vector"),
        "image_url": ("str | None", "Image URL"),
    }
}
```

### Custom Schema Configuration

#### Basic Custom Schema

```python
schema_config = {
    "embedding_dimension": 1024,
    "include_sparse_vectors": True,
    "metadata_fields": {
        "document_type": "str",
        "priority": "int",
        "tags": "list[str]",
        "confidence": "float",
        "is_public": "bool",
    },
    "enable_validation": True,
}
```

#### Advanced Custom Schema

```python
from typing import Any
from pydantic import Field

schema_config = {
    "embedding_dimension": 512,
    "include_sparse_vectors": True,
    "metadata_fields": {
        "file_path": "str",
        "language": "str",
        "complexity_score": "float",
        "test_coverage": "float",
        "dependencies": "list[str]",
    },
    "custom_fields": {
        "code_ast": (
            "dict[str, Any]",
            Field(description="Abstract syntax tree representation")
        ),
        "git_hash": (
            "str | None",
            Field(default=None, description="Git commit hash")
        ),
        "last_modified": (
            "str",
            Field(description="Last modification timestamp")
        ),
    },
    "enable_validation": True,
}
```

## Backend-Specific Configuration

### Qdrant DocArray Backend

```python
from codeweaver.backends.providers.docarray.config import QdrantDocArrayConfig

config = QdrantDocArrayConfig(
    # Basic connection
    url="http://localhost:6333",
    api_key="your-api-key",
    collection_name="codeweaver",

    # Schema configuration
    schema_config={
        "embedding_dimension": 512,
        "include_sparse_vectors": True,
        "schema_template": "code_search",
    },

    # Qdrant-specific settings
    prefer_grpc=False,
    grpc_port=6334,

    # Database configuration
    db_config={
        "distance": "Cosine",
        "timeout": 30.0,
        "retry_total": 3,
        "prefer_grpc": False,
    },

    # Runtime configuration
    runtime_config={
        "batch_size": 100,
        "scroll_size": 1000,
    },

    # Performance settings
    batch_size=100,
    enable_async=True,
    connection_timeout=30.0,
    retry_attempts=3,

    # Feature flags
    enable_hybrid_search=True,
    enable_compression=False,
    enable_caching=True,
)
```

### Pinecone DocArray Backend

```python
from codeweaver.backends.providers.docarray.config import PineconeDocArrayConfig

config = PineconeDocArrayConfig(
    # Basic connection
    api_key="your-pinecone-api-key",
    environment="us-west1-gcp",
    index_name="codeweaver-index",

    # Schema configuration
    schema_config={
        "embedding_dimension": 1536,  # OpenAI ada-002
        "include_sparse_vectors": False,  # Pinecone doesn't support native sparse
        "metadata_fields": {
            "file_path": "str",
            "language": "str",
            "timestamp": "str",
        },
    },

    # Pinecone-specific settings
    environment="us-west1-gcp",
    index_type="approximated",

    # Database configuration
    db_config={
        "metric": "cosine",
        "shards": 1,
        "replicas": 1,
        "pods": 1,
        "pod_type": "p1.x1",
    },

    # Performance settings
    batch_size=100,
    enable_async=True,
    connection_timeout=60.0,
    retry_attempts=5,

    # Feature flags
    enable_hybrid_search=False,  # No native support
    enable_compression=True,
    enable_caching=True,
)
```

### Weaviate DocArray Backend

```python
from codeweaver.backends.providers.docarray.config import WeaviateDocArrayConfig

config = WeaviateDocArrayConfig(
    # Basic connection
    url="http://localhost:8080",
    api_key="your-weaviate-api-key",

    # Schema configuration
    schema_config={
        "embedding_dimension": 768,
        "include_sparse_vectors": True,
        "schema_template": "semantic_search",
    },

    # Weaviate-specific settings
    class_name="CodeWeaverDoc",
    vectorizer=None,  # Use external embeddings

    # Database configuration
    db_config={
        "startup_period": 5,
        "additional_headers": {
            "X-Cohere-Api-Key": "your-cohere-key",  # If using Cohere
        },
    },

    # Runtime configuration
    runtime_config={
        "consistency_level": "ALL",
        "timeout": ("5s", "10s"),
    },

    # Performance settings
    batch_size=50,
    enable_async=True,
    connection_timeout=30.0,
    retry_attempts=3,

    # Feature flags
    enable_hybrid_search=True,  # Native BM25 support
    enable_compression=False,
    enable_caching=True,
)
```

## Environment Variable Reference

### Core Backend Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `CW_BACKEND_PROVIDER` | DocArray backend provider | None | `docarray_qdrant` |
| `CW_BACKEND_URL` | Backend connection URL | None | `http://localhost:6333` |
| `CW_BACKEND_API_KEY` | Authentication API key | None | `your-api-key` |
| `CW_BACKEND_COLLECTION` | Collection/index name | `codeweaver` | `my_codebase` |
| `CW_BACKEND_TIMEOUT` | Connection timeout (seconds) | `30.0` | `60.0` |
| `CW_BACKEND_RETRY_ATTEMPTS` | Number of retry attempts | `3` | `5` |

### DocArray Schema Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `CW_DOCARRAY_EMBEDDING_DIMENSION` | Vector dimension | `512` | `1536` |
| `CW_DOCARRAY_ENABLE_SPARSE` | Enable sparse vectors | `false` | `true` |
| `CW_DOCARRAY_SCHEMA_TEMPLATE` | Predefined schema template | None | `code_search` |
| `CW_DOCARRAY_ENABLE_VALIDATION` | Enable Pydantic validation | `true` | `false` |
| `CW_DOCARRAY_BATCH_SIZE` | Operation batch size | `100` | `50` |

### Backend-Specific Variables

#### Qdrant Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `CW_QDRANT_PREFER_GRPC` | Use gRPC instead of HTTP | `false` | `true` |
| `CW_QDRANT_GRPC_PORT` | gRPC port | None | `6334` |
| `CW_QDRANT_DISTANCE_METRIC` | Distance metric | `Cosine` | `Euclidean` |

#### Pinecone Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `CW_PINECONE_ENVIRONMENT` | Pinecone environment | None | `us-west1-gcp` |
| `CW_PINECONE_INDEX_TYPE` | Index type | `approximated` | `exact` |
| `CW_PINECONE_METRIC` | Distance metric | `cosine` | `euclidean` |
| `CW_PINECONE_SHARDS` | Number of shards | `1` | `2` |

#### Weaviate Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `CW_WEAVIATE_CLASS_NAME` | Weaviate class name | `CodeWeaverDoc` | `MyDoc` |
| `CW_WEAVIATE_VECTORIZER` | Vectorizer module | None | `text2vec-cohere` |
| `CW_WEAVIATE_CONSISTENCY_LEVEL` | Consistency level | `ALL` | `QUORUM` |

### Feature Flag Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `CW_DOCARRAY_ENABLE_HYBRID_SEARCH` | Enable hybrid search | `false` | `true` |
| `CW_DOCARRAY_ENABLE_COMPRESSION` | Enable vector compression | `false` | `true` |
| `CW_DOCARRAY_ENABLE_CACHING` | Enable query caching | `false` | `true` |
| `CW_DOCARRAY_ENABLE_ASYNC` | Enable async operations | `true` | `false` |

## Configuration Examples by Use Case

### Use Case 1: Code Search with Qdrant

**Scenario**: Semantic code search with hybrid capabilities for large codebases.

```toml
# codeweaver.toml
[backend]
provider = "docarray_qdrant"
url = "http://localhost:6333"
collection_name = "large_codebase"
batch_size = 200
enable_async = true
enable_hybrid_search = true

[backend.schema_config]
embedding_dimension = 512
include_sparse_vectors = true
schema_template = "code_search"

[backend.db_config]
distance = "Cosine"
timeout = 45.0
prefer_grpc = true

[backend.runtime_config]
scroll_size = 2000
batch_size = 200
```

```bash
# Environment variables
export CW_BACKEND_PROVIDER="docarray_qdrant"
export CW_BACKEND_URL="http://localhost:6333"
export CW_BACKEND_API_KEY="your-api-key"
export CW_DOCARRAY_EMBEDDING_DIMENSION="512"
export CW_DOCARRAY_ENABLE_SPARSE="true"
export CW_DOCARRAY_SCHEMA_TEMPLATE="code_search"
export CW_DOCARRAY_ENABLE_HYBRID_SEARCH="true"
export CW_QDRANT_PREFER_GRPC="true"
```

### Use Case 2: Document Search with Pinecone

**Scenario**: High-performance document search using Pinecone cloud service.

```toml
# codeweaver.toml
[backend]
provider = "docarray_pinecone"
environment = "us-west1-gcp"
index_name = "document-search"
batch_size = 100
enable_compression = true

[backend.schema_config]
embedding_dimension = 1536
include_sparse_vectors = false
schema_template = "semantic_search"

[backend.db_config]
metric = "cosine"
shards = 2
replicas = 1
pod_type = "p1.x2"
```

```python
# Programmatic configuration
config = PineconeDocArrayConfig(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="us-west1-gcp",
    index_name="document-search",
    schema_config={
        "embedding_dimension": 1536,
        "include_sparse_vectors": False,
        "metadata_fields": {
            "title": "str",
            "author": "str",
            "category": "str",
            "published_date": "str",
        },
    },
    db_config={
        "metric": "cosine",
        "shards": 2,
        "replicas": 1,
        "pod_type": "p1.x2",
    },
    enable_compression=True,
    enable_caching=True,
)
```

### Use Case 3: Multimodal Search with Weaviate

**Scenario**: Combined text and image search with hybrid capabilities.

```toml
# codeweaver.toml
[backend]
provider = "docarray_weaviate"
url = "http://localhost:8080"
class_name = "MultiModalDoc"
enable_hybrid_search = true

[backend.schema_config]
embedding_dimension = 768
include_sparse_vectors = true
schema_template = "multimodal"

[backend.schema_config.custom_fields]
image_embedding = ["NdArray[768]", "Image CLIP embedding"]
text_embedding = ["NdArray[768]", "Text CLIP embedding"]
image_url = ["str | None", "Image URL"]

[backend.db_config]
consistency_level = "QUORUM"
```

## Configuration Validation

### Automatic Validation

DocArray backends perform comprehensive configuration validation:

```python
from codeweaver.backends.providers.docarray.config import DocArrayConfigFactory

# Validate configuration
backend_type = "docarray_qdrant"
config_dict = {
    "url": "http://localhost:6333",
    "schema_config": {
        "embedding_dimension": 512,
        "include_sparse_vectors": True,
    }
}

is_valid, errors = DocArrayConfigFactory.validate_backend_config(
    backend_type, config_dict
)

if not is_valid:
    print("Configuration errors:", errors)
```

### Common Validation Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `embedding_dimension must be positive` | Dimension ≤ 0 | Set positive integer (e.g., 512) |
| `Invalid field name: 123field` | Non-identifier field name | Use valid Python identifiers |
| `Unsupported field type: custom` | Invalid metadata field type | Use supported types: str, int, float, bool |
| `Backend 'invalid' not supported` | Unknown backend provider | Use supported providers |
| `Missing required field: url` | Missing connection URL | Provide valid connection URL |

### Best Practices

1. **Use Environment Variables**: For sensitive data like API keys
2. **TOML for Complex Config**: For multi-environment setups
3. **Schema Templates**: Start with predefined templates
4. **Validate Early**: Check configuration before deployment
5. **Monitor Performance**: Adjust batch sizes based on usage
6. **Enable Async**: For better performance in production
7. **Use Hybrid Search**: When you need both semantic and keyword search
8. **Compression for Large Vectors**: Enable for dimensions > 1000
9. **Caching for Read-Heavy**: Enable for frequently accessed data
10. **Proper Error Handling**: Always handle configuration errors gracefully

## Troubleshooting

### Common Issues

#### Connection Issues
```bash
# Check backend availability
curl -f http://localhost:6333/collections  # Qdrant
curl -f http://localhost:8080/v1/meta      # Weaviate
```

#### Configuration Issues
```python
# Enable debug logging
import logging
logging.getLogger("codeweaver.backends.providers.docarray").setLevel(logging.DEBUG)
```

#### Performance Issues
```toml
# Optimize batch sizes
[backend]
batch_size = 50  # Reduce for memory constraints
enable_async = true
enable_compression = true  # For large vectors
```

#### Schema Issues
```python
# Validate schema before use
from codeweaver.backends.providers.docarray.schema import DocumentSchemaGenerator

try:
    schema = DocumentSchemaGenerator.create_schema(config)
    print(f"Schema created with fields: {list(schema.model_fields.keys())}")
except Exception as e:
    print(f"Schema creation failed: {e}")
```

This configuration reference provides comprehensive guidance for setting up and tuning DocArray backends in CodeWeaver, covering all major use cases and deployment scenarios.
