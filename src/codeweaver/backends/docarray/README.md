# DocArray Backend Integration

This module provides a unified interface to vector databases through DocArray, maintaining full compatibility with CodeWeaver's existing backend protocols.

## Overview

The DocArray integration enables CodeWeaver to support 10+ additional vector databases through a single, well-tested interface while preserving the ability to use optimized native implementations when needed.

## Architecture

```
CodeWeaver Factory System
├── Native Backends (e.g., QdrantHybridBackend)
└── DocArray Backends
    ├── Universal Adapter (BaseDocArrayAdapter, DocArrayHybridAdapter)
    ├── Dynamic Schema Generation (DocumentSchemaGenerator)
    ├── Configuration Management (DocArrayConfigFactory)
    └── Backend Implementations
        ├── QdrantDocArrayBackend ✅
        ├── PineconeDocArrayBackend (planned)
        └── WeaviateDocArrayBackend (planned)
```

## Key Features

- **Protocol Compliance**: Full VectorBackend and HybridSearchBackend compatibility
- **Dynamic Schemas**: Configurable document schemas for different use cases
- **Graceful Fallbacks**: Works without DocArray dependencies installed
- **Type Safety**: Strong typing with Pydantic v2 validation
- **Hybrid Search**: Native hybrid search support where available, RRF fallback otherwise

## Usage

### Basic Configuration

```python
from codeweaver.backends.docarray.config import DocArrayConfigFactory
from codeweaver._types import ProviderKind

# Create DocArray Qdrant backend configuration
config = DocArrayConfigFactory.create_config(
    "docarray_qdrant",
    url="http://localhost:6333",
    api_key="your-api-key",
    collection_name="code-embeddings"
)
```

### Schema Templates

```python
from codeweaver.backends.docarray.schema import SchemaTemplates

# Code search optimized schema
code_schema = SchemaTemplates.code_search_schema(512)

# General semantic search schema  
semantic_schema = SchemaTemplates.semantic_search_schema(512)

# Multimodal document schema
multimodal_schema = SchemaTemplates.multimodal_schema(512)
```

### Custom Schemas

```python
from codeweaver.backends.docarray.schema import DocumentSchemaGenerator, SchemaConfig

# Create custom schema configuration
config = SchemaConfig(
    embedding_dimension=768,
    include_sparse_vectors=True,
    metadata_fields={
        "file_path": str,
        "language": str,
        "complexity": int,
    }
)

# Generate schema class
CustomDoc = DocumentSchemaGenerator.create_schema(config, "CustomCodeDoc")
```

## Supported Backends

### Currently Implemented
- **DocArray Qdrant** (`docarray_qdrant`) - Full hybrid search support

### Planned
- **DocArray Pinecone** (`docarray_pinecone`) - Metadata filtering with hybrid search
- **DocArray Weaviate** (`docarray_weaviate`) - Native BM25 hybrid search
- **DocArray ChromaDB** (`docarray_chroma`) - Collection management
- **DocArray Redis** (`docarray_redis`) - Fast in-memory operations

## Dependencies

DocArray backends are optional and require additional dependencies:

```bash
# For Qdrant support
pip install docarray[qdrant] qdrant-client

# For Pinecone support (when implemented)
pip install docarray[pinecone] pinecone-client

# For Weaviate support (when implemented)  
pip install docarray[weaviate] weaviate-client
```

The system gracefully handles missing dependencies and only registers available backends.

## Benefits

1. **Unified Interface**: Single API across all vector databases
2. **Type Safety**: Strong Pydantic v2 integration with validation
3. **Rapid Development**: Faster integration of new vector databases
4. **Configuration Consistency**: Uniform configuration patterns
5. **Feature Parity**: Consistent feature set across backends
6. **Testing Simplification**: Single test suite for all backends

## Trade-offs

1. **Performance Overhead**: 5-25% overhead depending on backend
2. **Feature Limitations**: May not expose all backend-specific features
3. **Dependency Complexity**: Additional dependency chain
4. **Learning Curve**: DocArray-specific patterns and conventions

## Integration with CodeWeaver

The DocArray backends integrate seamlessly with CodeWeaver's factory system:

```python
from codeweaver.backends.factory import BackendFactory

# List all available providers (includes DocArray backends when dependencies available)
providers = BackendFactory.list_supported_providers()

# Create backend (will use DocArray if available, fallback to native otherwise)
backend = BackendFactory.create_backend(config)
```

## Testing

The implementation includes comprehensive tests:

```bash
# Test DocArray integration
pytest tests/unit/test_docarray_integration.py

# Test with DocArray installed (requires dependencies)
pytest tests/unit/test_docarray_integration.py::TestDocArrayWithDependencies
```

## Future Enhancements

- Multi-modal document support
- Advanced query builders  
- Custom distance metrics
- Connection pooling
- Query caching
- Additional backend implementations