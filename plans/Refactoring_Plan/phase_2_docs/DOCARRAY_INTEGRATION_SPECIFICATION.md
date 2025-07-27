# DocArray Integration Specification

<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

**Version**: 1.0.0  
**Date**: 2025-01-26  
**Status**: Design Specification

## Overview

This specification defines the integration of DocArray into CodeWeaver's plugin architecture, providing a unified vector database interface while maintaining full compatibility with existing backend protocols and factory patterns.

## Executive Summary

DocArray integration provides:
- **Unified Backend Access**: Single interface supporting 10+ vector databases
- **Protocol Compliance**: Full VectorBackend and HybridSearchBackend compatibility
- **Dynamic Schema Generation**: Configurable document schemas for different use cases
- **Seamless Registration**: Integration with existing BackendRegistry system
- **Fallback Strategies**: Graceful degradation to direct backends when needed

## Architecture Overview

### Integration Strategy: Plugin Backend System

The DocArray integration follows a **Plugin Backend System** approach that:

1. Implements DocArray backends as standard CodeWeaver plugins
2. Provides adapter classes for protocol compliance
3. Maintains existing factory and registry patterns
4. Enables dynamic document schema generation
5. Supports both native DocArray features and fallback implementations

### Key Components

```
DocArray Integration Architecture

┌─────────────────────────────────────────────────────────────┐
│                    CodeWeaver Factory                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Backend Registry │  │ Plugin Manager  │  │ Config Mgr   │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              DocArray Backend Registry                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Qdrant Index  │  │ Pinecone Index  │  │ Weaviate Idx │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            DocArray Backend Adapter                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Protocol Bridge │  │ Schema Generator│  │ Query Trans. │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   DocArray Core                             │
├─────────────────────────────────────────────────────────────┤
│  BaseDoc │ DocList │ DocumentIndex │ Pydantic Integration   │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. DocArray Backend Factory

**Purpose**: Creates DocArray-powered backend instances with dynamic schema generation.

**Key Features**:
- Dynamic document schema creation based on configuration
- Backend type mapping to DocArray document index classes
- Configuration translation between CodeWeaver and DocArray formats
- Support for custom fields and metadata structures

**Supported Backends**:
- **Qdrant**: `QdrantDocumentIndex` with hybrid search support
- **Pinecone**: `PineconeDocumentIndex` with metadata filtering
- **Weaviate**: `WeaviateDocumentIndex` with BM25 hybrid search
- **ChromaDB**: `ChromaDocumentIndex` with collection management
- **Redis**: `RedisDocumentIndex` with fast in-memory operations
- **Milvus**: `MilvusDocumentIndex` with high-performance search
- **Elasticsearch**: `ElasticsearchDocumentIndex` with full-text capabilities

### 2. Universal Backend Adapter

**Purpose**: Makes DocArray backends compatible with CodeWeaver's VectorBackend protocol.

**Key Responsibilities**:
- Protocol translation between CodeWeaver and DocArray APIs
- Vector point conversion to/from DocArray document format
- Search result transformation and filtering
- Error handling and exception mapping
- Performance optimization through batching

**Protocol Compliance**:
- ✅ `VectorBackend`: Full compatibility
- ✅ `HybridSearchBackend`: Native support where available, fallback otherwise
- ✅ `StreamingBackend`: Batch processing support
- 🔄 `TransactionalBackend`: Future enhancement

### 3. Dynamic Document Schema Generation

**Purpose**: Creates type-safe DocArray document schemas based on use case requirements.

**Schema Components**:
- **Base Fields**: `id`, `content`, `embedding`, `metadata`
- **Sparse Vectors**: Optional sparse vector support for hybrid search
- **Custom Metadata**: Typed metadata fields based on configuration
- **Extensibility**: Support for completely custom field definitions

**Type Safety Features**:
- Pydantic v2 validation and serialization
- Vector dimension validation at type level
- Automatic type conversion and coercion
- Runtime schema validation

### 4. Hybrid Search Integration

**Purpose**: Provides hybrid search capabilities using DocArray's native features where available.

**Implementation Strategy**:
- **Native Support**: Use DocArray's built-in hybrid search for supported backends
- **Fallback Fusion**: Client-side RRF fusion for backends without native support
- **Sparse Vector Management**: Automatic sparse vector generation and indexing
- **Query Optimization**: Intelligent query planning and execution

## Integration Points

### 1. Factory Integration

```python
# Enable DocArray backends in factory
factory = CodeWeaverFactory(enable_docarray=True)

# Create DocArray-powered backend
config = DocArrayBackendConfig(
    provider="docarray_qdrant",
    url="http://localhost:6333",
    embedding_dimension=512,
    enable_sparse_vectors=True
)
backend = factory.create_backend(config)
```

### 2. Registry Integration

```python
# DocArray backends registered alongside native backends
available_backends = factory.get_available_components()["backends"]

# Example output:
# {
#   "qdrant": BackendInfo(...),           # Native Qdrant
#   "docarray_qdrant": BackendInfo(...),  # DocArray Qdrant
#   "docarray_pinecone": BackendInfo(...), # DocArray Pinecone
#   # ...
# }
```

### 3. Configuration Integration

```python
# Enhanced configuration with DocArray-specific options
class DocArrayBackendConfig(BackendConfig):
    schema_config: dict[str, Any] = Field(default_factory=dict)
    embedding_dimension: int = Field(default=512)
    enable_sparse_vectors: bool = Field(default=False)
    custom_fields: dict[str, Any] = Field(default_factory=dict)
    runtime_config: dict[str, Any] = Field(default_factory=dict)
```

## Protocol Compatibility

### VectorBackend Protocol

| Method | DocArray Support | Implementation |
|--------|------------------|----------------|
| `create_collection` | ✅ Full | Automatic collection creation |
| `upsert_vectors` | ✅ Full | DocList batch operations |
| `search_vectors` | ✅ Full | DocumentIndex.find() |
| `delete_vectors` | ✅ Full | DocumentIndex.delete() |
| `get_collection_info` | ✅ Full | Index metadata extraction |
| `list_collections` | ✅ Full | Backend-specific listing |
| `delete_collection` | ✅ Full | Index deletion |

### HybridSearchBackend Protocol

| Method | DocArray Support | Implementation |
|--------|------------------|----------------|
| `create_sparse_index` | ✅ Conditional | Native where supported, fallback otherwise |
| `hybrid_search` | ✅ Conditional | Native hybrid search or RRF fusion |
| `update_sparse_vectors` | ✅ Conditional | Sparse vector updates |

### Performance Characteristics

| Backend | Native Performance | DocArray Overhead | Recommendation |
|---------|-------------------|-------------------|----------------|
| Qdrant | 100% | ~5-10% | Use for unified interface |
| Pinecone | 100% | ~5-15% | Use for feature consistency |
| Weaviate | 100% | ~5-10% | Use for hybrid search |
| ChromaDB | 100% | ~10-20% | Consider for development |
| In-Memory | 100% | ~15-25% | Use for testing |

## Benefits and Trade-offs

### Benefits

1. **Unified Interface**: Single API across all vector databases
2. **Type Safety**: Strong Pydantic v2 integration with validation
3. **Rapid Development**: Faster integration of new vector databases
4. **Configuration Consistency**: Uniform configuration patterns
5. **Feature Parity**: Consistent feature set across backends
6. **Testing Simplification**: Single test suite for all backends

### Trade-offs

1. **Performance Overhead**: 5-25% overhead depending on backend
2. **Feature Limitations**: May not expose all backend-specific features
3. **Dependency Complexity**: Additional dependency chain
4. **Learning Curve**: DocArray-specific patterns and conventions
5. **Version Dependencies**: Potential Pydantic v2 compatibility issues

## Migration Strategy

### Phase 1: Opt-in Integration (Current)
- DocArray backends available as additional options
- Existing backends remain unchanged
- Users can choose based on needs

### Phase 2: Feature Parity
- Ensure DocArray backends support all features
- Performance optimization and tuning
- Comprehensive testing and validation

### Phase 3: Unified Interface (Future)
- Consider making DocArray the primary interface
- Maintain native backends for performance-critical use cases
- Deprecation strategy for redundant implementations

## Implementation Guidelines

### 1. Error Handling

```python
# Graceful error handling with fallbacks
class DocArrayBackendAdapter:
    async def search_vectors(self, ...):
        try:
            return await self._native_search(...)
        except DocArrayError as e:
            logger.warning(f"DocArray search failed: {e}")
            return await self._fallback_search(...)
```

### 2. Performance Optimization

```python
# Batch operations for better performance
async def upsert_vectors(self, vectors: list[VectorPoint]):
    # Batch vectors into optimal sizes
    for batch in self._batch_vectors(vectors, batch_size=100):
        doc_list = DocList[self.doc_class]([
            self._vector_to_doc(v) for v in batch
        ])
        self.doc_index.index(doc_list)
```

### 3. Configuration Validation

```python
# Comprehensive configuration validation
def validate_docarray_config(config: DocArrayBackendConfig) -> ValidationResult:
    errors = []
    
    # Validate embedding dimension
    if config.embedding_dimension <= 0:
        errors.append("embedding_dimension must be positive")
    
    # Validate backend availability
    if not self._is_backend_available(config.provider):
        errors.append(f"Backend {config.provider} not available")
    
    return ValidationResult(is_valid=not errors, errors=errors)
```

## Testing Strategy

### 1. Unit Tests
- Protocol compliance testing
- Schema generation validation
- Configuration validation
- Error handling verification

### 2. Integration Tests
- End-to-end backend operations
- Performance benchmarking
- Feature compatibility testing
- Cross-backend consistency

### 3. Performance Tests
- Latency and throughput measurement
- Memory usage profiling
- Overhead quantification
- Scalability testing

## Security Considerations

### 1. Input Validation
- Strict schema validation using Pydantic
- Vector dimension validation
- Metadata sanitization
- Query parameter validation

### 2. Configuration Security
- Secure credential handling
- Environment variable integration
- Connection encryption support
- Access control validation

### 3. Data Privacy
- Metadata filtering
- Sensitive data handling
- Audit logging
- Compliance support

## Future Enhancements

### 1. Advanced Features
- Multi-modal document support
- Advanced query builders
- Custom distance metrics
- Batch processing optimization

### 2. Performance Optimizations
- Connection pooling
- Query caching
- Async optimization
- Memory management

### 3. Additional Backends
- MongoDB Atlas Vector Search
- Supabase Vector
- LanceDB integration
- Custom backend plugins

### 4. Monitoring Integration
- Performance metrics
- Usage analytics
- Error tracking
- Health monitoring

## Conclusion

The DocArray integration provides a powerful, unified interface for vector database operations while maintaining full compatibility with CodeWeaver's existing architecture. The plugin-based approach ensures flexibility and allows users to choose between native backends for maximum performance or DocArray backends for consistency and ease of use.

The design prioritizes:
- **Compatibility**: Full protocol compliance with existing interfaces
- **Flexibility**: Support for both unified and native approaches
- **Performance**: Minimal overhead while providing enhanced features
- **Maintainability**: Clean separation of concerns and extensible design
- **User Experience**: Consistent configuration and usage patterns

This integration positions CodeWeaver to support a wide range of vector databases through a single, well-tested interface while preserving the ability to use optimized native implementations when needed.