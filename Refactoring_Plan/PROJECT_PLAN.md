<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# CodeWeaver Extensibility & Configuration Architecture Design

**Research Date**: July 2025
**Status**: Design Phase
**Target**: Plugin-based architecture supporting multiple backends, embedding providers, and data sources

## 🎯 Executive Summary

This design transforms CodeWeaver from a tightly-coupled MCP server into an extensible platform supporting 15+ vector databases, 5+ embedding providers, and multiple data sources while maintaining backward compatibility and performance leadership.

### **Key Benefits**
- **Zero breaking changes** for existing deployments
- **10x faster** integration of new backends/providers
- **Community ecosystem** for third-party contributions
- **Enterprise flexibility** across diverse infrastructures
- **Future-proof** hybrid search capabilities

---

## 🔍 Current Architecture Analysis

### **Tight Coupling Points Identified**

| Component | Location | Issue |
|-----------|----------|-------|
| Vector Database | `server.py:77` | Direct Qdrant client instantiation |
| Reranking | `server.py:61-63` | Hardcoded VoyageAI reranker |
| Embeddings | `embeddings.py` | Only Voyage/OpenAI providers |
| Configuration | `config.py` | Provider-specific sections |
| Data Sources | File system only | Limited to local files |

### **Current Dependencies**
- **Vector Database**: Qdrant only
- **Embeddings**: VoyageAI + OpenAI only
- **Reranking**: VoyageAI only
- **Data Sources**: File system only
- **Chunking**: ast-grep only

---

## 🏗️ Proposed Extensible Architecture

### **1. Plugin-Based Backend Abstraction**

```python
# src/codeweaver/backends/base.py
from abc import ABC, abstractmethod
from typing import Any, Protocol

class VectorBackend(Protocol):
    """Protocol for vector database backends."""

    async def create_collection(self, name: str, dimension: int) -> None: ...
    async def upsert_vectors(self, vectors: list[VectorPoint]) -> None: ...
    async def search_vectors(self, query_vector: list[float],
                           filters: dict[str, Any] | None = None,
                           limit: int = 10) -> list[SearchResult]: ...
    async def delete_vectors(self, ids: list[str]) -> None: ...
    async def get_collection_info(self, name: str) -> CollectionInfo: ...

class HybridSearchBackend(VectorBackend, Protocol):
    """Extended protocol for hybrid search capabilities."""

    async def create_sparse_index(self, name: str, fields: list[str]) -> None: ...
    async def hybrid_search(self, dense_vector: list[float],
                          sparse_query: dict[str, Any],
                          hybrid_strategy: str = "rrf") -> list[SearchResult]: ...

# Backend implementations
class QdrantBackend(VectorBackend): ...
class PineconeBackend(VectorBackend): ...
class ChromaBackend(VectorBackend): ...
class WeaviateBackend(VectorBackend, HybridSearchBackend): ...
class PgVectorBackend(VectorBackend): ...
```

### **2. Embedding Provider Abstraction**

```python
# src/codeweaver/providers/embeddings.py
class EmbeddingProvider(Protocol):
    """Universal embedding provider interface."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...
    @property
    def dimension(self) -> int: ...
    @property
    def model_name(self) -> str: ...

class RerankProvider(Protocol):
    """Universal reranking provider interface."""

    async def rerank(self, query: str, documents: list[str],
                    top_k: int | None = None) -> list[RerankResult]: ...

# Provider implementations
class VoyageAIProvider(EmbeddingProvider, RerankProvider): ...
class OpenAIProvider(EmbeddingProvider): ...
class SentenceTransformersProvider(EmbeddingProvider): ...
class CohereProvider(EmbeddingProvider, RerankProvider): ...
class HuggingFaceProvider(EmbeddingProvider): ...

# Provider registry
class ProviderRegistry:
    @classmethod
    def create_CW_EMBEDDING_PROVIDER(cls, config: EmbeddingConfig) -> EmbeddingProvider:
        if config.provider == "voyage":
            return VoyageAIProvider(config)
        elif config.provider == "openai":
            return OpenAIProvider(config)
        # ... more providers
```

### **3. Data Source Abstraction**

```python
# src/codeweaver/sources/base.py
class DataSource(Protocol):
    """Universal data source interface."""

    async def discover_content(self, config: SourceConfig) -> list[ContentItem]: ...
    async def read_content(self, item: ContentItem) -> str: ...
    async def watch_changes(self, callback: Callable) -> SourceWatcher: ...

class ContentItem:
    path: str
    content_type: str  # 'file', 'url', 'database', 'api'
    metadata: dict[str, Any]
    last_modified: datetime | None

# Source implementations
class FileSystemSource(DataSource): ...
class GitRepositorySource(DataSource): ...
class DatabaseSource(DataSource): ...
class APISource(DataSource): ...
class WebCrawlerSource(DataSource): ...
```

### **4. Middleware Integration Layer**

```python
# src/codeweaver/middleware/core.py
class CodeWeaverMiddleware:
    """FastMCP middleware for CodeWeaver extensibility."""

    def __init__(self, backend_factory: BackendFactory,
                 provider_factory: ProviderFactory,
                 source_factory: SourceFactory):
        self.backend_factory = backend_factory
        self.provider_factory = provider_factory
        self.source_factory = source_factory

    async def on_call_tool(self, request: ToolRequest,
                          next_handler: Callable) -> ToolResponse:
        """Intercept tool calls to inject appropriate backends."""
        tool_name = request.params.name

        if tool_name in ['search_code', 'index_codebase']:
            # Inject configured backend and providers
            context = self._build_context(request)
            return await next_handler(request, context)

        return await next_handler(request)

# DocArray integration layer
class DocArrayBackendAdapter:
    """Adapter to use DocArray document indexes as backends."""

    def __init__(self, docarray_index: DocumentIndex):
        self.index = docarray_index

    async def search_vectors(self, query_vector: list[float],
                           limit: int = 10) -> list[SearchResult]:
        # Convert to DocArray query and back to our format
        results = self.index.find(query_vector, limit=limit)
        return [self._convert_result(r) for r in results]
```

### **5. Enhanced Configuration Schema**

```toml
# Example enhanced configuration
[backend]
provider = "qdrant"  # qdrant, pinecone, chroma, weaviate, pgvector, docarray
hybrid_search = true

[backend.connection]
url = "https://your-qdrant-url"
api_key = "your-key"

[backend.sparse_vector_config]
vector_name = "sparse"
index_type = "keyword"

[embedding]
provider = "voyage"
model = "voyage-code-3"
rerank_provider = "voyage"
rerank_model = "voyage-rerank-2"

[data_sources]
[[data_sources.sources]]
type = "filesystem"
enabled = true
priority = 1

[data_sources.sources.config]
root_path = "/path/to/code"
patterns = ["**/*.py", "**/*.js"]

[[data_sources.sources]]
type = "git"
enabled = false
priority = 2

[data_sources.sources.config]
repository_url = "https://github.com/user/repo"
branch = "main"
```

### **6. Hybrid Search Compatibility Design**

```python
# src/codeweaver/search/hybrid.py
class HybridSearchManager:
    """Manages hybrid search across different backends."""

    def __init__(self, backend: VectorBackend, config: BackendConfig):
        self.backend = backend
        self.config = config
        self.supports_hybrid = isinstance(backend, HybridSearchBackend)

    async def setup_hybrid_indexing(self, collection_name: str):
        """Setup sparse indexing for hybrid search if supported."""
        if not self.supports_hybrid or not self.config.hybrid_search:
            return

        await self.backend.create_sparse_index(
            name=f"{collection_name}_sparse",
            fields=["content", "node_kind", "chunk_type"]
        )

    async def hybrid_search(self, query: str, query_vector: list[float],
                          **kwargs) -> list[SearchResult]:
        """Perform hybrid search combining dense and sparse retrieval."""
        if not self.supports_hybrid:
            # Fallback to dense-only search
            return await self.backend.search_vectors(query_vector, **kwargs)

        # Build sparse query from text
        sparse_query = self._build_sparse_query(query)

        return await self.backend.hybrid_search(
            dense_vector=query_vector,
            sparse_query=sparse_query,
            hybrid_strategy=kwargs.get('hybrid_strategy', 'rrf')
        )
```

### **7. Qdrant Hybrid Search Integration**

```python
# src/codeweaver/backends/qdrant_hybrid.py
class QdrantHybridBackend(QdrantBackend, HybridSearchBackend):
    """Enhanced Qdrant backend with hybrid search support."""

    async def create_collection(self, name: str, dimension: int) -> None:
        """Create collection with both dense and sparse vector support."""
        vectors_config = {
            "dense": VectorParams(size=dimension, distance=Distance.COSINE)
        }

        sparse_vectors_config = None
        if self.sparse_config.get('enabled', False):
            sparse_vectors_config = {
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=self.sparse_config.get('on_disk', False)
                    )
                )
            }

        await self.client.create_collection(
            collection_name=name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config
        )

    async def hybrid_search(self, dense_vector: list[float],
                          sparse_query: dict[str, Any],
                          hybrid_strategy: str = "rrf") -> list[SearchResult]:
        """Perform hybrid search using Qdrant's Query API."""
        prefetch_queries = [
            Prefetch(
                query=dense_vector,
                using="dense",
                limit=50  # Get more for fusion
            )
        ]

        if sparse_query and self.sparse_config.get('enabled', False):
            sparse_vector = self._build_sparse_vector(sparse_query)
            prefetch_queries.append(
                Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=50
                )
            )

        fusion = Fusion.RRF if hybrid_strategy == "rrf" else Fusion.DBSF

        result = await self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch_queries,
            query=FusionQuery(fusion=fusion),
            limit=20
        )

        return [self._convert_result(hit) for hit in result.points]
```

### **8. Factory Pattern Implementation**

```python
# src/codeweaver/factories.py
class BackendFactory:
    """Factory for creating vector database backends."""

    _backends: dict[str, type[VectorBackend]] = {
        'qdrant': QdrantBackend,
        'pinecone': PineconeBackend,
        'chroma': ChromaBackend,
        'weaviate': WeaviateBackend,
        'pgvector': PgVectorBackend,
        'docarray': DocArrayBackendAdapter,
    }

    @classmethod
    def create(cls, config: BackendConfig) -> VectorBackend:
        backend_class = cls._backends.get(config.provider)
        if not backend_class:
            raise ValueError(f"Unknown backend: {config.provider}")
        return backend_class(config)

class ExtensibilityManager:
    """Central coordinator for all extensibility features."""

    def __init__(self, config: CodeWeaverConfig):
        self.config = config
        self.backend = BackendFactory.create(config.backend)
        self.CW_EMBEDDING_PROVIDER = ProviderFactory.create_embedding(config.embedding)
        self.rerank_provider = ProviderFactory.create_rerank(config.embedding)
        self.data_sources = [SourceFactory.create(src) for src in config.data_sources.sources]
        self.hybrid_manager = HybridSearchManager(self.backend, config.backend)
```

---

## 📋 Complete Implementation Roadmap

### **Phase 1: Foundation Layer (Weeks 1-3)**

**Week 1: Protocol Design**
- Create `VectorBackend` and `HybridSearchBackend` protocols
- Design `EmbeddingProvider` and `RerankProvider` interfaces
- Define `DataSource` abstraction and `ContentItem` models
- Setup testing framework for protocol compliance

**Week 2: Configuration Refactoring**
- Extend `CodeWeaverConfig` with backend/provider sections
- Create `BackendConfig`, `EmbeddingProviderConfig`, `DataSourceConfig`
- Implement configuration validation and migration tools
- Add example configurations for different backends

**Week 3: Factory Pattern Implementation**
- Create `BackendFactory`, `ProviderFactory`, `SourceFactory`
- Implement registration system for new backends/providers
- Add plugin discovery mechanism for future extensibility
- Setup dependency injection container

### **Phase 2: Backend Abstraction (Weeks 4-6)**

**Week 4: Qdrant Backend Adapter**
- Refactor existing Qdrant code into `QdrantBackend` class
- Implement `VectorBackend` protocol for Qdrant
- Add `QdrantHybridBackend` with sparse vector support

**Week 5: Additional Vector Backends**
- Pencil in adapters for Pinecone, Chroma, Weaviate, and PgVector
  - No implementation yet, just the structure
- Add backend-specific optimization settings

**Week 6: DocArray Integration**
- Create `DocArrayBackendAdapter` for universal backend support
- Integrate with DocArray's document indexes
- Add support for in-memory backends (FAISS, HNSW)
- Performance testing and optimization

### **Phase 3: Provider Ecosystem (Weeks 7-9)**

**Week 7: Embedding Provider Refactoring**
- Refactor existing Voyage/OpenAI code into provider pattern
- Create `SentenceTransformersProvider` for local models
- Add `CohereProvider` with reranking capabilities
- Implement OpenAI-compatible API standard support

**Week 8: Reranking Provider Expansion**
- Extend `VoyageAIReranker` into provider pattern
- No more reranker implementations right now, but leave the door open for future providers
- Enable multiple provider chaining
- Add reranking performance metrics and caching

**Week 9: Local Model Integration**
- Add support for `sentence-transformers` local models
- Integrate `fastembed` for lightweight deployments
- Create model downloading and caching system
- Add GPU acceleration support for local models

### **Phase 4: Data Source Diversification (Weeks 10-12)**

**Week 10: Core Data Sources**
- Refactor file system indexing into `FileSystemSource`
- Implement `GitRepositorySource` with branch/commit support
- Add `DatabaseSource` for SQL/NoSQL content indexing
- Create unified content discovery and watching

**Week 11: Web & API Sources**
- Implement `WebCrawlerSource` for documentation sites
- Add `APISource` for REST/GraphQL endpoint indexing
- Create `ConfluenceSource` and `JiraSource` integrations
- Add rate limiting and politeness policies

**Week 12: Enterprise Sources**
- Implement `SharePointSource` for enterprise content
- Add `SlackSource` and `DiscordSource` for chat indexing
- Create `JupyterNotebookSource` for notebook content
- Add content deduplication and versioning

### **Phase 5: Middleware & Performance (Weeks 13-15)**

**Week 13: FastMCP Middleware**
- Implement `CodeWeaverMiddleware` for request routing
- Add authentication and authorization middleware
- Create caching middleware with Redis/Memory backends
- Add request/response logging and metrics

**Week 14: Performance Optimization**
- Implement connection pooling for database backends
- Add request batching and parallel processing
- Create intelligent caching strategies per backend
- Add performance monitoring and alerting

**Week 15: Advanced Features**
- Implement query optimization and rewriting
- Add semantic query expansion using embeddings
- Create result diversification algorithms
- Add A/B testing framework for search quality

### **Phase 6: Hybrid Search Implementation (Weeks 16-18)**

**Week 16: Sparse Vector Foundation**
- Integrate FastEmbed for sparse embeddings (`Qdrant/bm25`)
- Implement `SparseEmbeddingProvider` interface
- Add sparse vector indexing for Qdrant backend
- Create keyword extraction and BM25 scoring

**Week 17: Hybrid Search Engine**
- Implement `HybridSearchManager` with fusion strategies
- Add RRF (Reciprocal Rank Fusion) and DBSF support
- Create query decomposition (dense + sparse components)
- Add hybrid search result merging and ranking

**Week 18: Backend Integration**
- Extend Qdrant, Weaviate backends for hybrid search
- Add hybrid search support to other compatible backends
- Create fallback strategies for non-hybrid backends
- Add hybrid search performance benchmarking

### **Phase 7: Testing & Documentation (Weeks 19-20)**

**Week 19: Comprehensive Testing**
- Create integration tests for all backends/providers
- Add performance benchmarks and regression tests
- Test migration paths from v1 to v2 architecture
- Add end-to-end testing with different configurations

**Week 20: Documentation & Examples**
- Create comprehensive API documentation
- Add configuration examples for different setups
- Write migration guide from existing installations
- Create performance tuning and optimization guides

---

## 🎯 Success Metrics

### **Technical Metrics**
- **15+ backends/providers** supported through plugin architecture
- **<100ms latency** for search operations across all backends
- **99.9% backward compatibility** for existing configurations
- **5x faster** development of new integrations via abstractions

### **Adoption Metrics**
- **Zero breaking changes** for existing deployments
- **<1 day migration time** from v1 to v2 architecture
- **Community contributions** for new backends/providers
- **Enterprise deployment** support for diverse infrastructures

---

## 🔍 Research Sources

### **Middleware Solutions Evaluated**
- **DocArray**: Universal document indexes supporting 10+ vector databases
- **FastMCP**: Request/response middleware with tool interception
- **LangChain**: Vector store abstractions with 20+ backends

### **Vector Database Comparison (2025)**
- **Performance**: Qdrant leads in RPS/latency, Milvus fastest indexing
- **Popularity**: Milvus (25k stars), Qdrant (9k), Weaviate (8k), Chroma (6k)
- **Use Cases**: Pinecone (scale), Chroma (prototyping), pgvector (SQL), FAISS (research)

### **Embedding Model Standardization**
- **OpenAI API Compatibility**: De facto standard for embedding APIs
- **Sentence Transformers**: Local model deployment with MLflow integration
- **Voyage AI**: Best-in-class code embeddings with specialized models

### **Qdrant Hybrid Search (v1.10+)**
- **Sparse Vectors**: Separate index for keyword-based retrieval
- **Query API**: Server-side fusion with RRF and DBSF strategies
- **FastEmbed Integration**: Built-in sparse embedding with `Qdrant/bm25`

---

## 🚀 Migration Strategy

### **Backward Compatibility Guarantee**
- Existing configurations continue working unchanged
- Default provider remains Qdrant + VoyageAI
- Gradual opt-in to new features via configuration flags
- Automatic configuration migration tools provided

### **Community Adoption Path**
- Plugin system enables third-party backend/provider contributions
- Clear protocols and testing frameworks for new implementations
- Documentation and examples for custom integrations
- Community showcase of successful deployments

This design ensures CodeWeaver becomes the most flexible and extensible code search solution while maintaining its performance leadership and operational simplicity.
