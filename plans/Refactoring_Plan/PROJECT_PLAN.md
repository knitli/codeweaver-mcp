<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Feature TODO List
This document outlines the design and implementation plans for CodeWeaver.

### **5. Enhanced Configuration Schema**



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
