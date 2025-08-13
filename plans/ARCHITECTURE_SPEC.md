# CodeWeaver Architecture Specification

## Executive Summary

**CodeWeaver v2 Architecture**: Simplified MCP server following pydantic ecosystem patterns, avoiding over-abstraction while delivering intelligent codebase context through a single `find_code` tool interface.

**Core Principle**: Single responsibility with extensible providers → Simple, powerful, maintainable.

## System Architecture

### High-Level Design

```
CodeWeaver MCP Server
├── find_code tool (single entry point)
├── Settings (unified config via pydantic-settings)  
├── Pipeline (pydantic-graph orchestration)
│   ├── Intent Analysis → Query classification
│   ├── File Discovery → rignore integration
│   ├── Semantic Analysis → ast-grep + fallback
│   ├── Vector Search → embeddings + ranking
│   ├── Results Ranking → multi-signal scoring
│   └── Response Assembly → token-optimized output
├── Providers (pluggable backends)
│   ├── EmbeddingProvider → Voyage AI / OpenAI / local
│   ├── VectorStoreProvider → Qdrant / Chroma
│   ├── IntentProvider → AI sampling / NLP fallback
│   └── AnalysisProvider → ast-grep / text parsing
└── Services (utilities)
    ├── FileDiscovery → rignore wrapper
    ├── BackgroundIndexer → watchfiles integration  
    ├── TokenEstimator → tiktoken usage
    └── TelemetryReporter → PostHog integration
```

## Architectural Principles

### 1. Simplicity Over Abstraction
- **Single Tool Interface**: One `find_code` tool vs multiple middleware layers
- **Direct Integration**: Provider pattern vs protocol abstractions
- **Clear Data Flow**: Pipeline visualization vs implicit middleware chains

### 2. Pydantic Ecosystem Alignment  
- **Settings Models**: Multi-source config with validation
- **Type Safety**: BaseModel inheritance throughout
- **Dependency Injection**: FastMCP Context pattern
- **Composition**: Provider injection vs inheritance hierarchies

### 3. Performance & Scalability
- **Pipeline Parallelization**: Independent operations run concurrently
- **Minimal Overhead**: Direct provider calls vs service layers
- **Efficient Caching**: Vector store + background indexing
- **Token Optimization**: Response assembly with limits

## Core Components

### Settings Architecture

```python
class CodeWeaverSettings(BaseSettings):
    """Unified configuration following pydantic ecosystem patterns"""
    
    model_config = SettingsConfigDict(
        env_prefix="CODEWEAVER_",
        env_nested_delimiter="__", 
        env_file=[".env", ".codeweaver.env"],
        toml_file=["pyproject.toml", ".codeweaver.toml"]
    )
    
    # Core settings
    project_path: Path = Field(default_factory=Path.cwd)
    token_limit: int = 10000
    max_file_size: int = 10_000_000
    
    # Provider configuration  
    embedding: EmbeddingConfig = Field(default_factory=VoyageConfig)
    vector_store: VectorStoreConfig = Field(default_factory=QdrantConfig)
    
    # FastMCP integration
    fastmcp: FastMCPConfig = FastMCPConfig()
```

### Provider Architecture

**Abstract Interfaces:**
```python
class EmbeddingProvider(BaseModel, ABC):
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]: ...
    
    @abstractmethod  
    async def embed_query(self, query: str) -> List[float]: ...

class VectorStoreProvider(BaseModel, ABC):
    @abstractmethod
    async def search(self, vector: List[float], limit: int) -> List[SearchResult]: ...
    
    @abstractmethod
    async def upsert_chunks(self, chunks: List[CodeChunk]) -> None: ...
```

**Concrete Implementations:**
- **VoyageEmbeddingProvider**: voyage-code-3 model integration (requires custom adapter - no native pydantic-ai integration)
- **QdrantVectorStoreProvider**: Local/cloud vector storage with named vectors support 
- **OpenAIEmbeddingProvider**: Alternative embedding backend (via pydantic-ai if available)
- **LocalEmbeddingProvider**: Offline/privacy-focused option

**Voyage AI Adapter Architecture:**
```python
from typing import List, Optional
import voyageai
from pydantic import BaseModel, Field

class VoyageEmbeddingProvider(BaseModel):
    """Custom adapter for Voyage AI (no native pydantic-ai integration)"""
    
    api_key: Optional[str] = Field(default=None, description="Voyage AI API key")
    model: str = Field(default="voyage-code-3", description="Embedding model")
    max_retries: int = Field(default=3, description="Max retry attempts")
    timeout: Optional[int] = Field(default=30, description="Request timeout")
    
    def __post_init__(self):
        self._client = voyageai.AsyncClient(
            api_key=self.api_key,
            max_retries=self.max_retries,
            timeout=self.timeout
        )
    
    async def embed_texts(
        self, 
        texts: List[str], 
        input_type: Optional[str] = None
    ) -> List[List[float]]:
        """Generate embeddings with batching support"""
        result = await self._client.embed(
            texts=texts,
            model=self.model,
            input_type=input_type
        )
        return result.embeddings
        
    async def rerank_documents(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[str]:
        """Rerank documents by relevance"""
        result = await self._client.rerank(
            query=query,
            documents=documents,
            model="rerank-2",
            top_k=top_k
        )
        return [r.document for r in result.results]
```

### Pipeline Architecture

**pydantic-graph Integration:**
```python
@dataclass
class CodeSearchPipeline:
    """Main resolution pipeline orchestrating all components"""
    
    async def execute(self, query: str, context: Context) -> FindCodeResponse:
        return await self.pipeline_graph.run(
            start_node=AnalyzeIntent(query),
            deps=self._create_deps(context.settings)
        )

# Pipeline nodes (pydantic-graph compatible)
@dataclass
class AnalyzeIntent(BaseNode[CodeSearchState, PipelineDeps, ContentMatch]):
    query: str
    
    async def run(self, ctx: GraphRunContext[CodeSearchState, PipelineDeps]) -> DiscoverFiles | End[ContentMatch]:
        # Access dependencies through ctx.deps
        intent_result = await ctx.deps.intent_analyzer.analyze(self.query)
        
        # Update shared state
        ctx.state.intent = intent_result.intent
        ctx.state.confidence = intent_result.confidence
        
        # Return next node based on analysis
        return DiscoverFiles(intent_result)

@dataclass
class DiscoverFiles(BaseNode[CodeSearchState, PipelineDeps, ContentMatch]):  
    intent: IntentResult
    
    async def run(self, ctx: GraphRunContext[CodeSearchState, PipelineDeps]) -> AnalyzeCode | End[ContentMatch]:
        # File discovery based on intent
        patterns = get_patterns_for_intent(self.intent)
        discovered_files = await ctx.deps.file_discovery.discover_files(patterns)
        
        # Update state
        ctx.state.discovered_files = discovered_files
        
        return AnalyzeCode(discovered_files)
```

## Integration Strategy

### FastMCP Integration
- **Context Injection**: Use FastMCP's native dependency injection
- **Settings Composition**: Nested FastMCPConfig within CodeWeaverSettings
- **Tool Registration**: Single @app.tool() decorator pattern
- **Sampling Integration**: AI-powered intent analysis via context.sample()

### Dependency Integration
- **Voyage AI**: Primary embedding provider with batch processing
- **Qdrant**: Vector storage with named vectors (semantic + syntactic)
- **ast-grep-py**: Semantic code analysis with language detection
- **watchfiles**: Background indexing with change detection
- **rignore**: File discovery respecting gitignore patterns

## Data Flow

### Request Processing
1. **Agent Request** → `find_code(query, intent?, token_limit?)`
2. **Intent Analysis** → Query classification + strategy selection  
3. **File Discovery** → Pattern-based file filtering via rignore
4. **Semantic Analysis** → AST parsing or text chunking fallback
5. **Vector Search** → Embedding generation + similarity search
6. **Results Ranking** → Multi-signal scoring (semantic + syntactic + keyword)
7. **Response Assembly** → Token-optimized context delivery

### Background Indexing
1. **File Change Detection** → watchfiles monitoring
2. **Incremental Processing** → Changed files only
3. **Chunk Generation** → AST-based or line-based segmentation
4. **Embedding Generation** → Batch processing for efficiency
5. **Vector Storage** → Qdrant upsert with metadata

## Extensibility Patterns

### Adding New Providers
1. Implement provider interface (EmbeddingProvider, VectorStoreProvider)
2. Add configuration model extending base config
3. Register in provider factory function
4. Update settings with new provider option

### Adding New Pipeline Nodes  
1. Create pydantic-graph Node subclass
2. Define input/output models
3. Implement async run() method
4. Insert into pipeline graph

### Adding New Intent Types
1. Extend intent classification logic
2. Add corresponding pipeline strategies  
3. Update API documentation

## Error Handling & Resilience

### Graceful Degradation
- **AST Analysis Failure** → Text-based chunking fallback
- **Vector Store Unavailable** → File-based search only
- **Embedding API Timeout** → Cached embeddings + keyword search
- **AI Intent Analysis Failure** → Rule-based classification

### Performance Monitoring
- **Token Usage Tracking** → tiktoken integration
- **Response Time Monitoring** → Pipeline stage timing
- **Cache Hit Rates** → Vector store performance metrics
- **Error Rate Tracking** → Failure point identification

## Security Considerations

### Input Validation
- **Query Sanitization** → Pydantic model validation
- **File Path Validation** → Prevent directory traversal
- **Token Limit Enforcement** → Hard caps on response size

### Data Privacy
- **Local-First Option** → In-memory vector storage
- **API Key Security** → SecretStr for sensitive config
- **Telemetry Opt-out** → PostHog optional integration

## Performance Targets

### Response Times
- **Simple Queries** → <500ms (file-based search)
- **Complex Queries** → <2s (full pipeline with AI)
- **Background Indexing** → <1s per changed file

### Resource Usage  
- **Memory** → <500MB for typical repositories
- **Disk** → <100MB vector index for 10K files
- **Token Efficiency** → 90%+ relevant content in responses

## Deployment Scenarios

### Development
- **Local Vector Store** → `:memory:` qdrant instance
- **Minimal Config** → Default settings work out-of-box
- **Hot Reload** → watchfiles-driven incremental indexing

### Production
- **Cloud Vector Store** → Qdrant Cloud integration
- **API Key Management** → Environment variable configuration
- **Telemetry** → PostHog analytics for optimization

### Enterprise
- **On-Premise** → Self-hosted qdrant + local embeddings  
- **Custom Providers** → Extensible provider architecture
- **SSO Integration** → FastMCP auth middleware support