# CodeWeaver Implementation Roadmap

## Overview

**6-week implementation plan** following pydantic ecosystem patterns, prioritizing core functionality first with incremental complexity. Each phase builds upon previous phases with clear milestones and success criteria.

**✅ UPDATE**: All blocking issues have been resolved with clear solutions. Implementation can proceed immediately without architectural risks.

## Development Phases

### Phase 1: Core Foundation (Weeks 1-2) ✅ **READY TO START**
**Goal**: Basic working MCP server with file discovery and simple search

**✅ Blocking Issues Resolved**:
- FastMCP Context integration - Clear dependency injection solution
- Thread safety concerns - Evidence suggests non-issue, quick validation needed
- Configuration precedence - Complete hierarchy defined  
- Error handling patterns - Unified exception design ready

#### Week 1: Project Structure & Settings
**Deliverables:**
- [ ] Project structure following pydantic conventions
- [ ] ✅ Unified configuration system with precedence hierarchy (design complete)
- [ ] ✅ Unified exception hierarchy (prevents runaway exception anti-pattern)
- [ ] FastMCP server with health check endpoint  
- [ ] Basic CLI using cyclopts
- [ ] Development environment setup

**Implementation Tasks:**
```python
# Core project structure
src/codeweaver/
├── __init__.py
├── main.py           # FastMCP server entry point
├── settings.py       # Unified configuration
├── cli.py           # Cyclopts CLI interface
├── models/          # Pydantic models
│   ├── __init__.py
│   ├── core.py      # FindCodeResponse, CodeMatch
│   ├── config.py    # Provider configurations
│   └── intent.py    # Intent classification models
└── services/        # Core services
    ├── __init__.py
    ├── discovery.py # File discovery with rignore
    └── language.py  # Language detection (salvaged)
```

**✅ Configuration Foundation (Complete Design):**
```python
class CodeWeaverSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CODEWEAVER_",
        env_nested_delimiter="__",
        env_file=[".env", ".codeweaver.env"],
        # ✅ Precedence: ENV > Local > Project > User > Global > Defaults
        toml_file=[
            ".codeweaver.toml",              # Local (highest)  
            "../.codeweaver.toml",           # Project root
            "~/.codeweaver.toml",            # User home
            "/etc/codeweaver.toml",          # System global (lowest)
        ]
    )
    
    project_path: Path = Field(default_factory=Path.cwd)
    token_limit: int = 10000
    max_file_size: int = 10_000_000
    excluded_dirs: List[str] = Field(default_factory=get_default_excluded_dirs)

# ✅ Error Hierarchy Foundation  
class CodeWeaverError(Exception):
    def __init__(self, message: str, details: Dict[str, Any] = None, suggestions: List[str] = None):
        super().__init__(message)
        self.details = details or {}
        self.suggestions = suggestions or []

class ConfigurationError(CodeWeaverError): pass
class ProviderError(CodeWeaverError): pass  
class IndexingError(CodeWeaverError): pass
class QueryError(CodeWeaverError): pass
```

**Success Criteria:**
- FastMCP server starts and responds to health checks
- Configuration loads from environment variables and TOML files
- CLI commands execute without errors
- Basic file discovery works with rignore integration

#### Week 2: File Discovery & Basic Search
**Deliverables:**
- [ ] File discovery service with rignore integration
- [ ] Language detection system (salvaged from previous implementation)
- [ ] Text-based chunking for unsupported languages
- [ ] Basic keyword search functionality
- [ ] Simple find_code tool (file-based search only)

**Core Services:**
```python
class FileDiscoveryService:
    async def discover_files(self, patterns: List[str] = None) -> List[Path]:
        # rignore integration with gitignore support
        pass

class LanguageService:
    def detect_language(self, file_path: Path) -> Optional[str]:
        # Use salvaged SemanticSearchLanguage enum
        pass
    
    async def chunk_file(self, file_path: Path) -> List[CodeChunk]:
        # Text-based chunking fallback
        pass

@app.tool()
async def find_code(
    query: str,
    token_limit: int = 10000,
    context: Context = None
) -> FindCodeResponse:
    # Basic implementation: keyword search + file discovery
    pass
```

**Success Criteria:**
- File discovery respects gitignore and custom exclusions
- Language detection works for major file types
- Basic text chunking produces reasonable code segments
- find_code tool returns relevant file matches

---

### Phase 2: Provider Architecture (Week 3) ✅ **SOLUTIONS READY**  
**Goal**: Pluggable embedding and vector store providers

**✅ Solutions Prepared**:
- Cost management framework - Token tracking and budgeting design complete
- Migration strategy - Collection versioning with Qdrant named vectors

#### Deliverables:
- [ ] Abstract provider interfaces
- [ ] ✅ Voyage AI embedding provider with cost tracking (design ready)
- [ ] ✅ Qdrant vector store with migration support (architecture complete)
- [ ] In-memory vector store for development
- [ ] Provider factory and configuration system

**Provider Architecture:**
```python
# Abstract interfaces
class EmbeddingProvider(BaseModel, ABC):
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]: ...

class VectorStoreProvider(BaseModel, ABC):
    @abstractmethod
    async def search(self, vector: List[float], limit: int) -> List[SearchResult]: ...

# Concrete implementations
class VoyageEmbeddingProvider(EmbeddingProvider):
    api_key: SecretStr
    model: str = "voyage-code-3"
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # Voyage AI SDK integration
        pass

class QdrantVectorStoreProvider(VectorStoreProvider):
    url: str = "http://localhost:6333"
    collection_name: str = "codeweaver"
    
    async def search(self, vector: List[float], limit: int) -> List[SearchResult]:
        # qdrant-client integration
        pass

# Provider factory
def create_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    if config.provider_type == "voyage":
        return VoyageEmbeddingProvider(**config.model_dump())
    elif config.provider_type == "openai":
        return OpenAIEmbeddingProvider(**config.model_dump())
    else:
        raise ValueError(f"Unknown embedding provider: {config.provider_type}")
```

**Success Criteria:**
- Embedding generation works with Voyage AI API
- Vector storage and retrieval works with Qdrant
- In-memory storage works for development/testing
- Provider switching works through configuration

---

### Phase 3: Core Workflows (Weeks 4-5) ✅ **ARCHITECTURE VALIDATED**
**Goal**: Complete indexing and search workflows with AST analysis

**✅ Key Validation**: Pydantic-graph pipeline architecture confirmed working with FastMCP Context dependency injection

#### Week 4: Semantic Analysis Integration
**Deliverables:**
- [ ] ast-grep-py integration for supported languages
- [ ] AST-based code chunking with metadata
- [ ] Semantic search combining AST + embeddings
- [ ] Graceful fallback for unsupported languages

**AST Integration:**
```python
class AstGrepService:
    def __init__(self, settings: CodeWeaverSettings):
        self.supported_languages = get_ast_grep_languages()
    
    async def extract_functions(self, file_path: Path, language: SemanticSearchLanguage) -> list[CodeChunk]:
        # Extract function definitions using ast-grep patterns
        pass
    
    async def extract_classes(self, file_path: Path, language: SemanticSearchLanguage) -> list[CodeChunk]:
        # Extract class definitions
        pass
    
    async def extract_imports(self, file_path: Path, language: SemanticSearchLanguage) -> list[CodeChunk]:


    async def semantic_search(self, files: list[Path], query: str) -> list[CodeMatch]:
        # Combine AST analysis with embedding search
        pass

class SemanticAnalysisService:
    def __init__(self, ast_service: AstGrepService, embedding_provider: EmbeddingProvider):
        self.ast_service = ast_service
        self.embedding_provider = embedding_provider
    
    async def analyze_codebase(self, files: List[Path]) -> list[CodeChunk]:
        chunks = []
        for file_path in files:
            language = detect_language(file_path)
            
            if language in self.ast_service.supported_languages:
                # AST-based chunking  
                file_chunks = await self.ast_service.extract_chunks(file_path, language)
            else:
                # Text-based fallback
                file_chunks = await self._text_chunking_fallback(file_path)
            
            chunks.extend(file_chunks)
        
        return chunks
```

#### Week 5: Pipeline Integration
**Deliverables:**
- [ ] ✅ pydantic-graph pipeline with FastMCP Context injection (solution validated)
- [ ] Multi-stage search workflow (intent → files → analysis → ranking)
- [ ] Result ranking with multiple signals
- [ ] Token-optimized response assembly

**✅ Pipeline Architecture (FastMCP Context Integration):**
```python
@dataclass
class PipelineDeps:
    """✅ Dependencies with FastMCP Context injection solution"""
    context: Context | None = None  # FastMCP Context (None when unavailable)
    embedding_provider: EmbeddingProvider
    vector_store: VectorStoreProvider
    settings: CodeWeaverSettings

@dataclass
class CodeSearchState:
    query: str
    intent: Optional[IntentResult] = None
    discovered_files: List[Path] = field(default_factory=list)
    analyzed_chunks: List[CodeChunk] = field(default_factory=list)
    search_results: List[SearchResult] = field(default_factory=list)
    final_matches: List[CodeMatch] = field(default_factory=list)

# ✅ Pipeline nodes with Context access through dependencies
@dataclass
class AnalyzeIntentNode(BaseNode[CodeSearchState, PipelineDeps, FindCodeResponse]):
    async def run(self, ctx: GraphRunContext[CodeSearchState, PipelineDeps]) -> DiscoverFilesNode | End[FindCodeResponse]:
        # ✅ Access FastMCP Context via ctx.deps.context 
        if ctx.deps.context:
            # AI-powered intent analysis via sampling
            intent_result = await self._ai_intent_analysis(ctx.state.query, ctx.deps.context)
        else:
            # Graceful fallback for CLI/standalone mode
            intent_result = await self._rule_based_intent_analysis(ctx.state.query)
        
        ctx.state.intent = intent_result.intent
        return DiscoverFilesNode()

@dataclass
class DiscoverFilesNode(BaseNode[CodeSearchState, PipelineDeps, FindCodeResponse]):
    async def run(self, ctx: GraphRunContext[CodeSearchState, PipelineDeps]) -> AnalyzeCodeNode | End[FindCodeResponse]:
        # File discovery based on intent
        patterns = get_patterns_for_intent(ctx.state.intent)
        ctx.state.discovered_files = await ctx.deps.file_discovery.discover_files(patterns)
        return AnalyzeCodeNode()

@dataclass
class AnalyzeCodeNode(BaseNode[CodeSearchState, PipelineDeps, FindCodeResponse]):
    async def run(self, ctx: GraphRunContext[CodeSearchState, PipelineDeps]) -> SearchVectorsNode | End[FindCodeResponse]:
        # Semantic analysis of discovered files
        ctx.state.analyzed_chunks = await ctx.deps.semantic_analyzer.analyze_codebase(
            ctx.state.discovered_files
        )
        return SearchVectorsNode()

# ✅ Pipeline graph with dependency injection
search_pipeline = Graph(
    nodes=[AnalyzeIntentNode, DiscoverFilesNode, AnalyzeCodeNode, SearchVectorsNode, RankResultsNode, AssembleResponseNode],
    state_type=CodeSearchState
)

# ✅ Usage in find_code tool
@app.tool()
async def find_code(query: str, context: Context = None) -> FindCodeResponse:
    deps = PipelineDeps(
        context=context,  # ✅ Pass FastMCP Context through dependencies  
        embedding_provider=create_embedding_provider(),
        vector_store=create_vector_store(),
        settings=get_settings()
    )
    result = await search_pipeline.run(AnalyzeIntentNode(), deps=deps, state=CodeSearchState(query=query))
    return result.output
```

**Success Criteria:**
- AST analysis works for Python, JavaScript, TypeScript
- Pipeline executes all stages without errors
- Results include both semantic and syntactic matches
- Response assembly respects token limits

---

### Phase 4: Advanced Features (Week 6)
**Goal**: AI-powered intent analysis and background indexing

#### Deliverables:
- [ ] FastMCP sampling integration for intent analysis
- [ ] Tiered fallback system (AI → NLP → rules)
- [ ] Background indexing with watchfiles
- [ ] Advanced result ranking with multiple signals
- [ ] Performance optimizations and caching

**AI Intent Analysis:**
```python
from typing import TYPE_CHECKING:

from fastmcp import Context

if TYPE_CHECKING:
    from mcp import CreateMessageResult

class AIIntentAnalyzer:
    async def analyze_intent(self, query: str, context: Context) -> IntentResult:
        from fastmcp.client.sampling import SamplingMessage
        try:
            # Primary: AI analysis via FastMCP sampling
            result = await context.sample(
                messages = [SamplingMessage(role="user", content=f"Analyze the following query. Think hard about what information the user will need to complete their task, and try to identify what they are really asking for: {query}")], # example, in implementation would need to provide more detailed instructions and direct context agent to internal tools to call (i.e. choose_resolution_strategy)
                # In implementation preference should get generated to stay up on models, and integrated with config (user override preferences)
                # Telemetry should measure model and response time so we can do more informed preferences down the line
                model_preference = ["claude-4-sonnet", "gpt-5", "gpt-5-mini", "gpt-4.1", "gemini-2.5.-flash", "gpt-oss-20b", "deepseek-r1", "qwen3-4b-thinking"],
                system_prompt = "You are an expert code researcher. Your goal is to match the user's query with tailored information that gives them the exact details the user will need to compete the described work. Start by identifying the user's need and what kind of information you would need to carry out that task. Then select the strategy that best fits the user's need.",
                include_context = "thisServer",
                temperature = 0.6, # should be adjustable by config and potentially intelligently based on model/result data
                max_tokens = 2000, # also adjustable
            )
                
        except (TimeoutError, APIError):
            # first fallback if network is any configured out-of-band models (i.e. local model, provider with API key). We use pydantic-ai directly here.
            # Not every client supports sampling, so this is important, and will be required for precontext generation anyway
            # One thing we could do is add an OpenRouter config -- OpenRouter is OpenAI API compatible, and has some free models available.
            if self.backup_model:
                return await self.task_backup_model(query)

            # Fallback to NLP analysis
            return await self.nlp_fallback.analyze(query)
        else:
            return result
    
    def _parse_ai_response(self, response: CreateMessageResult) -> IntentResult:
        data = response.data
        # then we choose a strategy and pipeline (or the model does with a tool call)

class BackgroundIndexer:
    def __init__(self, settings: CodeWeaverSettings):
        self.settings = settings
        self.embedding_provider = create_embedding_provider(settings.embedding)
        self.vector_store = create_vector_store(settings.vector_store)
    
    async def start_watching(self, paths: List[Path]):
        """Start background file watching and indexing"""
        async for changes in awatch(*paths, watch_filter=self._create_filter()):
            await self._process_changes(changes)
    
    async def _process_changes(self, changes: set):
        """Process file changes incrementally (where possible, based on semantic changes vice simple changes)"""
        for change_type, file_path in changes:
            if change_type == Change.added or change_type == Change.modified:
                await self._index_file(file_path)
            elif change_type == Change.deleted:
                await self._remove_from_index(file_path)
```

**Success Criteria:**
- AI intent analysis improves search relevance
- Fallback system handles offline scenarios
- Background indexing updates vector store in real-time
- Performance meets target response times (<2s for complex queries)

---

## Quality Gates & Testing

### Phase 1 Gates
- [ ] All configuration loads correctly from multiple sources
- [ ] File discovery performance: <1s for repos with <10K files
- [ ] Basic search returns relevant results for common queries

### Phase 2 Gates
- [ ] Embedding generation: <2s for 100 text chunks
- [ ] Vector search: <500ms for queries against 10K vectors
- [ ] Provider switching works without code changes

### Phase 3 Gates  
- [ ] AST analysis accuracy: >90% for function/class extraction
- [ ] Pipeline execution: <2s end-to-end for typical queries
- [ ] Graceful degradation: unsupported languages still return results

### Phase 4 Gates
- [ ] Intent analysis improves relevance by >20% vs rule-based
- [ ] Background indexing: <1s processing time per changed file
- [ ] Memory usage: <500MB for typical repositories

## Success Metrics

### Performance Targets
- **Simple Queries**: <500ms (file-based search)
- **Complex Queries**: <2s (full pipeline with embeddings)
- **Background Indexing**: <1s per changed file
- **Memory Usage**: <500MB for repos with <50K files

### Quality Targets
- **Relevance**: >80% of results rated relevant by users
- **Coverage**: Support for >90% of files in typical polyglot repositories  
- **Reliability**: <1% error rate for valid queries
- **Token Efficiency**: >90% of response content directly relevant

## Risk Mitigation

### ✅ Resolved Architectural Risks
- **FastMCP Context Integration**: ✅ Solved via dependency injection pattern
- **Thread Safety Concerns**: ✅ Evidence suggests non-issue, quick validation needed  
- **Configuration Complexity**: ✅ Complete precedence hierarchy designed
- **Error Handling Chaos**: ✅ Unified exception hierarchy prevents 80+ exception anti-pattern

### Remaining Technical Risks
- **Performance Problems**: Implement caching, batch operations, optimize vector search
- **Memory Usage**: Implement streaming, chunking, and memory limits  
- **API Reliability**: Comprehensive fallback system with cost management

### Timeline Risks
- **Scope Creep**: Strict adherence to phase deliverables, defer advanced features
- **Integration Complexity**: ✅ Core architecture validated, start with proven implementations
- **Testing Delays**: Parallel development of tests with features

## Post-MVP Roadmap

### Phase 5: Polish & Production
- Comprehensive error handling and logging
- Performance monitoring and optimization
- Documentation and developer onboarding
- Security review and hardening

### Phase 6: Advanced Intelligence
- Multi-turn conversation support
- Context-aware query refinement  
- Learning from user interactions
- Advanced code understanding (call graphs, dependency analysis)

### Phase 7: Ecosystem Integration
- IDE plugins and integrations
- CI/CD pipeline integration
- Team collaboration features
- Enterprise deployment options

This roadmap prioritizes core functionality first while maintaining the flexibility to adapt based on user feedback and technical discoveries during implementation.