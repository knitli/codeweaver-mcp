# CodeWeaver API Design Specification

## Primary Tool Interface

### `find_code` Tool

**Single intelligent interface for codebase context discovery** - following FastAPI/FastMCP patterns with automatic validation and type safety.

```python
@app.tool()
async def find_code(
    query: str,
    intent: Optional[Literal["understand", "implement", "debug", "optimize", "test"]] = None,
    token_limit: int = 10000,
    include_tests: bool = True,
    focus_languages: Optional[List[str]] = None,
    context: Context = None
) -> FindCodeResponse:
    """
    Intelligently discover and retrieve relevant codebase context.
    
    Args:
        query: Natural language description of information needed
        intent: Optional hint about the type of task (auto-detected if None)
        token_limit: Maximum tokens to include in response
        include_tests: Whether to include test files in results  
        focus_languages: Limit search to specific programming languages
        context: FastMCP context (injected automatically)
        
    Returns:
        Structured response with relevant code matches and metadata
        
    Examples:
        query="authentication middleware setup"
        query="database connection pooling configuration" 
        query="user registration validation logic"
    """
```

## Response Models

### Primary Response Model

```python
class FindCodeResponse(BaseModel):
    """Structured response from find_code tool"""
    
    # Core results
    matches: List[CodeMatch] = Field(description="Relevant code matches ranked by relevance")
    summary: str = Field(description="High-level summary of findings")
    
    # Metadata
    query_intent: str = Field(description="Detected or specified intent")
    total_matches: int = Field(description="Total matches found before ranking")
    token_count: int = Field(description="Actual tokens used in response")
    execution_time_ms: float = Field(description="Total processing time")
    
    # Context information
    search_strategy: List[str] = Field(description="Search methods used")
    languages_found: List[str] = Field(description="Programming languages in results")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "matches": [...],
                "summary": "Found authentication middleware in 3 files...",
                "query_intent": "understand",
                "total_matches": 15,
                "token_count": 8543,
                "execution_time_ms": 1234.5
            }
        }
    )
```

### Code Match Model

```python
class CodeMatch(BaseModel):
    """Individual code match with context and metadata"""
    
    # File information
    file_path: Path = Field(description="Relative path to file from project root")
    language: Optional[str] = Field(description="Detected programming language")
    
    # Content
    content: str = Field(description="Relevant code content")
    line_range: Tuple[int, int] = Field(description="Start and end line numbers")
    
    # Relevance scoring
    relevance_score: float = Field(
        ge=0.0, le=1.0, 
        description="Relevance score (0.0-1.0)"
    )
    match_type: Literal["semantic", "syntactic", "keyword", "file_pattern"] = Field(
        description="Primary match method"
    )
    
    # Context
    surrounding_context: Optional[str] = Field(
        description="Additional context around the match"
    )
    related_symbols: List[str] = Field(
        default_factory=list,
        description="Related functions, classes, or symbols"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "file_path": "src/auth/middleware.py",
                "language": "python", 
                "content": "class AuthMiddleware(BaseMiddleware): ...",
                "line_range": [15, 45],
                "relevance_score": 0.92,
                "match_type": "semantic"
            }
        }
    )
```

## Configuration Models

### Core Settings Models

```python
class CodeWeaverSettings(BaseSettings):
    """Main configuration model following pydantic-settings patterns"""
    
    model_config = SettingsConfigDict(
        env_prefix="CODEWEAVER_",
        env_nested_delimiter="__",
        env_file=[".env", ".codeweaver.env"],
        toml_file=["pyproject.toml", ".codeweaver.toml"],
        case_sensitive=False,
        validate_assignment=True
    )
    
    # Project configuration
    project_path: Path = Field(
        default_factory=Path.cwd,
        description="Root path of the codebase to analyze"
    )
    project_name: Optional[str] = Field(
        default=None,
        description="Project name (auto-detected from directory if None)"
    )
    
    # Performance settings
    token_limit: int = Field(
        default=10000, gt=0, le=100000,
        description="Default maximum tokens per response"
    )
    max_file_size: int = Field(
        default=10_000_000,
        description="Maximum file size to process (bytes)"
    )
    max_results: int = Field(
        default=50, gt=0, le=500,
        description="Maximum code matches to return"
    )
    
    # File filtering
    excluded_dirs: List[str] = Field(
        default_factory=lambda: [
            "node_modules", ".git", "__pycache__", ".venv", "venv",
            "build", "dist", ".next", ".nuxt", "target"
        ],
        description="Directories to exclude from analysis"
    )
    excluded_extensions: List[str] = Field(
        default_factory=lambda: [
            ".log", ".tmp", ".cache", ".lock", ".pyc", ".pyo"
        ],
        description="File extensions to exclude"
    )
    
    # Provider configuration
    embedding: EmbeddingConfig = Field(
        default_factory=VoyageConfig,
        description="Embedding provider configuration"
    )
    vector_store: VectorStoreConfig = Field(
        default_factory=QdrantConfig,  
        description="Vector store provider configuration"
    )
    
    # Feature flags
    enable_background_indexing: bool = Field(
        default=True,
        description="Enable automatic background indexing"
    )
    enable_telemetry: bool = Field(
        default=True,
        description="Enable privacy-friendly usage telemetry"
    )
    enable_ai_intent_analysis: bool = Field(
        default=True,
        description="Enable AI-powered intent analysis via FastMCP sampling"
    )
```

### Provider Configuration Models

```python
class EmbeddingConfig(BaseModel, ABC):
    """Base configuration for embedding providers"""
    provider_type: str
    batch_size: int = Field(default=32, gt=0, le=128)
    timeout_seconds: float = Field(default=30.0, gt=0)

class VoyageConfig(EmbeddingConfig):
    """Voyage AI embedding configuration"""
    provider_type: Literal["voyage"] = "voyage"
    api_key: SecretStr = Field(description="Voyage AI API key")
    model: str = Field(
        default="voyage-code-3",
        description="Embedding model to use"
    )
    
    # Validation
    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        valid_models = ["voyage-code-3", "voyage-3", "voyage-3.5"]
        if v not in valid_models:
            raise ValueError(f"Model must be one of {valid_models}")
        return v

class QdrantConfig(BaseModel):
    """Qdrant vector store configuration"""
    provider_type: Literal["qdrant"] = "qdrant"
    url: str = Field(default="http://localhost:6333")
    api_key: Optional[SecretStr] = None
    collection_name: str = Field(default="codeweaver")
    
    # Vector configuration
    vector_size: int = Field(default=1536, description="Embedding vector size")
    distance_metric: Literal["cosine", "dot", "euclidean"] = "cosine"
    
    # Performance settings
    batch_size: int = Field(default=100, gt=0, le=1000)
    timeout_seconds: float = Field(default=30.0, gt=0)
```

## Intent Classification

### Intent Types

```python
class QueryIntent(BaseModel):
    """Classified query intent with confidence scoring"""
    
    intent_type: Literal[
        "understand",    # Code comprehension, explanation
        "implement",     # Feature implementation, code generation  
        "debug",        # Bug investigation, error analysis
        "optimize",     # Performance improvement, refactoring
        "test",         # Testing, validation, coverage
        "configure",    # Setup, configuration, deployment
        "document"      # Documentation, comments, explanations
    ]
    
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Why this intent was detected")
    
    # Intent-specific parameters
    focus_areas: List[str] = Field(
        default_factory=list,
        description="Specific areas of focus within the intent"
    )
    complexity_level: Literal["simple", "moderate", "complex"] = "moderate"
```

### Intent-Specific Response Strategies

```python
class IntentStrategy(BaseModel):
    """Strategy configuration for different intent types"""
    
    # File discovery strategy
    file_patterns: List[str] = Field(description="File patterns to prioritize")
    exclude_patterns: List[str] = Field(default_factory=list)
    
    # Search strategy  
    semantic_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    syntactic_weight: float = Field(default=0.3, ge=0.0, le=1.0) 
    keyword_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    
    # Response formatting
    include_context: bool = True
    max_matches_per_file: int = Field(default=3, gt=0)
    prioritize_entry_points: bool = False

# Intent-specific strategies
INTENT_STRATEGIES = {
    "understand": IntentStrategy(
        file_patterns=["*.md", "*.rst", "*.py", "*.js", "*.ts"],
        semantic_weight=0.7,
        include_context=True
    ),
    "implement": IntentStrategy( 
        file_patterns=["*.py", "*.js", "*.ts", "*.go", "*.rs"],
        syntactic_weight=0.5,
        prioritize_entry_points=True
    ),
    "debug": IntentStrategy(
        file_patterns=["*.log", "*.py", "*.js", "*.ts"],
        keyword_weight=0.3,
        max_matches_per_file=5
    )
}
```

## Error Models

### Structured Error Responses

```python
class CodeWeaverError(BaseModel):
    """Structured error information"""
    
    error_type: Literal[
        "configuration_error",
        "provider_error", 
        "indexing_error",
        "validation_error",
        "timeout_error"
    ]
    message: str
    details: Optional[Dict[str, Any]] = None
    suggestions: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error_type": "provider_error",
                "message": "Voyage AI API key not configured",
                "suggestions": [
                    "Set CODEWEAVER_EMBEDDING__API_KEY environment variable",
                    "Add api_key to .codeweaver.toml configuration file"
                ]
            }
        }
    )
```

## CLI Interface

### Command Structure

```python
# Using cyclopts for CLI
@app.command  
async def server(
    *,
    config_file: Annotated[
        Optional[Path], 
        Parameter(name=["--config", "-c"])
    ] = None,
    project_path: Annotated[
        Optional[Path],
        Parameter(name=["--project", "-p"])
    ] = None,
    host: str = "localhost",
    port: int = 8080,
    debug: bool = False
) -> None:
    """Start CodeWeaver MCP server"""

@app.command
async def index(
    project_path: Annotated[
        Optional[Path],
        Parameter(name=["--project", "-p"])
    ] = None,
    *,
    force_reindex: bool = False,
    dry_run: bool = False
) -> None:
    """Build or rebuild the project index"""

@app.command
async def search(
    query: str,
    *,
    intent: Optional[str] = None,
    limit: int = 10,
    format: Literal["json", "table", "markdown"] = "table"
) -> None:  
    """Search codebase from command line"""
```

## Type Safety & Validation

### Runtime Validation

```python
# All models include comprehensive validation
class CodeMatch(BaseModel):
    # ... fields ...
    
    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: Path) -> Path:
        if not v.is_absolute():
            # Convert to absolute path during validation
            return v.resolve()
        return v
    
    @field_validator("relevance_score") 
    @classmethod
    def validate_score(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Relevance score must be between 0.0 and 1.0")
        return v
    
    @model_validator(mode="after")
    def validate_line_range(self) -> "CodeMatch":
        start, end = self.line_range
        if start > end:
            raise ValueError("Start line must be <= end line")
        if start < 1:
            raise ValueError("Line numbers must start from 1")
        return self
```

### JSON Schema Generation

All models automatically generate OpenAPI-compatible JSON schemas for:
- **MCP Tool Registration**: Automatic parameter validation
- **CLI Help Generation**: Type-aware help text
- **Documentation**: API reference generation
- **Client SDKs**: Type-safe client generation

## Backward Compatibility

### Version Management  
- **API Versioning**: Semantic versioning for breaking changes
- **Configuration Migration**: Automatic config upgrade utilities  
- **Deprecation Warnings**: Gradual migration paths for breaking changes

### Legacy Support
- **v1 Configuration**: Migration utilities for existing setups
- **Provider Interfaces**: Stable provider contracts with version negotiation