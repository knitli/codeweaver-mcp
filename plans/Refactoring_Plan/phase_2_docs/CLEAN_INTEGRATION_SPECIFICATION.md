<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# CodeWeaver Clean Integration Specification

## Executive Summary

This specification provides a clean, migration-free approach to integrating CodeWeaver's components into the existing plugin architecture with proper FastMCP middleware integration. This replaces the original specification which included unnecessary migration layers and architectural misalignments.

## Core Architecture Strategy

### Principles
1. **No Migration Layers**: Direct integration without backward compatibility code
2. **FastMCP Native**: Use FastMCP's actual middleware API, not custom frameworks
3. **Plugin System Enhancement**: Extend existing architecture, don't replace it
4. **Clean Slate Approach**: Remove legacy components completely after integration

### Integration Classification (Revised)

| Component | Action | Target Location | Implementation Method |
|-----------|--------|-----------------|----------------------|
| `chunker.py` | **INTEGRATE** | FastMCP Middleware | Convert to FastMCP middleware hooks |
| `search.py` | **MERGE** | `sources/filesystem.py` | Integrate AST-grep into filesystem source |
| `file_filter.py` | **INTEGRATE** | FastMCP Middleware | Convert to FastMCP middleware hooks |
| `file_watcher.py` | **INTEGRATE** | FastMCP Middleware | Convert to FastMCP middleware hooks |
| `models.py` | **MIGRATE** | `_types/content.py` | Convert to Pydantic models |
| `rate_limiter.py` | **REMOVE** | N/A | Use FastMCP's `RateLimitingMiddleware` |
| `task_search.py` | **ENHANCE** | Keep current location | Add plugin system integration |
| `server.py` | **REWRITE** | `server.py` | Clean implementation using plugin system |

## FastMCP Middleware Integration

### 1. Chunking Service (FastMCP Middleware)

**Approach**: Convert chunking functionality to FastMCP middleware using proper hooks

```python
from fastmcp import Middleware, MiddlewareContext

class ChunkingMiddleware(Middleware):
    """FastMCP middleware providing code chunking services."""
    
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        # Add chunking service to context for tools that need it
        if self._needs_chunking(context):
            context.metadata["chunking_service"] = self
        
        result = await call_next(context)
        return result
    
    def _needs_chunking(self, context: MiddlewareContext) -> bool:
        """Check if this tool call needs chunking services."""
        return (context.request.method == "tools/call" and 
                context.request.params.name in ["index_codebase"])
    
    async def chunk_file(self, file_path: Path, content: str) -> List[CodeChunk]:
        """Public API for chunking files."""
        # Implementation moved from chunker.py
        pass
```

### 2. File Filtering Service (FastMCP Middleware)

**Approach**: Convert file filtering to FastMCP middleware with service injection

```python
class FileFilteringMiddleware(Middleware):
    """FastMCP middleware providing file filtering services."""
    
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        # Add filtering service to context
        if self._needs_filtering(context):
            context.metadata["filtering_service"] = self
        
        result = await call_next(context)
        return result
    
    async def find_files(self, base_path: Path) -> List[Path]:
        """Public API for finding filtered files using rignore.walk()."""
        # Implementation moved from file_filter.py
        pass
```

### 3. Built-in FastMCP Services

**Rate Limiting**: Replace custom rate limiter with FastMCP's `RateLimitingMiddleware`
**Logging**: Use FastMCP's `LoggingMiddleware` for request/response logging

```python
from fastmcp.middleware import RateLimitingMiddleware, LoggingMiddleware

# Server setup
mcp.add_middleware(RateLimitingMiddleware(
    requests_per_minute=60,
    algorithm="token_bucket"
))

mcp.add_middleware(LoggingMiddleware(
    level="INFO",
    format="structured",
    include_request_body=True
))
```

## Component Integration Details

### Filesystem Source Enhancement

**Strategy**: Enhance existing filesystem source to use middleware services

```python
class FilesystemSource:
    """Enhanced filesystem source using middleware services."""
    
    async def index_content(self, path: Path, context: Optional[Dict] = None) -> List[CodeChunk]:
        """Index content using middleware services from context."""
        
        # Get services from FastMCP context if available
        filtering_service = context.get("filtering_service") if context else None
        chunking_service = context.get("chunking_service") if context else None
        
        # Find files using filtering service or fallback
        if filtering_service:
            files = await filtering_service.find_files(path)
        else:
            files = self._fallback_file_discovery(path)
        
        # Process files using chunking service or fallback
        all_chunks = []
        for file_path in files:
            content = file_path.read_text(encoding='utf-8')
            
            if chunking_service:
                chunks = await chunking_service.chunk_file(file_path, content)
            else:
                chunks = self._fallback_chunking(file_path, content)
                
            all_chunks.extend(chunks)
        
        return all_chunks
```

### Models Migration to Pydantic

**Strategy**: Convert dataclass models to Pydantic in `_types/content.py`

```python
# src/codeweaver/_types/content.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class CodeChunk(BaseModel):
    """Pydantic model replacing dataclass version."""
    
    content: str = Field(description="Code content")
    file_path: str = Field(description="Source file path")
    language: str = Field(description="Programming language")
    node_type: Optional[str] = Field(default=None, description="AST node type")
    start_line: Optional[int] = Field(default=None, description="Start line")
    end_line: Optional[int] = Field(default=None, description="End line")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        extra = "allow"  # Allow additional fields for extensibility
```

### Task Search Enhancement

**Strategy**: Keep current implementation, add plugin system integration

```python
class TaskSearchCoordinator:
    """Enhanced task search with plugin system integration."""
    
    def __init__(self, plugin_registry: ExtensibilityManager):
        self.plugin_registry = plugin_registry
        # Keep existing functionality
        
    async def search_tasks(self, query: str, context: Optional[Dict] = None) -> List[Task]:
        """Search tasks using plugin system components."""
        # Use plugin registry to access providers and backends
        # Keep existing search logic
        pass
```

## Clean Server Implementation

### Server Architecture (No Legacy Code)

```python
from fastmcp import FastMCP, Context
from fastmcp.middleware import RateLimitingMiddleware, LoggingMiddleware

class CodeWeaverServer:
    """Clean server implementation using plugin system exclusively."""
    
    def __init__(self, config: CodeWeaverConfig):
        self.config = config
        self.mcp = FastMCP("Code Weaver")
        self.plugin_registry = ExtensibilityManager()
    
    async def initialize(self):
        """Initialize with FastMCP middleware and plugin system."""
        
        # Add FastMCP built-in middleware
        self.mcp.add_middleware(RateLimitingMiddleware(
            requests_per_minute=self.config.rate_limiting.requests_per_minute
        ))
        
        self.mcp.add_middleware(LoggingMiddleware(
            level=self.config.logging.level,
            format=self.config.logging.format
        ))
        
        # Add custom middleware
        self.mcp.add_middleware(ChunkingMiddleware(self.config.chunking))
        self.mcp.add_middleware(FileFilteringMiddleware(self.config.filtering))
        
        # Initialize plugin system
        await self.plugin_registry.initialize(self.config)
        
        # Register tools
        self._register_mcp_tools()
    
    def _register_mcp_tools(self):
        """Register MCP tools using plugin system."""
        
        @self.mcp.tool()
        async def index_codebase(ctx: Context, path: str) -> dict:
            """Index codebase using middleware services and plugin system."""
            
            # Get components from plugin system
            source = self.plugin_registry.get_source("filesystem")
            backend = self.plugin_registry.get_backend("qdrant")
            provider = self.plugin_registry.get_provider("voyage")
            
            # Index using middleware services (available in ctx.metadata)
            chunks = await source.index_content(Path(path), ctx.metadata)
            embeddings = await provider.embed_batch([c.content for c in chunks])
            await backend.store_vectors(embeddings)
            
            return {"indexed_chunks": len(chunks)}
        
        @self.mcp.tool()
        async def search_code(ctx: Context, query: str, limit: int = 10) -> dict:
            """Search code using plugin system components."""
            
            # Get components
            backend = self.plugin_registry.get_backend("qdrant")
            provider = self.plugin_registry.get_provider("voyage")
            
            # Search implementation
            query_embedding = await provider.embed_query(query)
            results = await backend.search_vectors(query_embedding, limit)
            
            return {"results": [r.model_dump() for r in results]}
        
        # Register other tools similarly...
```

## Configuration Schema (Simplified)

```toml
[server]
name = "Code Weaver"
version = "1.0.0"

# FastMCP built-in middleware configuration
[middleware.rate_limiting]
enabled = true
requests_per_minute = 60
algorithm = "token_bucket"

[middleware.logging]
enabled = true
level = "INFO"
format = "structured"
include_request_body = true

# Custom middleware configuration  
[middleware.chunking]
enabled = true
max_chunk_size = 1500
min_chunk_size = 50
ast_grep_enabled = true

[middleware.filtering]
enabled = true
use_gitignore = true
max_file_size = "1MB"
excluded_dirs = ["node_modules", ".git", "__pycache__"]

# Plugin system configuration (existing)
[providers.voyage]
model = "voyage-code-3"
api_key_env = "CW_EMBEDDING_API_KEY"

[backends.qdrant]
url = "http://localhost:6333"
collection_name = "codeweaver"

[sources.filesystem]
enabled = true
```

## Implementation Phases (Simplified)

### Phase 1: FastMCP Integration (1 week)
- Replace rate limiter with FastMCP middleware
- Add FastMCP logging middleware
- Create chunking and filtering middleware using FastMCP hooks

### Phase 2: Component Integration (1 week)
- Move models to Pydantic in `_types/`
- Enhance filesystem source to use middleware services
- Integrate AST-grep search into filesystem source
- Enhance task search with plugin system

### Phase 3: Server Cleanup (1 week)
- Rewrite server using clean plugin system architecture
- Remove all legacy files (`chunker.py`, `search.py`, `file_filter.py`, `rate_limiter.py`, `models.py`)
- Update tool implementations to use plugin system exclusively

### Phase 4: Testing and Validation (3 days)
- Test FastMCP middleware integration
- Validate all tools work with new architecture
- Performance testing and optimization

**Total Timeline**: 3-4 weeks (significantly reduced from original 5-8 weeks)

## Success Criteria

### Technical Validation
- [ ] All MCP tools maintain identical external interfaces
- [ ] FastMCP middleware works correctly (rate limiting, logging, custom middleware)
- [ ] Plugin system enhanced with middleware services
- [ ] No legacy code remains in codebase
- [ ] Performance maintained or improved

### Architecture Validation
- [ ] Server uses FastMCP's native middleware system
- [ ] Plugin system components work with middleware services
- [ ] Configuration system simplified and FastMCP-aligned
- [ ] No migration or compatibility layers exist

### Quality Validation
- [ ] Comprehensive test coverage for new architecture
- [ ] All integration tests pass
- [ ] Performance benchmarks met
- [ ] Documentation reflects clean architecture

## Benefits of Clean Integration

1. **Simplified Architecture**: No custom middleware framework to maintain
2. **FastMCP Alignment**: Uses actual FastMCP middleware API correctly
3. **Reduced Complexity**: Eliminates migration layers and compatibility code
4. **Better Performance**: Leverages FastMCP's optimized middleware pipeline
5. **Future-Proof**: Aligns with FastMCP's evolution and updates
6. **Faster Implementation**: 3-4 weeks instead of 5-8 weeks

This clean integration specification provides a practical, technically sound approach to modernizing CodeWeaver's architecture without the unnecessary complexity and architectural misalignments of the original plans.