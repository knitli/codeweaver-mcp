<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# CodeWeaver Plugin Architecture Implementation Plan (Revised)

## Overview

This document provides a corrected implementation plan for transitioning CodeWeaver to a modular plugin architecture with proper FastMCP middleware integration. This revision addresses critical issues in the original plans and aligns with the actual FastMCP middleware API.

## Critical Corrections from Original Plans

### Issues Corrected:
1. **FastMCP Middleware API Alignment**: Uses actual FastMCP middleware hooks instead of custom framework
2. **Removed Migration/Compatibility Layers**: No backward compatibility needed for unreleased product
3. **Added FastMCP Logging**: Replaces existing logging with FastMCP's logging middleware
4. **Simplified Architecture**: Leverages existing plugin system without parallel middleware framework
5. **Rate Limiter Replacement**: Uses FastMCP's built-in rate limiting middleware

## Implementation Strategy

### Core Principles
1. **Direct FastMCP Integration**: Use FastMCP's native middleware system
2. **Plugin System Enhancement**: Extend existing plugin architecture, don't replace it
3. **Clean Slate Approach**: No migration layers or backward compatibility code
4. **FastMCP-First**: Leverage FastMCP's built-in capabilities before building custom solutions

## Phase 1: FastMCP Middleware Integration (1-2 weeks)

### 1.1 Replace Rate Limiter with FastMCP Rate Limiting

**Tasks**:
- [ ] Remove `src/codeweaver/rate_limiter.py` completely
- [ ] Update server initialization to use FastMCP's `RateLimitingMiddleware`
- [ ] Configure rate limiting through FastMCP middleware configuration
- [ ] Update configuration schema to use FastMCP rate limiting options
- [ ] Remove all imports and references to custom rate limiter

**Implementation**:
```python
# In main.py or server setup
from fastmcp.middleware import RateLimitingMiddleware

mcp = FastMCP("CodeWeaver")
mcp.add_middleware(RateLimitingMiddleware(
    requests_per_minute=60,
    algorithm="token_bucket"  # or "sliding_window"
))
```

### 1.2 Replace Logging with FastMCP Logging Middleware

**Tasks**:
- [ ] Remove custom logging setup from server initialization
- [ ] Add FastMCP `LoggingMiddleware` for request/response logging
- [ ] Configure structured logging through FastMCP configuration
- [ ] Update any custom log handlers to work with FastMCP's logging system
- [ ] Remove redundant logging code from tool implementations

**Implementation**:
```python
from fastmcp.middleware import LoggingMiddleware

mcp.add_middleware(LoggingMiddleware(
    level="INFO",
    format="structured",  # or "human"
    include_request_body=True,
    include_response_body=True
))
```

### 1.3 Create FastMCP Middleware for Cross-Cutting Concerns

**Tasks**:
- [ ] Create `ChunkingMiddleware` using FastMCP's `Middleware` base class
- [ ] Create `FilteringMiddleware` using FastMCP's middleware hooks
- [ ] Create `WatchingMiddleware` for file system monitoring (optional)
- [ ] Implement proper FastMCP middleware lifecycle methods
- [ ] Register middleware with FastMCP server

**FastMCP Middleware Pattern**:
```python
from fastmcp import Middleware, MiddlewareContext

class ChunkingMiddleware(Middleware):
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        # Pre-processing: extract chunking parameters
        if context.request.method == "tools/call" and context.request.params.name == "index_codebase":
            # Add chunking metadata to context
            context.metadata["chunking_enabled"] = True

        # Call next middleware/handler
        result = await call_next(context)

        # Post-processing: apply chunking if needed
        if context.metadata.get("chunking_enabled"):
            # Apply chunking logic here
            pass

        return result
```

## Phase 2: Legacy Component Integration (1-2 weeks)

### 2.1 Filesystem Source Enhancement

**Tasks**:
- [ ] Integrate chunking logic directly into `FilesystemSource`
- [ ] Add AST-grep search capabilities from `search.py`
- [ ] Integrate filtering logic from `file_filter.py`
- [ ] Use `rignore.walk()` for efficient file discovery
- [ ] Remove circular dependencies and duplicate code

**Enhanced Filesystem Source**:
```python
class FilesystemSource:
    """Enhanced filesystem source with integrated chunking and filtering."""

    def __init__(self, chunking_middleware: ChunkingMiddleware, filtering_middleware: FilteringMiddleware):
        self.chunking_middleware = chunking_middleware
        self.filtering_middleware = filtering_middleware

    async def index_content(self, path: Path) -> List[CodeChunk]:
        # Use filtering middleware for file discovery
        files = await self.filtering_middleware.find_files(path)

        # Use chunking middleware for content processing
        chunks = []
        for file_path in files:
            file_chunks = await self.chunking_middleware.chunk_file(file_path)
            chunks.extend(file_chunks)

        return chunks
```

### 2.2 Move Models to Pydantic in _types/

**Tasks**:
- [ ] Create Pydantic versions of `CodeChunk` and related models in `_types/content.py`
- [ ] Update all imports throughout codebase to use `_types` models
- [ ] Remove `src/codeweaver/models.py` completely
- [ ] Ensure serialization compatibility with existing data
- [ ] Update all type hints and annotations

**Pydantic Models**:
```python
# src/codeweaver/_types/content.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class CodeChunk(BaseModel):
    """Pydantic model for code chunks."""

    content: str = Field(description="The actual code content")
    file_path: str = Field(description="Path to source file")
    language: str = Field(description="Programming language")
    node_type: Optional[str] = Field(default=None, description="AST node type")
    start_line: Optional[int] = Field(default=None, description="Starting line number")
    end_line: Optional[int] = Field(default=None, description="Ending line number")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
```

### 2.3 Task Search Integration

**Tasks**:
- [ ] Keep `TaskSearchCoordinator` as-is (already well-designed)
- [ ] Add plugin registry integration for task search capabilities
- [ ] Extend configuration support for task search settings
- [ ] Update to use new Pydantic models from `_types/`

## Phase 3: Server Refactoring (2 weeks)

### 3.1 Clean Server Implementation

**Tasks**:
- [ ] Create new server implementation using pure plugin system
- [ ] Integrate FastMCP middleware stack properly
- [ ] Remove all legacy server code and compatibility layers
- [ ] Use factory pattern for component creation
- [ ] Implement configuration-driven initialization

**Clean Server Architecture**:
```python
class CodeWeaverServer:
    """Clean plugin-based MCP server implementation."""

    def __init__(self, config: CodeWeaverConfig):
        self.config = config
        self.mcp = FastMCP("Code Weaver")
        self.middleware_stack = []
        self.plugin_registry = ExtensibilityManager()

    async def initialize(self):
        """Initialize server with FastMCP middleware and plugins."""
        # Add FastMCP built-in middleware
        self.mcp.add_middleware(RateLimitingMiddleware(**config.rate_limiting))
        self.mcp.add_middleware(LoggingMiddleware(**config.logging))

        # Add custom middleware
        if config.chunking.enabled:
            self.mcp.add_middleware(ChunkingMiddleware(config.chunking))
        if config.filtering.enabled:
            self.mcp.add_middleware(FilteringMiddleware(config.filtering))

        # Initialize plugin system
        await self.plugin_registry.initialize(config)

        # Register MCP tools
        self._register_tools()

    def _register_tools(self):
        """Register MCP tools with FastMCP server."""

        @self.mcp.tool()
        async def index_codebase(ctx: Context, path: str) -> dict:
            # Use plugin system components
            source = self.plugin_registry.get_source("filesystem")
            backend = self.plugin_registry.get_backend("qdrant")
            provider = self.plugin_registry.get_provider("voyage")

            # Implementation using plugin system
            chunks = await source.index_content(Path(path))
            embeddings = await provider.embed_batch([c.content for c in chunks])
            await backend.store_vectors(embeddings)

            return {"indexed_chunks": len(chunks)}
```

### 3.2 Tool Migration

**Tasks**:
- [ ] Migrate all four MCP tools (`index_codebase`, `search_code`, `ast_grep_search`, `get_supported_languages`)
- [ ] Use plugin system exclusively for all tool implementations
- [ ] Ensure feature parity with current implementation
- [ ] Maintain exact same external interfaces
- [ ] Remove any legacy tool code

### 3.3 Legacy Code Removal

**Tasks**:
- [ ] Remove `src/codeweaver/chunker.py` (functionality moved to middleware)
- [ ] Remove `src/codeweaver/search.py` (integrated into filesystem source)
- [ ] Remove `src/codeweaver/file_filter.py` (functionality moved to middleware)
- [ ] Remove `src/codeweaver/file_watcher.py` (functionality moved to middleware)
- [ ] Remove `src/codeweaver/rate_limiter.py` (replaced with FastMCP middleware)
- [ ] Remove `src/codeweaver/models.py` (moved to `_types/`)
- [ ] Update all imports and remove dead code

## Phase 4: Configuration and Testing (1 week)

### 4.1 Configuration Schema Update

**Tasks**:
- [ ] Create new configuration schema for FastMCP middleware
- [ ] Remove all migration and compatibility configuration options
- [ ] Add FastMCP rate limiting and logging configuration
- [ ] Validate configuration schema with actual FastMCP requirements
- [ ] Update configuration documentation

**New Configuration Structure**:
```toml
[server]
name = "Code Weaver"
version = "1.0.0"

[middleware.rate_limiting]
enabled = true
requests_per_minute = 60
algorithm = "token_bucket"
burst_size = 10

[middleware.logging]
enabled = true
level = "INFO"
format = "structured"
include_request_body = true
include_response_body = false

[middleware.chunking]
enabled = true
max_chunk_size = 1500
min_chunk_size = 50
supported_languages = ["python", "rust", "typescript", "javascript"]
ast_grep_enabled = true

[middleware.filtering]
enabled = true
use_gitignore = true
max_file_size = "1MB"
excluded_dirs = ["node_modules", ".git", "__pycache__"]

[sources.filesystem]
chunking_enabled = true
filtering_enabled = true
ast_grep_search = true
semantic_search = true

[providers.voyage]
model = "voyage-code-3"
api_key_env = "CW_EMBEDDING_API_KEY"

[backends.qdrant]
url = "http://localhost:6333"
collection_name = "codeweaver"
```

### 4.2 Testing and Validation

**Tasks**:
- [ ] Create comprehensive test suite for FastMCP middleware integration
- [ ] Test all MCP tools with new architecture
- [ ] Validate FastMCP middleware functionality
- [ ] Performance testing with new middleware stack
- [ ] Integration testing for plugin system

**Testing Focus**:
- FastMCP middleware hooks work correctly
- Rate limiting functions as expected
- Logging captures request/response data
- All MCP tools maintain feature parity
- Plugin system components integrate properly

## Resource Requirements

### Development Team
- **Senior Developer**: FastMCP integration and architecture refactoring
- **Plugin Developer**: Middleware implementation and plugin system enhancement

### Timeline
- **Phase 1**: 1-2 weeks (FastMCP middleware integration)
- **Phase 2**: 1-2 weeks (Legacy component integration)
- **Phase 3**: 2 weeks (Server refactoring and cleanup)
- **Phase 4**: 1 week (Configuration and testing)

**Total Timeline**: 5-7 weeks

## Success Criteria

### Technical Requirements
- [ ] All functionality preserved with new architecture
- [ ] FastMCP middleware properly integrated
- [ ] No legacy compatibility code remains
- [ ] Performance maintained or improved
- [ ] Plugin system enhanced, not replaced

### Quality Gates
- [ ] All MCP tools function identically to before
- [ ] FastMCP rate limiting and logging work correctly
- [ ] Configuration system uses FastMCP patterns
- [ ] No circular dependencies or architectural issues
- [ ] Comprehensive test coverage (>90%)

### Final Validation
- [ ] Server runs with FastMCP middleware stack
- [ ] All legacy files removed from codebase
- [ ] Plugin system enhanced with new capabilities
- [ ] Documentation reflects new architecture
- [ ] Performance benchmarks met

This revised implementation plan provides a realistic and technically sound approach to transitioning CodeWeaver to a plugin architecture with proper FastMCP middleware integration, without the architectural misalignments and unnecessary complexity of the original plans.
