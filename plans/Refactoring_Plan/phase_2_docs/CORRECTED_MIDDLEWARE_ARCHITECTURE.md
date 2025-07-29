<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# CodeWeaver FastMCP Middleware Architecture (Corrected)

## Overview

This document provides the corrected architecture for integrating CodeWeaver with FastMCP's native middleware system. The original middleware design proposed a custom framework that didn't align with FastMCP's actual API. This corrected version uses FastMCP's built-in middleware capabilities properly.

## FastMCP Middleware System Integration

### Core FastMCP Middleware Pattern

FastMCP middleware uses a hook-based system with `MiddlewareContext` and `call_next` patterns:

```python
from fastmcp import Middleware, MiddlewareContext

class CodeWeaverMiddleware(Middleware):
    async def on_message(self, context: MiddlewareContext, call_next):
        # Pre-processing
        result = await call_next(context)
        # Post-processing
        return result

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        # Tool-specific middleware logic
        result = await call_next(context)
        return result
```

### Available FastMCP Hooks

- **`on_message`**: All MCP messages (broadest scope)
- **`on_request`**: Request messages only
- **`on_call_tool`**: Tool execution requests
- **`on_read_resource`**: Resource read requests
- **`on_list_tools`**: Tool listing requests

## Corrected Middleware Implementations

### 1. Chunking Middleware (FastMCP-Compliant)

```python
from fastmcp import Middleware, MiddlewareContext
from pathlib import Path
from typing import Dict, Any, List
import ast_grep_py as ag
from codeweaver.types import CodeChunk

class ChunkingMiddleware(Middleware):
    """FastMCP middleware for code chunking services."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.max_chunk_size = self.config.get("max_chunk_size", 1500)
        self.min_chunk_size = self.config.get("min_chunk_size", 50)
        self.ast_grep_enabled = self.config.get("ast_grep_enabled", True)

        # Language mappings
        self.SUPPORTED_LANGUAGES = {
            ".py": "python", ".rs": "rust", ".ts": "typescript",
            ".js": "javascript", ".tsx": "tsx", ".jsx": "jsx",
            ".go": "go", ".java": "java", ".cpp": "cpp", ".c": "c"
        }

        # AST-grep patterns for semantic chunking
        self.CHUNK_PATTERNS = {
            "python": ["function_definition", "class_definition"],
            "rust": ["function_item", "struct_item", "impl_item"],
            "typescript": ["function_declaration", "class_declaration", "interface_declaration"],
            "javascript": ["function_declaration", "class_declaration"]
        }

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        """Handle tool calls that need chunking services."""

        # Check if this is an indexing operation that needs chunking
        if (context.request.method == "tools/call" and
            context.request.params.name == "index_codebase"):

            # Add chunking service to context for the tool to use
            context.metadata["chunking_service"] = self

        # Continue with normal tool execution
        result = await call_next(context)

        return result

    async def chunk_file(self, file_path: Path, content: str) -> List[CodeChunk]:
        """Chunk file content using AST-grep or fallback methods."""
        language = self._detect_language(file_path)

        if self.ast_grep_enabled and language in self.CHUNK_PATTERNS:
            return await self._chunk_with_ast_grep(content, language, file_path)
        else:
            return await self._chunk_with_fallback(content, file_path)

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        suffix = file_path.suffix.lower()
        return self.SUPPORTED_LANGUAGES.get(suffix, "unknown")

    async def _chunk_with_ast_grep(self, content: str, language: str, file_path: Path) -> List[CodeChunk]:
        """Chunk content using AST-grep patterns."""
        try:
            root = ag.SgRoot(content, language)
            patterns = self.CHUNK_PATTERNS[language]
            chunks = []

            for pattern in patterns:
                matches = root.root().find_all(pattern)
                for match in matches:
                    chunk_content = match.text()
                    if self.min_chunk_size <= len(chunk_content) <= self.max_chunk_size:
                        chunks.append(CodeChunk(
                            content=chunk_content,
                            file_path=str(file_path),
                            language=language,
                            node_type=pattern,
                            start_line=match.start_pos().line,
                            end_line=match.end_pos().line
                        ))

            return chunks

        except Exception:
            # Fall back to simple chunking
            return await self._chunk_with_fallback(content, file_path)

    async def _chunk_with_fallback(self, content: str, file_path: Path) -> List[CodeChunk]:
        """Fallback chunking using simple line-based approach."""
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_size = 0

        for line in lines:
            line_size = len(line) + 1  # +1 for newline

            if current_size + line_size > self.max_chunk_size and current_chunk:
                chunk_content = '\n'.join(current_chunk)
                if len(chunk_content) >= self.min_chunk_size:
                    chunks.append(CodeChunk(
                        content=chunk_content,
                        file_path=str(file_path),
                        language="unknown",
                        node_type="fallback_chunk"
                    ))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size

        # Handle remaining chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            if len(chunk_content) >= self.min_chunk_size:
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_path=str(file_path),
                    language="unknown",
                    node_type="fallback_chunk"
                ))

        return chunks
```

### 2. File Filtering Middleware (FastMCP-Compliant)

```python
from fastmcp import Middleware, MiddlewareContext
from pathlib import Path
from typing import List, Set, Optional
import rignore

class FileFilteringMiddleware(Middleware):
    """FastMCP middleware for file filtering services."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.use_gitignore = self.config.get("use_gitignore", True)
        self.max_file_size = self._parse_size(self.config.get("max_file_size", "1MB"))
        self.excluded_dirs = set(self.config.get("excluded_dirs",
                                               ["node_modules", ".git", "__pycache__"]))

        extensions = self.config.get("included_extensions")
        self.included_extensions = set(extensions) if extensions else None

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        """Handle tool calls that need filtering services."""

        # Check if this is an operation that needs file filtering
        if (context.request.method == "tools/call" and
            context.request.params.name in ["index_codebase", "search_code"]):

            # Add filtering service to context
            context.metadata["filtering_service"] = self

        result = await call_next(context)
        return result

    async def find_files(self, base_path: Path, patterns: List[str] = None) -> List[Path]:
        """Find files using rignore.walk() with filtering criteria."""
        patterns = patterns or ["*"]
        found_files = []

        try:
            # Use rignore.walk() idiomatically
            walker = rignore.walk(str(base_path))

            for entry in walker:
                if entry.is_file():
                    file_path = Path(entry.path)

                    # Apply filtering criteria
                    if (await self._should_include_file(file_path, base_path) and
                        self._matches_patterns(file_path, patterns)):
                        found_files.append(file_path)

        except Exception as e:
            # Log error but don't fail completely
            print(f"File discovery error: {e}")

        return found_files

    async def _should_include_file(self, file_path: Path, base_path: Path) -> bool:
        """Check if file should be included based on filtering criteria."""
        try:
            # Check file size
            if file_path.stat().st_size > self.max_file_size:
                return False
        except (OSError, FileNotFoundError):
            return False

        # Check excluded directories
        for part in file_path.relative_to(base_path).parts:
            if part in self.excluded_dirs:
                return False

        # Check included extensions
        if self.included_extensions:
            if file_path.suffix.lower() not in self.included_extensions:
                return False

        return True

    def _matches_patterns(self, file_path: Path, patterns: List[str]) -> bool:
        """Check if file matches any of the given patterns."""
        import fnmatch

        for pattern in patterns:
            if fnmatch.fnmatch(file_path.name, pattern):
                return True
            if fnmatch.fnmatch(str(file_path), pattern):
                return True

        return False

    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '1MB' to bytes."""
        if isinstance(size_str, int):
            return size_str

        size_str = size_str.upper()
        multipliers = {
            'B': 1, 'KB': 1024, 'MB': 1024 * 1024, 'GB': 1024 * 1024 * 1024
        }

        for suffix, multiplier in multipliers.items():
            if size_str.endswith(suffix):
                return int(float(size_str[:-len(suffix)]) * multiplier)

        return int(size_str)  # Assume bytes if no suffix
```

### 3. Performance Monitoring Middleware (FastMCP-Compliant)

```python
from fastmcp import Middleware, MiddlewareContext
import time
import logging

class PerformanceMonitoringMiddleware(Middleware):
    """FastMCP middleware for performance monitoring."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        """Monitor tool execution performance."""
        tool_name = context.request.params.name
        start_time = time.time()

        try:
            result = await call_next(context)
            duration = time.time() - start_time

            self.logger.info(
                "Tool execution completed",
                extra={
                    "tool_name": tool_name,
                    "duration_seconds": duration,
                    "status": "success"
                }
            )

            return result

        except Exception as e:
            duration = time.time() - start_time

            self.logger.error(
                "Tool execution failed",
                extra={
                    "tool_name": tool_name,
                    "duration_seconds": duration,
                    "status": "error",
                    "error": str(e)
                }
            )

            raise  # Re-raise the exception
```

## Server Integration with FastMCP Middleware

### Proper FastMCP Server Setup

```python
from fastmcp import FastMCP
from fastmcp.middleware import RateLimitingMiddleware, LoggingMiddleware

class CodeWeaverServer:
    """CodeWeaver server with proper FastMCP middleware integration."""

    def __init__(self, config: CodeWeaverConfig):
        self.config = config
        self.mcp = FastMCP("Code Weaver")
        self.plugin_registry = ExtensibilityManager()

    async def initialize(self):
        """Initialize server with FastMCP middleware stack."""

        # Add FastMCP built-in middleware
        if self.config.middleware.rate_limiting.enabled:
            self.mcp.add_middleware(RateLimitingMiddleware(
                requests_per_minute=self.config.middleware.rate_limiting.requests_per_minute,
                algorithm=self.config.middleware.rate_limiting.algorithm
            ))

        if self.config.middleware.logging.enabled:
            self.mcp.add_middleware(LoggingMiddleware(
                level=self.config.middleware.logging.level,
                format=self.config.middleware.logging.format,
                include_request_body=self.config.middleware.logging.include_request_body,
                include_response_body=self.config.middleware.logging.include_response_body
            ))

        # Add custom CodeWeaver middleware
        if self.config.middleware.chunking.enabled:
            self.mcp.add_middleware(ChunkingMiddleware(self.config.middleware.chunking.model_dump()))

        if self.config.middleware.filtering.enabled:
            self.mcp.add_middleware(FileFilteringMiddleware(self.config.middleware.filtering.model_dump()))

        # Add performance monitoring
        self.mcp.add_middleware(PerformanceMonitoringMiddleware())

        # Initialize plugin system
        await self.plugin_registry.initialize(self.config)

        # Register MCP tools
        self._register_tools()

    def _register_tools(self):
        """Register MCP tools that can access middleware services."""

        @self.mcp.tool()
        async def index_codebase(ctx: Context, path: str) -> dict:
            """Index a codebase using middleware services."""

            # Access middleware services from context
            chunking_service = ctx.metadata.get("chunking_service")
            filtering_service = ctx.metadata.get("filtering_service")

            # Use plugin system components
            source = self.plugin_registry.get_source("filesystem")
            backend = self.plugin_registry.get_backend("qdrant")
            provider = self.plugin_registry.get_provider("voyage")

            # Find files using filtering service
            base_path = Path(path)
            if filtering_service:
                files = await filtering_service.find_files(base_path)
            else:
                files = list(base_path.rglob("*.py"))  # Fallback

            # Process files using chunking service
            all_chunks = []
            for file_path in files:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    if chunking_service:
                        chunks = await chunking_service.chunk_file(file_path, content)
                    else:
                        # Fallback chunking
                        chunks = [CodeChunk(content=content, file_path=str(file_path), language="unknown")]
                    all_chunks.extend(chunks)
                except Exception as e:
                    ctx.logger.warning(f"Failed to process file {file_path}: {e}")
                    continue

            # Generate embeddings and store
            if all_chunks:
                embeddings = await provider.embed_batch([chunk.content for chunk in all_chunks])
                await backend.store_vectors(embeddings)

            return {
                "indexed_files": len(files),
                "indexed_chunks": len(all_chunks),
                "success": True
            }
```

## Configuration Schema for FastMCP Integration

```toml
[middleware.rate_limiting]
enabled = true
requests_per_minute = 60
algorithm = "token_bucket"  # or "sliding_window"
burst_size = 10

[middleware.logging]
enabled = true
level = "INFO"
format = "structured"  # or "human"
include_request_body = true
include_response_body = false

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
included_extensions = [".py", ".rs", ".ts", ".js", ".tsx", ".jsx"]
```

## Key Architectural Corrections

### 1. Proper FastMCP Hook Usage
- Uses `on_call_tool`, `on_message`, etc. instead of custom request/response patterns
- Leverages `MiddlewareContext` and `call_next` properly
- Follows FastMCP's middleware execution order

### 2. Service Injection Pattern
- Middleware adds services to context metadata
- Tools access services through context rather than direct instantiation
- Clean separation between middleware services and tool logic

### 3. Built-in FastMCP Middleware
- Uses FastMCP's `RateLimitingMiddleware` instead of custom implementation
- Uses FastMCP's `LoggingMiddleware` for request/response logging
- Leverages FastMCP's configuration patterns

### 4. No Custom Middleware Framework
- Removed the proposed `MiddlewareRequest`/`MiddlewareResponse` pattern
- Removed custom middleware registry and processing pipeline
- Uses FastMCP's native middleware system exclusively

## Benefits of Corrected Architecture

1. **API Alignment**: Properly uses FastMCP's actual middleware API
2. **Built-in Features**: Leverages FastMCP's rate limiting and logging
3. **Simpler Integration**: No custom middleware framework to maintain
4. **Better Performance**: Uses FastMCP's optimized middleware pipeline
5. **Future Compatibility**: Aligns with FastMCP's evolution and updates

This corrected architecture provides proper FastMCP middleware integration while maintaining the functionality needed for CodeWeaver's plugin system.
