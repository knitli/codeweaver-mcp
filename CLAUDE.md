<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CodeWeaver is a Model Context Protocol (MCP) server that provides semantic code search powered by Voyage AI embeddings and ast-grep structural search. It enables AI assistants to understand and navigate codebases using natural language queries and precise structural patterns.

**Key Technologies:**
- **Voyage AI**: Best-in-class code embeddings (`voyage-code-3`) and reranking (`voyage-rerank-2`)
- **ast-grep**: Tree-sitter based structural search supporting 20+ programming languages
- **Qdrant**: Vector database for efficient similarity search
- **FastMCP**: Model Context Protocol implementation

## Architecture

### Core Components

- **`src/code_weaver_mcp/main.py`**: Main MCP server implementation with all tools and logic
- **`src/code_weaver_mcp/__init__.py`**: Package initialization (currently empty)
- **`src/code_weaver_mcp/assistant_guide.md`**: Comprehensive usage guide for AI assistants

### Key Classes

1. **`CodeEmbeddingsServer`**: Main MCP server orchestrating all functionality
2. **`AstGrepChunker`**: Handles intelligent code chunking using ast-grep parsers
3. **`VoyageAIEmbedder`**: Manages Voyage AI embeddings for semantic search
4. **`VoyageAIReranker`**: Handles result reranking for improved relevance
5. **`AstGrepStructuralSearch`**: Provides direct ast-grep pattern matching
6. **`CodeChunk`**: Data structure representing semantic code segments

## Common Development Commands

### Environment Setup
```bash
# Install dependencies using uv (fast Python package manager)
uv sync

# Install development dependencies
uv sync --group dev

# Create virtual environment (if needed)
uv venv

# Activate virtual environment
source .venv/bin/activate
```

### Running the Server
```bash
# Run the MCP server directly
uv run python src/code_weaver_mcp/main.py

# Run using the CLI entry point
uv run code-weaver

# Run with environment variables
VOYAGE_API_KEY=your_key QDRANT_URL=your_url uv run code-weaver
```

### Code Quality & Linting
```bash
# Run linting with ruff
uv run ruff check

# Auto-fix linting issues
uv run ruff check --fix

# Format code
uv run ruff format

# Check specific files
uv run ruff check src/code_weaver_mcp/main.py
```

### Testing
This project currently uses manual testing via the main script. To test:
```bash
# Test with a sample codebase
uv run python src/code_weaver_mcp/main.py
```

### Building and Distribution
```bash
# Build the package
uv build

# Publish to PyPI (requires credentials)
uv publish
```

## Configuration Files

- **`pyproject.toml`**: Project metadata, dependencies, and build configuration
- **`ruff.toml`**: Comprehensive linting and formatting rules
- **`mise.toml`**: Development environment tool management
- **`.mcp.json`**: MCP server configuration for Claude Desktop
- **`uv.lock`**: Dependency lockfile for reproducible builds

## MCP Tools Provided

The server exposes four main tools:

1. **`index_codebase`**: Semantically chunk and embed a codebase for search
2. **`search_code`**: Natural language search with advanced filtering options
3. **`ast_grep_search`**: Structural search using ast-grep patterns
4. **`get_supported_languages`**: List supported languages and capabilities

## Environment Variables Required

- **`VOYAGE_API_KEY`**: Your Voyage AI API key (required)
- **`QDRANT_URL`**: Your Qdrant Cloud URL (required)
- **`QDRANT_API_KEY`**: Your Qdrant API key (optional if no auth)
- **`COLLECTION_NAME`**: Vector collection name (defaults to "code-embeddings")

## Language Support

The system supports 20+ programming languages with proper AST-aware chunking:

**Web**: HTML, CSS, JavaScript, TypeScript, React TSX, Vue, Svelte
**Systems**: Rust, Go, C/C++, C#, Zig
**High-level**: Python, Java, Kotlin, Scala, Ruby, PHP, Swift, Dart
**Functional**: Haskell, OCaml, Elm, Elixir, Erlang, Clojure
**Data**: JSON, YAML, TOML, XML, SQL
**Scripts**: Bash, PowerShell, Docker, Make, CMake

## Development Notes

### Code Style
- Uses Google docstring convention with plain language, active voice, and present tense
- DO NOT USE f-strings in *logging* statements
- Line length: 100 characters
- Auto-fixes enabled in ruff configuration
- Type hints required for public functions (encouraged for all)
- Use modern Python typing (>=3.11): `typing.Self`, `typing.Literal`, piped unions (`int | str`), constructors-as-types (`list[str]`), etc.
- **strong typing**: Use `typing.TypedDict` for structured data, `typing.Protocol` for interfaces, `typing.NamedTuple` for immutable data structures, and `enum.Enum` types for fixed sets of values.
  - Avoid generic types like dict[str, Any] if your know the structure -- use TypedDict or NamedTuple instead.

### Dependencies
- **Production**: `ast-grep-py`, `fastmcp`, `qdrant-client`, `rignore`, `watchdog`
- **Development**: `pytest` (for future testing)
- **Build**: Uses `uv_build` backend

### Key Design Decisions
- **Semantic chunking**: Uses ast-grep for intelligent code segmentation
- **Hybrid search**: Combines semantic embeddings with structural patterns
- **Fallback parsing**: Graceful degradation when ast-grep unavailable
- **Batched processing**: Efficient handling of large codebases
- **Smart filtering**: File type, language, and directory-based filtering

### Performance Considerations
- Chunks limited to 1500 characters max, 50 characters min
- Files larger than 1MB are skipped during indexing
- Batch processing (8 files at a time) for memory efficiency
- Automatic ignoring of common build/cache directories

## Integration Examples

### Claude Desktop Configuration
Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "code-weaver": {
      "command": "uv",
      "args": ["run", "code-weaver"],
      "env": {
        "VOYAGE_API_KEY": "your-key",
        "QDRANT_URL": "your-url",
        "QDRANT_API_KEY": "your-api-key"
      }
    }
  }
}
```

### Typical Usage Flow
1. Index a codebase: `index_codebase` with project root path
2. Search semantically: `search_code` with natural language queries
3. Find patterns: `ast_grep_search` with structural patterns
4. Get language info: `get_supported_languages` for capabilities

## Troubleshooting

### Common Issues
- **"ast-grep not available"**: Install with `uv add ast-grep-py`
- **API key errors**: Verify Voyage AI credentials
- **No results**: Ensure codebase is indexed first
- **Large files skipped**: Files >1MB are automatically excluded

### Performance Optimization
- Use specific file/language filters to narrow search scope
- Disable reranking (`rerank: false`) for faster results
- Consider smaller embedding dimensions for cost optimization
- Index only necessary directories to reduce processing time
