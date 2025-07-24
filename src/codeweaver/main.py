# sourcery skip: avoid-global-variables
# !/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Voyage AI Code Embeddings MCP Server with FastMCP and ast-grep Integration.

A modern Model Context Protocol server using FastMCP that provides semantic code search using:
- Voyage AI's voyage-code-3 embeddings (best-in-class for code)
- Voyage AI's voyage-rerank-2 reranker
- Qdrant Cloud vector database
- ast-grep tree-sitter parsing for 20+ languages
- Direct ast-grep structural search capabilities

Usage:
    python main.py

Environment Variables:
    VOYAGE_API_KEY: Your Voyage AI API key
    QDRANT_URL: Your Qdrant Cloud URL
    QDRANT_API_KEY: Your Qdrant Cloud API key
    COLLECTION_NAME: Name for the code collection (default: code-embeddings)
"""

import asyncio
import logging

from typing import Any

from fastmcp import Context, FastMCP

from codeweaver.chunker import AST_GREP_AVAILABLE
from codeweaver.config import get_config_manager
from codeweaver.server import create_server, detect_configuration_type


logger = logging.getLogger(__name__)

# FastMCP Server Setup
mcp = FastMCP("Code Weaver")

# Global server instance (initialized lazily)
server_instance = None
config_manager = None


def get_server_instance():
    """Get or create the server instance with automatic type detection."""
    global server_instance, config_manager
    if server_instance is None:
        if config_manager is None:
            config_manager = get_config_manager()
        config = config_manager.load_config()

        # Use factory function with automatic server type detection
        server_instance = create_server(config, server_type="auto")

        # Log which server type was created
        server_type = type(server_instance).__name__
        config_type = detect_configuration_type(config)
        logger.info("Created %s based on %s configuration", server_type, config_type)

    return server_instance


@mcp.tool
async def index_codebase(
    root_path: str, patterns: list[str] | None = None, ctx: Context = None
) -> dict[str, Any]:
    """Index a codebase for semantic search using Voyage AI embeddings and ast-grep parsing.

    Args:
        root_path: Root directory path of the codebase to index
        patterns: File patterns to include (optional, defaults to all supported languages)
        ctx: FastMCP context for logging and session management

    Returns:
        dictionary with indexing results including files processed, chunks created, and languages found
    """
    if ctx:
        await ctx.info(f"Starting indexing of codebase at: {root_path}")

    server = get_server_instance()
    result = await server.index_codebase(root_path=root_path, patterns=patterns)

    if ctx:
        await ctx.info(
            f"Indexing complete: {result['total_chunks']} chunks from {result['files_processed']} files"
        )

    return result


@mcp.tool
async def search_code(
    query: str,
    limit: int = 10,
    *,
    file_filter: str | None = None,
    language_filter: str | None = None,
    chunk_type_filter: str | None = None,
    rerank: bool = True,
    ctx: Context = None,
) -> list[dict[str, Any]]:
    """Search indexed code using natural language queries with advanced filtering.

    Args:
        query: Natural language search query
        limit: Maximum number of results to return
        file_filter: Optional file path filter (e.g., 'src/api')
        language_filter: Optional language filter (e.g., 'python', 'javascript')
        chunk_type_filter: Optional chunk type filter (e.g., 'function', 'class')
        rerank: Whether to use Voyage AI reranker for better results
        ctx: FastMCP context for logging and session management

    Returns:
        list of search results with content, file paths, and relevance scores
    """
    if ctx:
        await ctx.info(f"Searching for: {query}")

    server = get_server_instance()
    results = await server.search_code(
        query=query,
        limit=limit,
        file_filter=file_filter,
        language_filter=language_filter,
        chunk_type_filter=chunk_type_filter,
        rerank=rerank,
    )

    if ctx:
        await ctx.info(f"Found {len(results)} results")

    return results


@mcp.tool
async def ast_grep_search(
    pattern: str, language: str, root_path: str, limit: int = 20, ctx: Context = None
) -> list[dict[str, Any]]:
    """Perform structural code search using ast-grep patterns (requires ast-grep-py).

    Args:
        pattern: ast-grep pattern (e.g., 'function $_($$_) { $$_ }' for JS functions)
        language: Programming language (python, javascript, rust, etc.)
        root_path: Root directory to search in
        limit: Maximum number of results
        ctx: FastMCP context for logging and session management

    Returns:
        list of structural matches with file paths and precise locations

    Raises:
        ValueError: If ast-grep is not available
    """
    if not AST_GREP_AVAILABLE:
        raise ValueError("ast-grep not available, install with: pip install ast-grep-py")

    if ctx:
        await ctx.info(f"Searching for pattern '{pattern}' in {language} files")

    server = get_server_instance()
    results = await server.ast_grep_search(
        pattern=pattern, language=language, root_path=root_path, limit=limit
    )

    if ctx:
        await ctx.info(f"Found {len(results)} structural matches")

    return results


@mcp.tool
async def get_supported_languages(ctx: Context = None) -> dict[str, Any]:
    """Get information about supported languages and ast-grep capabilities.

    Args:
        ctx: FastMCP context for logging and session management

    Returns:
        dictionary with supported languages, extensions, and capability information
    """
    if ctx:
        await ctx.info("Retrieving supported languages and capabilities")

    server = get_server_instance()
    return await server.get_supported_languages()


async def main() -> None:
    """Main entry point with configuration management."""
    global config_manager

    try:
        # Initialize configuration manager
        config_manager = get_config_manager()
        config = config_manager.load_config()

        # Configure logging based on config
        logging.basicConfig(
            level=getattr(logging, config.server.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        logger.info("Starting Code Weaver MCP Server...")
        logger.info("Server version: %s", config.server.server_version)
        logger.info("Embedding provider: %s", config.embedding.provider)
        logger.info("Collection name: %s", config.qdrant.collection_name)
        logger.info("Configuration type: %s", detect_configuration_type(config))

        # Pre-initialize server to show which type will be used
        server = get_server_instance()
        logger.info("Server type: %s", type(server).__name__)

        # Check ast-grep availability
        if AST_GREP_AVAILABLE:
            logger.info("✅ ast-grep available - using tree-sitter parsing for 20+ languages")
        else:
            logger.warning("⚠️  ast-grep not available - using fallback parsing")
            logger.info("Install with: uv add ast-grep-py")

        # Run the FastMCP server
        await mcp.run()

    except ValueError as e:
        logger.exception("❌ Configuration error")
        print(f"\n❌ Configuration error: {e}")
        print("\n📋 Configuration help:")
        print("You can configure Code Weaver using:")
        print("1. Environment variables (VOYAGE_API_KEY, QDRANT_URL, etc.)")
        print("2. TOML config files in these locations:")
        print("   - .local.code-weaver.toml (workspace local)")
        print("   - .code-weaver.toml (repository)")
        print("   - ~/.config/code-weaver/config.toml (user)")
        print("\n🔧 Example config:")
        print(config_manager.get_example_config() if config_manager else "")
        return
    except Exception as e:
        logger.exception("❌ Startup error")
        print(f"❌ Error starting server: {e}")
        return


if __name__ == "__main__":
    asyncio.run(main())
