#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Voyage AI Code Embeddings MCP Server with FastMCP and ast-grep Integration

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
import os
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP, Context

from .chunker import AST_GREP_AVAILABLE
from .server import CodeEmbeddingsServer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastMCP Server Setup
mcp = FastMCP("Code Weaver")

# Global server instance (initialized lazily)
server_instance = None

def get_server_instance() -> CodeEmbeddingsServer:
    """Get or create the server instance."""
    global server_instance
    if server_instance is None:
        server_instance = CodeEmbeddingsServer()
    return server_instance

@mcp.tool
async def index_codebase(
    root_path: str, 
    patterns: Optional[List[str]] = None,
    ctx: Context = None
) -> Dict[str, Any]:
    """Index a codebase for semantic search using Voyage AI embeddings and ast-grep parsing.
    
    Args:
        root_path: Root directory path of the codebase to index
        patterns: File patterns to include (optional, defaults to all supported languages)
        ctx: FastMCP context for logging and session management
    
    Returns:
        Dictionary with indexing results including files processed, chunks created, and languages found
    """
    if ctx:
        await ctx.info(f"Starting indexing of codebase at: {root_path}")
    
    server = get_server_instance()
    result = await server.index_codebase(root_path=root_path, patterns=patterns)
    
    if ctx:
        await ctx.info(f"Indexing complete: {result['total_chunks']} chunks from {result['files_processed']} files")
    
    return result

@mcp.tool
async def search_code(
    query: str,
    limit: int = 10,
    file_filter: Optional[str] = None,
    language_filter: Optional[str] = None,
    chunk_type_filter: Optional[str] = None,
    rerank: bool = True,
    ctx: Context = None
) -> List[Dict[str, Any]]:
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
        List of search results with content, file paths, and relevance scores
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
        rerank=rerank
    )
    
    if ctx:
        await ctx.info(f"Found {len(results)} results")
    
    return results

@mcp.tool
async def ast_grep_search(
    pattern: str,
    language: str,
    root_path: str,
    limit: int = 20,
    ctx: Context = None
) -> List[Dict[str, Any]]:
    """Perform structural code search using ast-grep patterns (requires ast-grep-py).
    
    Args:
        pattern: ast-grep pattern (e.g., 'function $_($$_) { $$_ }' for JS functions)
        language: Programming language (python, javascript, rust, etc.)
        root_path: Root directory to search in
        limit: Maximum number of results
        ctx: FastMCP context for logging and session management
    
    Returns:
        List of structural matches with file paths and precise locations
    
    Raises:
        ValueError: If ast-grep is not available
    """
    if not AST_GREP_AVAILABLE:
        raise ValueError("ast-grep not available, install with: pip install ast-grep-py")
    
    if ctx:
        await ctx.info(f"Searching for pattern '{pattern}' in {language} files")
    
    server = get_server_instance()
    results = await server.ast_grep_search(
        pattern=pattern,
        language=language,
        root_path=root_path,
        limit=limit
    )
    
    if ctx:
        await ctx.info(f"Found {len(results)} structural matches")
    
    return results

@mcp.tool
async def get_supported_languages(ctx: Context = None) -> Dict[str, Any]:
    """Get information about supported languages and ast-grep capabilities.
    
    Args:
        ctx: FastMCP context for logging and session management
    
    Returns:
        Dictionary with supported languages, extensions, and capability information
    """
    if ctx:
        await ctx.info("Retrieving supported languages and capabilities")
    
    server = get_server_instance()
    info = await server.get_supported_languages()
    
    return info

async def main():
    """Main entry point."""
    # Verify environment
    required_vars = ["VOYAGE_API_KEY", "QDRANT_URL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("\n📋 Required environment variables:")
        print("  VOYAGE_API_KEY: Your Voyage AI API key")
        print("  QDRANT_URL: Your Qdrant Cloud URL")
        print("  QDRANT_API_KEY: Your Qdrant Cloud API key (optional if not using auth)")
        print("  COLLECTION_NAME: Collection name (optional, defaults to 'code-embeddings')")
        print("\n🔧 Optional: Install ast-grep for better parsing:")
        print("  pip install ast-grep-py")
        return
    
    # Check ast-grep availability
    if AST_GREP_AVAILABLE:
        print("✅ ast-grep available - using tree-sitter parsing for 20+ languages")
    else:
        print("⚠️  ast-grep not available - using fallback parsing")
        print("   Install with: pip install ast-grep-py")
    
    print("🚀 Starting Voyage AI Code Embeddings MCP Server with FastMCP...")
    
    # Run the FastMCP server
    await mcp.run()

if __name__ == "__main__":
    asyncio.run(main())