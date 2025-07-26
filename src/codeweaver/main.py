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
    CW_EMBEDDING_API_KEY: Your embedding provider API key
    CW_VECTOR_BACKEND_URL: Your vector database URL
    CW_VECTOR_BACKEND_API_KEY: Your vector database API key
    CW_VECTOR_BACKEND_COLLECTION: Name for the code collection (default: code-embeddings)
"""

import asyncio
import logging

from typing import Any

from fastmcp import Context, FastMCP

from codeweaver.middleware.chunking import AST_GREP_AVAILABLE
from codeweaver.server import create_clean_server
from codeweaver.config import get_config_manager


logger = logging.getLogger(__name__)


# Global server instance (initialized lazily)
server_instance = None
config_manager = None


def get_server_instance():
    """Get or create the clean server instance."""
    global server_instance, config_manager
    if server_instance is None:
        if config_manager is None:
            config_manager = get_config_manager()
        config = config_manager.get_config()

        # Create the new clean server implementation
        server_instance = create_clean_server(config)

        # Log server type
        logger.info("Created CleanCodeWeaverServer with plugin system and FastMCP middleware")

    return server_instance


async def main() -> None:
    """Main entry point with configuration management."""
    global config_manager

    try:
        # Initialize configuration manager
        config_manager = get_config_manager()
        config = config_manager.get_config()

        # Configure logging based on config
        logging.basicConfig(
            level=getattr(logging, config.server.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        logger.info("Starting Code Weaver MCP Server...")
        logger.info("Server version: %s", config.server.server_version)
        logger.info("Embedding provider: %s", config.get_effective_embedding_provider())
        logger.info("Collection name: %s", config.backend.collection_name)

        # Pre-initialize server to show which type will be used
        server = get_server_instance()
        logger.info("Server type: %s", type(server).__name__)

        # Check ast-grep availability
        if AST_GREP_AVAILABLE:
            logger.info("✅ ast-grep available - using tree-sitter parsing for 20+ languages")
        else:
            logger.warning("⚠️  ast-grep not available - using fallback parsing")
            logger.info("Install with: uv add ast-grep-py")

        # Run the clean server (which handles its own FastMCP instance)
        await server.run()

    except ValueError as e:
        logger.exception("❌ Configuration error")
        print(f"\n❌ Configuration error: {e}")
        print("\n📋 Configuration help:")
        print("You can configure Code Weaver using:")
        print("1. Environment variables (CW_EMBEDDING_API_KEY, CW_VECTOR_BACKEND_URL, etc.)")
        print("2. TOML config files in these locations:")
        print("   - .local.codeweaver.toml (workspace local)")
        print("   - .codeweaver.toml (repository)")
        print("   - ~/.config/codeweaver/config.toml (user)")
        print("\n🔧 Example config:")
        print(config_manager.get_example_config() if config_manager else "")
        return
    except Exception as e:
        logger.exception("❌ Startup error")
        print(f"❌ Error starting server: {e}")
        return


if __name__ == "__main__":
    asyncio.run(main())
