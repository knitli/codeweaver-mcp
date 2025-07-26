# sourcery skip: avoid-global-variables
# !/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
CodeWeaver Extensible MCP Server with Plugin Architecture.

An extensible Model Context Protocol server built on factory patterns and protocol-based
interfaces. Supports multiple embedding providers, vector databases, and data sources
through a comprehensive plugin system and configuration-driven initialization.

Key Architecture Features:
- Factory pattern for dynamic component creation and management
- Protocol-based interfaces for provider, backend, and source abstraction
- Plugin discovery engine with entry point and directory scanning
- Configuration-driven initialization supporting multiple formats (TOML, env vars)
- FastMCP middleware integration for cross-cutting concerns
- Runtime capability querying and component validation

Supported Components:
- Embedding Providers: Voyage AI, OpenAI, Cohere, HuggingFace, Custom
- Vector Backends: Qdrant, Pinecone, Weaviate, ChromaDB, Custom
- Data Sources: Filesystem, Git, Database, API, Web, Custom
- Languages: 20+ programming languages via ast-grep tree-sitter parsing

Usage:
    python main.py

Configuration:
    Uses hierarchical configuration system with environment variables, TOML files,
    and runtime discovery. See config.py for full configuration options.
"""

import asyncio
import logging

from typing import TYPE_CHECKING

from codeweaver.config import get_config_manager
from codeweaver.middleware.chunking import AST_GREP_AVAILABLE
from codeweaver.server import create_clean_server


if TYPE_CHECKING:
    from codeweaver.server import CodeWeaverServer


logger = logging.getLogger(__name__)


# Global server instance (initialized lazily)
server_instance = None
config_manager = None


def get_server_instance() -> "CodeWeaverServer":
    """Get or create the clean server instance."""
    global server_instance, config_manager
    if server_instance is None:
        if config_manager is None:
            config_manager = get_config_manager()
        config = config_manager.get_config()

        # Create the new clean server implementation
        server_instance = create_clean_server(config)

        # Log server type
        logger.info("Created CodeWeaverServer with plugin system and FastMCP middleware")

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
