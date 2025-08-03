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
from codeweaver.cw_types import BackendConnectionError, ComponentCreationError, ConfigurationError
from codeweaver.middleware.chunking import AST_GREP_AVAILABLE
from codeweaver.server import create_server


if TYPE_CHECKING:
    from codeweaver.server import CodeWeaverServer


logger = logging.getLogger(__name__)
server_instance = None
config_manager = None


def get_server_instance() -> "CodeWeaverServer":
    """Get or create the clean server instance."""
    global server_instance, config_manager
    if server_instance is None:
        if config_manager is None:
            config_manager = get_config_manager()
        config = config_manager.get_config()
        server_instance = create_server(config)
        logger.info("Created CodeWeaverServer with plugin system and FastMCP middleware")
    return server_instance


async def _handle_main_value_error(e: ValueError) -> None:
    """Handle ValueError during main execution."""
    print(f"\n❌ Configuration error: {e}")
    print("\n📋 Configuration help:")
    print("You can configure Code Weaver using:")
    print("1. Environment variables (CW_EMBEDDING_API_KEY, CW_VECTOR_BACKEND_URL, etc.)")
    print("2. TOML config files in these locations:")
    print("   - .local.codeweaver.toml (workspace local)")
    print("   - .codeweaver.toml (repository)")
    print("   - ~/.config/codeweaver/config.toml (user)")
    print("\n🔧 Example config:")
    print(await config_manager.get_example_config() if config_manager else "")


async def _handle_main_configuration_error(e: ConfigurationError) -> None:
    """Handle ConfigurationError during main execution."""
    print(f"\n❌ Configuration error: {e}")
    print("\n💡 Please check your configuration files:")
    print("   - .local.codeweaver.toml (workspace)")
    print("   - .codeweaver/.codeweaver.toml (repository)")
    print("   - ~/.config/codeweaver/config.toml (user)")
    print("\n🔧 Example config:")
    print(await config_manager.get_example_config() if config_manager else "")


async def main() -> None:
    """Main entry point with configuration management."""
    global config_manager
    try:
        config_manager = get_config_manager()
        config = config_manager.get_config()
        logging.basicConfig(
            level=getattr(logging, config.server.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger.info("Starting Code Weaver MCP Server...")
        logger.info("Server version: %s", config.server.server_version)
        logger.info("Embedding provider: %s", config.get_effective_embedding_provider())
        logger.info("Collection name: %s", config.backend.collection_name)
        server = get_server_instance()
        logger.info("Server type: %s", type(server).__name__)
        if AST_GREP_AVAILABLE:
            logger.info("✅ ast-grep available - using tree-sitter parsing for 20+ languages")
        else:
            logger.warning("⚠️  ast-grep not available - using fallback parsing")
            logger.info("Install with: uv add ast-grep-py")
        await server.run()
    except ValueError as e:
        logger.exception("❌ Configuration error")
        await _handle_main_value_error(e)
        return
    except ConfigurationError as e:
        logger.exception("❌ Configuration error")
        await _handle_main_configuration_error(e)
        return
    except (ComponentCreationError, BackendConnectionError) as e:
        logger.exception("❌ Component setup error")
        print(f"❌ Component setup error: {e}")
        print("\n💡 Please check your provider settings and backend connectivity")
        return
    except Exception as e:
        logger.exception("❌ Unexpected startup error")
        print(f"❌ Unexpected error starting server: {e}")
        return


if __name__ == "__main__":
    asyncio.run(main())
