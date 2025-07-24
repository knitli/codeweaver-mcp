# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Example of how to use the unified factory system in CodeEmbeddingsServer.

This demonstrates the migration path from direct instantiation to factory-based
component creation while maintaining backward compatibility.
"""

import logging

from codeweaver.chunker import AstGrepChunker
from codeweaver.config import CodeWeaverConfig, get_config
from codeweaver.factories.extensibility_manager import ExtensibilityConfig, ExtensibilityManager
from codeweaver.factories.integration import LegacyCompatibilityAdapter, ServerMigrationHelper
from codeweaver.file_watcher import FileWatcherManager
from codeweaver.search import AstGrepStructuralSearch
from codeweaver.task_search import TaskSearchCoordinator


logger = logging.getLogger(__name__)


class CodeEmbeddingsServerWithFactories:
    """
    Enhanced MCP server using the unified factory system.

    This version demonstrates how to use the extensibility manager and factories
    while maintaining the same external interface.
    """

    def __init__(self, config: CodeWeaverConfig | None = None):
        """Initialize the code embeddings server with factory support.

        Args:
            config: Optional configuration object. If None, loads from default sources.
        """
        # Load configuration
        self.config = config or get_config()

        # Initialize extensibility manager
        extensibility_config = ExtensibilityConfig(
            enable_plugin_discovery=True,  # Enable plugin support
            auto_load_plugins=True,
            lazy_initialization=False,  # Initialize components immediately
            component_caching=True,
            enable_legacy_fallbacks=True,
        )

        self._extensibility_manager = ExtensibilityManager(self.config, extensibility_config)
        self._compatibility_adapter: LegacyCompatibilityAdapter | None = None

        # Components that don't need factory creation
        self.chunker = AstGrepChunker(
            max_chunk_size=self.config.chunking.max_chunk_size,
            min_chunk_size=self.config.chunking.min_chunk_size,
        )
        self.structural_search = AstGrepStructuralSearch()
        self.task_coordinator = TaskSearchCoordinator(task_tool_available=True)
        self.file_watcher_manager = FileWatcherManager(self.config)

        # These will be set by the factory system
        self.qdrant = None
        self.embedder = None
        self.reranker = None
        self.rate_limiter = None

        # Initialize asynchronously in a separate method
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the server with factory-created components."""
        if self._initialized:
            return

        logger.info("Initializing CodeEmbeddingsServer with factory system")

        # Initialize extensibility system
        await self._extensibility_manager.initialize()

        # Create compatibility adapter
        self._compatibility_adapter = LegacyCompatibilityAdapter(self._extensibility_manager)

        # Get factory-created components
        self.rate_limiter = self._extensibility_manager.get_rate_limiter()
        self.embedder = await self._extensibility_manager.get_embedding_provider()
        self.reranker = await self._extensibility_manager.get_reranking_provider()

        # Get backend (Qdrant-compatible interface)
        self.qdrant = await self._compatibility_adapter.get_qdrant_client()

        # Ensure collection exists
        await self._ensure_collection()

        self._initialized = True
        logger.info("Server initialization complete")

    async def _ensure_collection(self):
        """Ensure the Qdrant collection exists."""
        try:
            collections = await self.qdrant.get_collections()
            collection_names = [c.name for c in collections.collections]

            collection_name = self.config.qdrant.collection_name
            if collection_name not in collection_names:
                logger.info("Creating collection: %s", collection_name)

                # Use factory system to get vector params
                from qdrant_client.models import Distance, VectorParams

                await self.qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.embedder.dimension, distance=Distance.COSINE
                    ),
                )
            else:
                logger.info("Collection %s already exists", collection_name)
        except Exception:
            logger.exception("Error ensuring collection")
            raise

    async def cleanup(self) -> None:
        """Cleanup resources when shutting down."""
        await self._extensibility_manager.shutdown()

    # ... rest of the methods remain the same as original server.py ...
    # (index_codebase, search_code, ast_grep_search, etc.)


async def demonstrate_factory_migration() -> None:
    """Demonstrate how to migrate an existing server to use factories."""
    # Load configuration
    config = get_config()

    # Option 1: Use the enhanced server directly
    logger.info("Option 1: Direct factory-based server")
    server = CodeEmbeddingsServerWithFactories(config)
    await server.initialize()

    # Use the server normally...
    languages = await server.get_supported_languages()
    logger.info("Supported languages: %d", len(languages["supported_languages"]))

    await server.cleanup()

    # Option 2: Migrate existing server using helper
    logger.info("\nOption 2: Migration helper approach")
    from codeweaver.server import CodeEmbeddingsServer

    legacy_server = CodeEmbeddingsServer(config)
    migration_helper = ServerMigrationHelper(legacy_server)

    await migration_helper.migrate_to_factories()

    # Server now uses factory-created components
    # ... use server normally ...

    await migration_helper.cleanup()

    # Option 3: Use extensibility manager directly for custom scenarios
    logger.info("\nOption 3: Direct extensibility manager usage")
    from codeweaver.factories.integration import create_extensibility_context

    async with create_extensibility_context(config) as manager:
        # Get individual components as needed
        backend = await manager.get_backend()
        embedder = await manager.get_embedding_provider()

        # Use components directly...
        logger.info("Backend ready: %s", backend.__class__.__name__)
        logger.info("Embedder dimension: %d", embedder.dimension)

        # Get component information
        info = manager.get_component_info()
        logger.info("Available backends: %s", list(info["backends"].keys()))


async def demonstrate_validation() -> None:
    """Demonstrate factory validation capabilities."""
    from codeweaver.factories.validation import FactoryValidator, ValidationLevel

    config = get_config()

    # Create extensibility manager
    manager = ExtensibilityManager(config)
    await manager.initialize()

    # Get the unified factory
    factory = manager.get_unified_factory()

    # Create validator
    validator = FactoryValidator(factory, level=ValidationLevel.COMPREHENSIVE)

    # Generate health report
    health_report = await validator.generate_health_report(config)

    logger.info("System health: %s", health_report.overall_health)
    logger.info("Validation results: %d checks", len(health_report.validation_results))
    logger.info("Compatibility results: %d checks", len(health_report.compatibility_results))

    # Show any issues
    for result in health_report.validation_results:
        if not result.passed:
            logger.warning("%s validation failed: %s", result.component, result.message)

    # Show recommendations
    for rec in health_report.recommendations:
        logger.info("Recommendation: %s", rec)

    await manager.shutdown()


if __name__ == "__main__":
    import asyncio

    # Run demonstrations
    asyncio.run(demonstrate_factory_migration())
    asyncio.run(demonstrate_validation())
