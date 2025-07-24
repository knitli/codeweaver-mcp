# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Integration utilities for migrating to the unified factory system.

Provides backward compatibility helpers and migration utilities for
seamlessly transitioning from direct instantiation to factory-based creation.
"""

import logging

from contextlib import asynccontextmanager, contextmanager
from typing import Any

from codeweaver.backends.base import VectorBackend
from codeweaver.config import CodeWeaverConfig
from codeweaver.factories.extensibility_manager import ExtensibilityConfig, ExtensibilityManager
from codeweaver.providers.base import EmbeddingProvider, RerankProvider
from codeweaver.rate_limiter import RateLimiter


logger = logging.getLogger(__name__)


class LegacyCompatibilityAdapter:
    """
    Adapter to maintain backward compatibility with legacy code.

    Provides the same interface as direct component instantiation while
    using the new factory system under the hood.
    """

    def __init__(self, extensibility_manager: ExtensibilityManager):
        """Initialize the compatibility adapter.

        Args:
            extensibility_manager: The extensibility manager instance
        """
        self._manager = extensibility_manager

    async def get_qdrant_client(self) -> Any:
        """Get a Qdrant client instance for backward compatibility.

        Returns:
            QdrantClient instance or compatible wrapper
        """
        backend = await self._manager.get_backend()

        # If it's actually a Qdrant backend, return the underlying client
        if hasattr(backend, "_client") and backend.__class__.__name__ == "QdrantBackend":
            return backend._client  # noqa: SLF001

        # Otherwise, return a compatibility wrapper
        return QdrantCompatibilityWrapper(backend)

    async def get_embedder(self) -> EmbeddingProvider:
        """Get an embedder instance for backward compatibility."""
        return await self._manager.get_embedding_provider()

    async def get_reranker(self) -> RerankProvider | None:
        """Get a reranker instance for backward compatibility."""
        return await self._manager.get_reranking_provider()

    def get_rate_limiter(self) -> RateLimiter:
        """Get the rate limiter instance."""
        return self._manager.get_rate_limiter()


class QdrantCompatibilityWrapper:
    """
    Wrapper to make any VectorBackend compatible with Qdrant client interface.

    Maps Qdrant-specific method calls to the generic VectorBackend protocol.
    """

    def __init__(self, backend: VectorBackend):
        """Initialize the compatibility wrapper.

        Args:
            backend: The vector backend to wrap
        """
        self._backend = backend

    async def get_collections(self) -> Any:
        """Get collections (mapped to backend's collection info)."""

        # This is a simplified mapping - extend as needed
        class CollectionsResponse:
            """Represents a response containing a list of collections.

            This class is used to wrap collections for compatibility with Qdrant client interface.
            """
            def __init__(self, collections):
                self.collections = collections

        class Collection:
            """Represents a collection with a name for compatibility purposes.

            This class is used to wrap collection names for compatibility with Qdrant client interface.
            """
            def __init__(self, name):
                self.name = name

        # Most backends will have some way to list collections
        if hasattr(self._backend, "list_collections"):
            collections = await self._backend.list_collections()
            return CollectionsResponse([Collection(name) for name in collections])

        # Fallback: return empty list
        return CollectionsResponse([])

    async def create_collection(self, collection_name: str, vectors_config: Any) -> None:
        """Create a collection (mapped to backend's create_collection)."""
        dimension = vectors_config.size if hasattr(vectors_config, "size") else 1536
        await self._backend.create_collection(collection_name, dimension)

    async def upsert(self, collection_name: str, points: list[Any]) -> None:
        """Upsert points (mapped to backend's upsert_vectors)."""
        # Convert Qdrant points to generic vector points
        vector_points = []
        for point in points:
            vector_point = {"id": str(point.id), "vector": point.vector, "metadata": point.payload}
            vector_points.append(vector_point)

        await self._backend.upsert_vectors(vector_points)

    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        query_filter: Any | None = None,
        limit: int = 10,
    ) -> list[Any]:
        """Search vectors (mapped to backend's search_vectors)."""
        filters = self._convert_qdrant_filter(query_filter) if query_filter else None
        results = await self._backend.search_vectors(
            query_vector=query_vector, filters=filters, limit=limit
        )

        class SearchResult:
            """Represents a single search result with score and payload.

            This class holds the score and associated payload for a search result.
            """
            def __init__(self, score, payload):
                self.score = score
                self.payload = payload

        return [SearchResult(r.score, r.metadata) for r in results]

    def _convert_qdrant_filter(self, qdrant_filter: Any) -> dict[str, Any]:
        """Convert Qdrant filter to generic filter format."""
        # This is a simplified conversion - extend based on actual usage
        filters = {}

        if hasattr(qdrant_filter, "must"):
            for condition in qdrant_filter.must:
                if hasattr(condition, "key") and hasattr(condition, "match"):
                    filters[condition.key] = condition.match.value

        return filters


@asynccontextmanager
async def create_extensibility_context(
    config: CodeWeaverConfig, extensibility_config: ExtensibilityConfig | None = None
) -> ExtensibilityManager:
    """Create an extensibility context for easy migration.

    Args:
        config: Main CodeWeaver configuration
        extensibility_config: Optional extensibility configuration

    Yields:
        Initialized ExtensibilityManager instance
    """
    manager = ExtensibilityManager(config, extensibility_config)

    try:
        await manager.initialize()
        yield manager
    finally:
        await manager.shutdown()


def create_migration_config(
    *, enable_plugins: bool = False, enable_legacy_fallback: bool = True, lazy_init: bool = True
) -> ExtensibilityConfig:
    """Create an extensibility configuration for migration scenarios.

    Args:
        enable_plugins: Whether to enable plugin discovery
        enable_legacy_fallback: Whether to enable legacy fallbacks
        lazy_init: Whether to use lazy initialization

    Returns:
        ExtensibilityConfig configured for migration
    """
    return ExtensibilityConfig(
        enable_plugin_discovery=enable_plugins,
        auto_load_plugins=enable_plugins,
        enable_legacy_fallbacks=enable_legacy_fallback,
        migration_mode=True,
        lazy_initialization=lazy_init,
        component_caching=True,
    )


class ServerMigrationHelper:
    """Helper class for migrating CodeEmbeddingsServer to use factories."""

    def __init__(self, server_instance: Any):
        """Initialize the migration helper.

        Args:
            server_instance: The CodeEmbeddingsServer instance to migrate
        """
        self.server = server_instance
        self._manager: ExtensibilityManager | None = None
        self._adapter: LegacyCompatibilityAdapter | None = None

    async def migrate_to_factories(self) -> None:
        """Migrate the server to use the factory system."""
        logger.info("Starting server migration to factory system")

        # Create extensibility manager with migration config
        migration_config = create_migration_config(
            enable_plugins=False,  # Start without plugins
            enable_legacy_fallback=True,
            lazy_init=False,  # Initialize immediately for migration
        )

        self._manager = ExtensibilityManager(self.server.config, migration_config)
        await self._manager.initialize()

        # Create compatibility adapter
        self._adapter = LegacyCompatibilityAdapter(self._manager)

        # Replace server components with factory-created ones
        logger.info("Replacing server components with factory instances")

        # Replace Qdrant client
        if hasattr(self.server, "qdrant"):
            self.server.qdrant = await self._adapter.get_qdrant_client()

        # Replace embedder
        if hasattr(self.server, "embedder"):
            self.server.embedder = await self._adapter.get_embedder()

        # Replace reranker
        if hasattr(self.server, "reranker"):
            reranker = await self._adapter.get_reranker()
            if reranker:
                self.server.reranker = reranker

        # Replace rate limiter
        if hasattr(self.server, "rate_limiter"):
            self.server.rate_limiter = self._adapter.get_rate_limiter()

        logger.info("Server migration complete")

    async def cleanup(self) -> None:
        """Cleanup migration resources."""
        if self._manager:
            await self._manager.shutdown()

    @contextmanager
    def temporary_migration(self) -> None:
        """Context manager for temporary migration (useful for testing)."""
        original_components = {
            attr: getattr(self.server, attr)
            for attr in ["qdrant", "embedder", "reranker", "rate_limiter"]
            if hasattr(self.server, attr)
        }
        try:
            yield
        finally:
            # Restore original components
            for attr, value in original_components.items():
                setattr(self.server, attr, value)


def validate_migration_readiness(config: CodeWeaverConfig) -> dict[str, Any]:
    """Validate if a configuration is ready for migration.

    Args:
        config: Configuration to validate

    Returns:
        Validation results with readiness status and any issues
    """
    results = {"ready": True, "issues": [], "warnings": [], "recommendations": []}

    # Check for required configuration sections
    if not hasattr(config, "backend") or not config.backend:
        results["ready"] = False
        results["issues"].append("Missing backend configuration")

    if not hasattr(config, "embedding") or not config.embedding:
        results["ready"] = False
        results["issues"].append("Missing embedding configuration")

    # Check for deprecated patterns
    if hasattr(config, "qdrant") and config.qdrant:
        results["warnings"].append(
            "Direct Qdrant configuration detected - will be migrated to backend config"
        )

    # Provide recommendations
    if results["ready"]:
        results["recommendations"].append(
            "Configuration is ready for migration. Use ServerMigrationHelper for smooth transition."
        )
    else:
        results["recommendations"].append("Fix configuration issues before attempting migration.")

    return results
