# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Usage examples for the vector database backend abstraction.

Demonstrates how to migrate from the current Qdrant-specific implementation
to the new extensible backend system supporting 15+ vector databases.
"""

import asyncio
import logging

from codeweaver.backends.factory import BackendFactory
from codeweaver.base import DistanceMetric, FilterCondition, SearchFilter, VectorPoint
from codeweaver.config import (
    BackendConfigExtended,
    create_backend_config_from_env,
    create_backend_config_from_legacy,
)


logger = logging.getLogger(__name__)


async def example_basic_usage() -> None:
    """
    Basic usage example showing vector operations.

    Demonstrates the core vector database operations that work
    across all supported backends.
    """
    # Create backend configuration
    config = BackendConfigExtended(
        provider="qdrant",
        url="https://your-cluster.qdrant.io",
        api_key="your-api-key",
        collection_name="example-collection",
    )

    # Create backend instance
    backend = BackendFactory.create_backend(config)

    try:
        # Create collection
        await backend.create_collection(
            name="example-collection", dimension=1024, distance_metric=DistanceMetric.COSINE
        )

        # Prepare some example vectors
        vectors = [
            VectorPoint(
                id="vec1",
                vector=[0.1] * 1024,  # Example vector
                payload={
                    "content": "def hello_world():\n    print('Hello, World!')",
                    "file_path": "example.py",
                    "language": "python",
                    "chunk_type": "function",
                },
            ),
            VectorPoint(
                id="vec2",
                vector=[0.2] * 1024,
                payload={
                    "content": "class MyClass:\n    def __init__(self):\n        pass",
                    "file_path": "example.py",
                    "language": "python",
                    "chunk_type": "class",
                },
            ),
        ]

        # Upsert vectors
        await backend.upsert_vectors("example-collection", vectors)
        logger.info("Upserted %d vectors", len(vectors))

        # Search for similar vectors
        query_vector = [0.15] * 1024  # Example query vector
        results = await backend.search_vectors(
            collection_name="example-collection", query_vector=query_vector, limit=5
        )

        logger.info("Found %d search results", len(results))
        for result in results:
            logger.info("Result ID: %s, Score: %.3f", result.id, result.score)

        # Get collection information
        info = await backend.get_collection_info("example-collection")
        logger.info("Collection info: %d vectors, dimension: %d", info.points_count, info.dimension)

    except Exception:
        logger.exception("Error in basic usage example: %s", config.collection_name)
        raise


async def example_hybrid_search() -> None:
    """
    Hybrid search example combining dense and sparse vectors.

    Shows how to use advanced hybrid search capabilities
    available in backends like Qdrant, Weaviate, and Vespa.
    """
    # Create backend with hybrid search enabled
    config = BackendConfigExtended(
        provider="qdrant",
        url="https://your-cluster.qdrant.io",
        api_key="your-api-key",
        collection_name="hybrid-collection",
        enable_hybrid_search=True,
        enable_sparse_vectors=True,
    )

    # Create hybrid backend
    backend = BackendFactory.create_backend(config)

    try:
        # Create collection with sparse vector support
        await backend.create_collection(
            name="hybrid-collection", dimension=1024, distance_metric=DistanceMetric.COSINE
        )

        # Create sparse index for keyword search
        await backend.create_sparse_index(
            collection_name="hybrid-collection", fields=["content", "chunk_type"], index_type="bm25"
        )

        # Prepare vectors with sparse components
        vectors = [
            VectorPoint(
                id="hybrid1",
                vector=[0.1] * 1024,
                sparse_vector={
                    hash("function") % 10000: 1.0,
                    hash("python") % 10000: 0.8,
                    hash("hello") % 10000: 0.6,
                },
                payload={
                    "content": "def hello_world():\n    print('Hello, World!')",
                    "language": "python",
                    "chunk_type": "function",
                },
            )
        ]

        await backend.upsert_vectors("hybrid-collection", vectors)

        # Perform hybrid search
        dense_vector = [0.15] * 1024
        sparse_query = {"function": 1.0, "python": 0.8}

        results = await backend.hybrid_search(
            collection_name="hybrid-collection",
            dense_vector=dense_vector,
            sparse_query=sparse_query,
            limit=10,
            hybrid_strategy="rrf",  # Reciprocal Rank Fusion
            alpha=0.7,  # Favor dense search slightly
        )

        logger.info("Hybrid search found %d results", len(results))

    except Exception:
        logger.exception("Error in hybrid search example: %s", config.collection_name)


async def example_filtered_search() -> None:
    """
    Example showing advanced filtering capabilities.

    Demonstrates how to use filters that work consistently
    across different vector database backends.
    """
    config = BackendConfigExtended(
        provider="qdrant",
        url="https://your-cluster.qdrant.io",
        api_key="your-api-key",
        collection_name="filtered-collection",
    )

    backend = BackendFactory.create_backend(config)

    try:
        await backend.create_collection("filtered-collection", dimension=1024)

        # Create vectors with rich metadata
        vectors = [
            VectorPoint(
                id=f"vec{i}",
                vector=[0.1 * i] * 1024,
                payload={
                    "language": "python" if i % 2 == 0 else "javascript",
                    "chunk_type": "function" if i % 3 == 0 else "class",
                    "file_size": 1000 + i * 100,
                    "complexity_score": i * 0.1,
                },
            )
            for i in range(10)
        ]

        await backend.upsert_vectors("filtered-collection", vectors)

        # Complex filter example
        search_filter = SearchFilter(
            must=[
                SearchFilter(
                    conditions=[FilterCondition(field="language", operator="eq", value="python")]
                ),
                SearchFilter(
                    conditions=[FilterCondition(field="file_size", operator="gt", value=1200)]
                ),
            ],
            should=[
                SearchFilter(
                    conditions=[
                        FilterCondition(field="chunk_type", operator="eq", value="function")
                    ]
                ),
                SearchFilter(
                    conditions=[FilterCondition(field="complexity_score", operator="lt", value=0.5)]
                ),
            ],
        )

        results = await backend.search_vectors(
            collection_name="filtered-collection",
            query_vector=[0.05] * 1024,
            limit=5,
            search_filter=search_filter,
            score_threshold=0.8,
        )

        logger.info("Filtered search found %d results", len(results))

    except Exception:
        logger.exception("Error in filtered search example: %s", config.collection_name)


async def example_migration_from_legacy() -> None:
    """
    Example showing how to migrate from legacy Qdrant-specific code.

    Demonstrates backward compatibility and smooth migration path
    from the current server.py implementation.
    """
    # Old way (current server.py approach)
    # from qdrant_client import QdrantClient
    # qdrant = QdrantClient(url="...", api_key="...")

    # New way - migrate existing configuration
    legacy_config = create_backend_config_from_legacy(
        qdrant_url="https://your-cluster.qdrant.io",
        qdrant_api_key="your-api-key",
        collection_name="code-embeddings",
        enable_sparse_vectors=False,
    )

    # Create backend using the same settings
    backend = BackendFactory.create_backend(legacy_config)

    # The API remains similar but now works with any backend
    await backend.create_collection(
        name="code-embeddings", dimension=1024, distance_metric=DistanceMetric.COSINE
    )

    logger.info("Successfully migrated from legacy configuration")


async def example_multi_backend_setup() -> None:
    """
    Example showing how to use multiple backends simultaneously.

    Useful for scenarios like primary/backup storage, A/B testing,
    or gradual migration between providers.
    """
    # Primary backend (Qdrant with hybrid search)
    primary_config = BackendConfigExtended(
        provider="qdrant",
        url="https://primary-cluster.qdrant.io",
        api_key="primary-api-key",
        enable_hybrid_search=True,
    )
    primary_backend = BackendFactory.create_backend(primary_config)

    # Backup backend (different provider)
    BackendConfigExtended(
        provider="pinecone",  # Would be implemented
        api_key="backup-api-key",
        provider_options={"environment": "us-west1-gcp"},
    )
    # backup_backend = BackendFactory.create_backend(backup_config)

    # Use primary for writes, both for reads
    await primary_backend.create_collection("dual-storage", dimension=1024)

    vectors = [VectorPoint(id="test", vector=[0.1] * 1024, payload={"test": True})]
    await primary_backend.upsert_vectors("dual-storage", vectors)

    # Could implement read distribution, failover logic, etc.
    logger.info("Multi-backend setup completed")


async def example_environment_based_config() -> None:
    """
    Example using environment-based configuration.

    Shows how to configure backends through environment variables
    for different deployment scenarios (dev, staging, production).
    """
    # Load configuration from environment
    config = create_backend_config_from_env()

    # Override specific settings for this use case
    config.batch_size = 50
    config.enable_hybrid_search = True

    BackendFactory.create_backend(config)

    # List available providers and their capabilities
    providers = BackendFactory.list_supported_providers()
    logger.info("Available providers: %s", list(providers.keys()))

    for provider, capabilities in providers.items():
        if capabilities["available"]:
            logger.info(
                "Provider %s: hybrid=%s, streaming=%s",
                provider,
                capabilities["supports_hybrid_search"],
                capabilities.get("supports_streaming", False),
            )


async def main() -> None:
    """Run all examples."""
    logging.basicConfig(level=logging.INFO)

    examples = [
        ("Basic Usage", example_basic_usage),
        ("Hybrid Search", example_hybrid_search),
        ("Filtered Search", example_filtered_search),
        ("Migration from Legacy", example_migration_from_legacy),
        ("Multi-Backend Setup", example_multi_backend_setup),
        ("Environment Configuration", example_environment_based_config),
    ]

    for name, example_func in examples:
        try:
            logger.info("Running example: %s", name)
            await example_func()
            logger.info("Completed example: %s", name)
        except Exception:
            logger.exception("Failed example %s.", name)

        print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())
