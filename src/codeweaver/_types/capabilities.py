# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Backend capabilities model for vector databases.

This module defines the BackendCapabilities Pydantic model that describes
what features and operations a vector database backend supports.
"""

from typing import Annotated

from pydantic import ConfigDict, Field

from codeweaver._types.core import BaseCapabilities
from codeweaver._types.provider_enums import (
    DistanceMetric,
    HybridFusionStrategy,
    IndexType,
    SparseIndexType,
    StorageType,
)


class BackendCapabilities(BaseCapabilities):
    """Comprehensive capabilities model for vector database backends.

    This model describes all the features, operations, and constraints
    that a vector database backend supports. It's used to make intelligent
    decisions about which backend to use for specific operations and to
    provide proper fallback behavior when certain features aren't available.
    """

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for future extensibility
        json_schema_extra={
            "title": "Vector Database Backend Capabilities",
            "description": "Describes what features and operations a vector database supports",
        },
    )

    # Core vector operations
    supports_vector_search: Annotated[
        bool,
        Field(
            default=True,
            description="Whether the backend supports similarity search on dense vectors",
        ),
    ]
    supports_exact_search: Annotated[
        bool,
        Field(
            default=False, description="Whether the backend supports exact (non-approximate) search"
        ),
    ]
    supports_hybrid_search: Annotated[
        bool,
        Field(default=False, description="Whether the backend supports hybrid dense+sparse search"),
    ]
    supports_sparse_vectors: Annotated[
        bool, Field(default=False, description="Whether the backend supports sparse vector indices")
    ]

    # CRUD operations
    supports_updates: Annotated[
        bool,
        Field(default=True, description="Whether the backend supports updating existing vectors"),
    ]
    supports_deletes: Annotated[
        bool, Field(default=True, description="Whether the backend supports deleting vectors")
    ]
    supports_batch_operations: Annotated[
        bool,
        Field(default=True, description="Whether the backend supports batch insert/update/delete"),
    ]

    # Advanced search features
    supports_filtering: Annotated[
        bool,
        Field(
            default=True,
            description="Whether the backend supports metadata filtering during search",
        ),
    ]
    supports_pagination: Annotated[
        bool,
        Field(default=True, description="Whether the backend supports paginated search results"),
    ]
    supports_grouping: Annotated[
        bool,
        Field(
            default=False,
            description="Whether the backend supports grouping/aggregation of results",
        ),
    ]

    # Index and distance metrics
    supported_distance_metrics: Annotated[
        list[DistanceMetric],
        Field(
            default=[DistanceMetric.COSINE],
            description="List of supported distance metrics for similarity search",
        ),
    ]
    supported_index_types: Annotated[
        list[IndexType],
        Field(default=[IndexType.FLAT], description="List of supported vector index types"),
    ]
    supported_sparse_index_types: Annotated[
        list[SparseIndexType],
        Field(default=[], description="List of supported sparse index types for hybrid search"),
    ]
    supported_fusion_strategies: Annotated[
        list[HybridFusionStrategy],
        Field(default=[], description="List of supported fusion strategies for hybrid search"),
    ]

    # Storage and persistence
    storage_type: Annotated[
        StorageType,
        Field(default=StorageType.DISK, description="Primary storage type used by the backend"),
    ]
    supports_persistence: Annotated[
        bool, Field(default=True, description="Whether data persists across restarts")
    ]
    supports_snapshots: Annotated[
        bool, Field(default=False, description="Whether the backend supports snapshots/backups")
    ]
    supports_replication: Annotated[
        bool, Field(default=False, description="Whether the backend supports data replication")
    ]

    # Performance and scaling
    supports_sharding: Annotated[
        bool, Field(default=False, description="Whether the backend supports horizontal sharding")
    ]
    supports_async_indexing: Annotated[
        bool, Field(default=False, description="Whether indexing can happen asynchronously")
    ]
    supports_streaming: Annotated[
        bool,
        Field(
            default=False, description="Whether the backend supports streaming large result sets"
        ),
    ]
    max_vector_dims: Annotated[
        int | None,
        Field(default=None, description="Maximum supported vector dimensions (None = unlimited)"),
    ]
    max_vectors_per_collection: Annotated[
        int | None,
        Field(default=None, description="Maximum vectors per collection (None = unlimited)"),
    ]

    # Data types and formats
    supports_string_ids: Annotated[
        bool, Field(default=True, description="Whether the backend supports string identifiers")
    ]
    supports_integer_ids: Annotated[
        bool, Field(default=True, description="Whether the backend supports integer identifiers")
    ]
    supports_uuid_ids: Annotated[
        bool, Field(default=True, description="Whether the backend supports UUID identifiers")
    ]
    supports_custom_metadata: Annotated[
        bool, Field(default=True, description="Whether vectors can have custom metadata/payload")
    ]

    # Transactions and consistency
    supports_transactions: Annotated[
        bool, Field(default=False, description="Whether the backend supports ACID transactions")
    ]
    supports_optimistic_locking: Annotated[
        bool,
        Field(
            default=False, description="Whether the backend supports optimistic concurrency control"
        ),
    ]
    consistency_model: Annotated[
        str,
        Field(
            default="eventual",
            description="Consistency model: 'strong', 'eventual', 'linearizable'",
        ),
    ]

    # Cloud and deployment
    is_cloud_native: Annotated[
        bool, Field(default=False, description="Whether this is a cloud-native service")
    ]
    supports_multi_tenancy: Annotated[
        bool, Field(default=False, description="Whether the backend supports multi-tenancy")
    ]
    requires_api_key: Annotated[
        bool,
        Field(default=False, description="Whether the backend requires API key authentication"),
    ]

    # Special features
    supports_quantization: Annotated[
        bool, Field(default=False, description="Whether the backend supports vector quantization")
    ]
    supports_compression: Annotated[
        bool, Field(default=False, description="Whether the backend supports data compression")
    ]
    supports_custom_scoring: Annotated[
        bool, Field(default=False, description="Whether custom scoring functions can be defined")
    ]

    def supports_distance_metric(self, metric: DistanceMetric) -> bool:
        """Check if a specific distance metric is supported.

        Args:
            metric: The distance metric to check

        Returns:
            True if the metric is supported, False otherwise
        """
        return metric in self.supported_distance_metrics

    def supports_index_type(self, index_type: IndexType) -> bool:
        """Check if a specific index type is supported.

        Args:
            index_type: The index type to check

        Returns:
            True if the index type is supported, False otherwise
        """
        return index_type in self.supported_index_types

    def is_compatible_with_requirements(
        self,
        *,
        require_exact_search: bool = False,
        require_hybrid_search: bool = False,
        require_filtering: bool = False,
        require_persistence: bool = False,
        require_transactions: bool = False,
        min_vector_dims: int | None = None,
        max_vector_dims: int | None = None,
    ) -> bool:
        """Check if this backend meets specific requirements.

        Args:
            require_exact_search: Whether exact search is required
            require_hybrid_search: Whether hybrid search is required
            require_filtering: Whether filtering is required
            require_persistence: Whether persistence is required
            require_transactions: Whether transactions are required
            min_vector_dims: Minimum required vector dimensions
            max_vector_dims: Maximum required vector dimensions

        Returns:
            True if all requirements are met, False otherwise
        """
        if require_exact_search and not self.supports_exact_search:
            return False
        if require_hybrid_search and not self.supports_hybrid_search:
            return False
        if require_filtering and not self.supports_filtering:
            return False
        if require_persistence and not self.supports_persistence:
            return False
        if require_transactions and not self.supports_transactions:
            return False

        if self.max_vector_dims is not None:
            if min_vector_dims is not None and min_vector_dims > self.max_vector_dims:
                return False
            if max_vector_dims is not None and max_vector_dims > self.max_vector_dims:
                return False

        return True


def get_all_backend_capabilities() -> dict[str, BackendCapabilities]:
    """Get centralized backend capabilities for all supported providers.

    This is the single source of truth for backend capabilities across
    the entire CodeWeaver system. Used by config validation, factory creation,
    and capability queries.

    Returns:
        Dictionary mapping backend names to their capabilities
    """
    from codeweaver._types.provider_enums import (
        DistanceMetric,
        HybridFusionStrategy,
        IndexType,
        SparseIndexType,
        StorageType,
    )

    return {
        # Currently implemented backends
        "qdrant": BackendCapabilities(
            supports_hybrid_search=True,
            supports_sparse_vectors=True,
            supports_filtering=True,
            supports_updates=True,
            supports_deletes=True,
            supports_snapshots=True,
            supports_replication=True,
            supported_distance_metrics=[
                DistanceMetric.COSINE,
                DistanceMetric.EUCLIDEAN,
                DistanceMetric.DOT_PRODUCT,
            ],
            supported_index_types=[IndexType.HNSW, IndexType.FLAT],
            supported_sparse_index_types=[SparseIndexType.BM25],
            supported_fusion_strategies=[HybridFusionStrategy.RRF, HybridFusionStrategy.LINEAR],
            storage_type=StorageType.DISK,
            supports_persistence=True,
            requires_api_key=True,
            is_cloud_native=True,
        ),
        # Planned backend implementations
        "pinecone": BackendCapabilities(
            supports_hybrid_search=False,
            supports_sparse_vectors=False,
            supports_filtering=True,
            max_vector_dims=40000,
            supported_distance_metrics=[
                DistanceMetric.COSINE,
                DistanceMetric.EUCLIDEAN,
                DistanceMetric.DOT_PRODUCT,
            ],
            supported_index_types=[IndexType.HNSW],
            is_cloud_native=True,
            requires_api_key=True,
            supports_multi_tenancy=True,
        ),
        "chroma": BackendCapabilities(
            supports_hybrid_search=False,
            supports_sparse_vectors=False,
            supports_filtering=True,
            supported_distance_metrics=[
                DistanceMetric.COSINE,
                DistanceMetric.EUCLIDEAN,
                DistanceMetric.MANHATTAN,
            ],
            supported_index_types=[IndexType.HNSW],
            storage_type=StorageType.MEMORY,
        ),
        "weaviate": BackendCapabilities(
            supports_hybrid_search=True,
            supports_sparse_vectors=True,
            supports_filtering=True,
            supports_grouping=True,
            supported_distance_metrics=[
                DistanceMetric.COSINE,
                DistanceMetric.DOT_PRODUCT,
                DistanceMetric.MANHATTAN,
            ],
            supported_index_types=[IndexType.HNSW],
            supported_sparse_index_types=[SparseIndexType.BM25],
            supported_fusion_strategies=[HybridFusionStrategy.DBSF, HybridFusionStrategy.RRF],
            is_cloud_native=True,
            requires_api_key=True,
            supports_multi_tenancy=True,
        ),
        "pgvector": BackendCapabilities(
            supports_hybrid_search=False,
            supports_sparse_vectors=False,
            supports_filtering=True,
            supports_transactions=True,
            supported_distance_metrics=[
                DistanceMetric.COSINE,
                DistanceMetric.EUCLIDEAN,
                DistanceMetric.MANHATTAN,
            ],
            supported_index_types=[IndexType.IVF_FLAT, IndexType.HNSW],
            consistency_model="strong",
            storage_type=StorageType.DISK,
            supports_persistence=True,
        ),
        "redis": BackendCapabilities(
            supports_hybrid_search=False,
            supports_sparse_vectors=False,
            supports_filtering=True,
            supported_distance_metrics=[
                DistanceMetric.COSINE,
                DistanceMetric.EUCLIDEAN,
                DistanceMetric.MANHATTAN,
            ],
            supported_index_types=[IndexType.HNSW, IndexType.FLAT],
            storage_type=StorageType.MEMORY,
            supports_persistence=True,
        ),
        "elasticsearch": BackendCapabilities(
            supports_hybrid_search=True,
            supports_sparse_vectors=True,
            supports_filtering=True,
            supports_grouping=True,
            supports_sharding=True,
            supported_distance_metrics=[
                DistanceMetric.COSINE,
                DistanceMetric.EUCLIDEAN,
                DistanceMetric.DOT_PRODUCT,
            ],
            supported_index_types=[IndexType.HNSW],
            supported_sparse_index_types=[SparseIndexType.BM25, SparseIndexType.TF_IDF],
            supported_fusion_strategies=[HybridFusionStrategy.RRF, HybridFusionStrategy.LINEAR],
            storage_type=StorageType.DISK,
            supports_persistence=True,
        ),
        "opensearch": BackendCapabilities(
            supports_hybrid_search=True,
            supports_sparse_vectors=True,
            supports_filtering=True,
            supports_grouping=True,
            supports_sharding=True,
            supported_distance_metrics=[
                DistanceMetric.COSINE,
                DistanceMetric.EUCLIDEAN,
                DistanceMetric.MANHATTAN,
            ],
            supported_index_types=[IndexType.HNSW, IndexType.FLAT],
            supported_sparse_index_types=[SparseIndexType.BM25],
            supported_fusion_strategies=[HybridFusionStrategy.RRF],
            storage_type=StorageType.DISK,
            supports_persistence=True,
        ),
        "milvus": BackendCapabilities(
            supports_hybrid_search=False,
            supports_sparse_vectors=False,
            supports_filtering=True,
            supports_sharding=True,
            max_vector_dims=32768,
            supported_distance_metrics=[
                DistanceMetric.COSINE,
                DistanceMetric.EUCLIDEAN,
                DistanceMetric.MANHATTAN,
            ],
            supported_index_types=[
                IndexType.IVF_FLAT,
                IndexType.IVF_PQ,
                IndexType.HNSW,
                IndexType.ANNOY,
            ],
            storage_type=StorageType.DISK,
            supports_persistence=True,
        ),
        "vespa": BackendCapabilities(
            supports_hybrid_search=True,
            supports_sparse_vectors=True,
            supports_filtering=True,
            supports_grouping=True,
            supports_custom_scoring=True,
            supports_sharding=True,
            supported_distance_metrics=[
                DistanceMetric.COSINE,
                DistanceMetric.EUCLIDEAN,
                DistanceMetric.DOT_PRODUCT,
            ],
            supported_index_types=[IndexType.HNSW],
            supported_sparse_index_types=[SparseIndexType.BM25],
            supported_fusion_strategies=[HybridFusionStrategy.LINEAR, HybridFusionStrategy.RRF],
            is_cloud_native=True,
            storage_type=StorageType.DISK,
            supports_persistence=True,
        ),
        "faiss": BackendCapabilities(
            supports_hybrid_search=False,
            supports_sparse_vectors=False,
            supports_filtering=False,  # Limited filtering support
            supports_updates=False,  # Requires rebuild for updates
            supports_deletes=False,  # Requires rebuild for deletes
            supported_distance_metrics=[
                DistanceMetric.COSINE,
                DistanceMetric.EUCLIDEAN,
                DistanceMetric.DOT_PRODUCT,
            ],
            supported_index_types=[
                IndexType.FLAT,
                IndexType.IVF_FLAT,
                IndexType.IVF_PQ,
                IndexType.HNSW,
            ],
            storage_type=StorageType.MEMORY,
            supports_persistence=False,  # Requires manual serialization
        ),
        "annoy": BackendCapabilities(
            supports_hybrid_search=False,
            supports_sparse_vectors=False,
            supports_filtering=False,
            supports_updates=False,  # Immutable after building
            supports_deletes=False,  # Immutable after building
            supported_distance_metrics=[
                DistanceMetric.COSINE,
                DistanceMetric.EUCLIDEAN,
                DistanceMetric.MANHATTAN,
                DistanceMetric.DOT_PRODUCT,
            ],
            supported_index_types=[IndexType.ANNOY],
            storage_type=StorageType.DISK,
            supports_persistence=True,
        ),
        "scann": BackendCapabilities(
            supports_hybrid_search=False,
            supports_sparse_vectors=False,
            supports_filtering=False,
            supports_quantization=True,
            supported_distance_metrics=[DistanceMetric.COSINE, DistanceMetric.DOT_PRODUCT],
            supported_index_types=[IndexType.SCANN],
            storage_type=StorageType.MEMORY,
        ),
        "lancedb": BackendCapabilities(
            supports_hybrid_search=False,
            supports_sparse_vectors=False,
            supports_filtering=True,
            supports_updates=True,
            supports_deletes=True,
            supported_distance_metrics=[
                DistanceMetric.COSINE,
                DistanceMetric.EUCLIDEAN,
                DistanceMetric.DOT_PRODUCT,
            ],
            supported_index_types=[IndexType.IVF_PQ, IndexType.HNSW],
            storage_type=StorageType.DISK,
            supports_persistence=True,
        ),
        "marqo": BackendCapabilities(
            supports_hybrid_search=True,
            supports_sparse_vectors=True,
            supports_filtering=True,
            supports_multi_tenancy=True,
            supported_distance_metrics=[
                DistanceMetric.COSINE,
                DistanceMetric.EUCLIDEAN,
                DistanceMetric.DOT_PRODUCT,
            ],
            supported_index_types=[IndexType.HNSW],
            supported_sparse_index_types=[SparseIndexType.BM25],
            supported_fusion_strategies=[HybridFusionStrategy.RRF],
            is_cloud_native=True,
            requires_api_key=True,
            storage_type=StorageType.DISK,
            supports_persistence=True,
        ),
    }
