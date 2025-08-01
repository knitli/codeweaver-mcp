# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Common types for search backend providers (e.g. Qdrant).
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic.dataclasses import dataclass

from codeweaver.cw_types.backends.enums import BackendProvider, DistanceMetric, HybridFusionStrategy
from codeweaver.cw_types.providers import ProviderCapability, RerankResult


@dataclass
class ProviderInfo:
    """Information about a provider's capabilities and configuration."""

    name: str
    display_name: str
    description: str
    supported_capabilities: ProviderCapability
    default_models: dict[str, str] | None  # capability -> default model
    supported_models: dict[str, list[str]] | None  # capability -> list of models
    rate_limits: dict[str, int] | None = None  # operation -> limit per minute
    requires_api_key: bool = True
    max_batch_size: int | None = None
    max_input_length: int | None = None
    native_dimensions: dict[str, int] | None = None  # model -> native dimensions


class EmbeddingProviderBase(ABC):
    """Abstract base class for embedding providers with common functionality."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the provider with configuration.

        Args:
            config: Configuration dictionary containing provider-specific settings
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the provider configuration.

        Raises:
            ValueError: If configuration is invalid or missing required values
        """
        ...

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple documents."""
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the current model name."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the embedding dimension."""
        ...

    @property
    def max_batch_size(self) -> int | None:
        """Get the maximum batch size (default: None = unlimited)."""
        return None

    @property
    def max_input_length(self) -> int | None:
        """Get the maximum input length (default: None = unlimited)."""
        return None

    @abstractmethod
    def get_provider_info(self) -> ProviderInfo:
        """Get information about this provider's capabilities."""
        ...


class RerankProviderBase(ABC):
    """Abstract base class for reranking providers with common functionality."""

    def __init__(self, config: dict[str, Any]):
        """Initialize the provider with configuration.

        Args:
            config: Configuration dictionary containing provider-specific settings
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the provider configuration.

        Raises:
            ValueError: If configuration is invalid or missing required values
        """
        ...

    @abstractmethod
    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list["RerankResult"]:
        """Rerank documents based on relevance to the query."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the current reranking model name."""
        ...

    @property
    def max_documents(self) -> int | None:
        """Get the maximum number of documents (default: None = unlimited)."""
        return None

    @property
    def max_query_length(self) -> int | None:
        """Get the maximum query length (default: None = unlimited)."""
        return None

    @abstractmethod
    def get_provider_info(self) -> ProviderInfo:
        """Get information about this provider's capabilities."""
        ...


# Import BackendCapabilities here to avoid circular imports
from codeweaver.cw_types.backends.capabilities import BackendCapabilities
from codeweaver.cw_types.backends.enums import IndexType, StorageType


@dataclass
class ProviderRegistryEntry:
    """Registry entry for a vector database provider.

    This associates a backend provider with its implementation class
    and default capabilities.
    """

    provider: BackendProvider
    backend_class: type | None  # Will be populated when backends are registered
    default_capabilities: BackendCapabilities
    description: str

    @property
    def is_available(self) -> bool:
        """Check if this provider has an implementation available."""
        return self.backend_class is not None


# Default provider capabilities registry
PROVIDER_REGISTRY: dict[BackendProvider, ProviderRegistryEntry] = {
    BackendProvider.QDRANT: ProviderRegistryEntry(
        provider=BackendProvider.QDRANT,
        backend_class=None,  # Will be set when QdrantBackend is registered
        default_capabilities=BackendCapabilities(
            supports_vector_search=True,
            supports_exact_search=True,
            supports_hybrid_search=True,
            supports_sparse_vectors=True,
            supports_filtering=True,
            supports_batch_operations=True,
            supports_snapshots=True,
            supports_sharding=True,
            supports_async_indexing=True,
            supported_distance_metrics=[
                DistanceMetric.COSINE,
                DistanceMetric.EUCLIDEAN,
                DistanceMetric.DOT_PRODUCT,
                DistanceMetric.MANHATTAN,
            ],
            supported_index_types=[IndexType.HNSW, IndexType.FLAT],
            supported_fusion_strategies=[HybridFusionStrategy.RRF],
            is_cloud_native=True,
            supports_multi_tenancy=True,
            supports_quantization=True,
            supports_compression=True,
        ),
        description="Qdrant - High-performance vector database with rich features",
    ),
    BackendProvider.PINECONE: ProviderRegistryEntry(
        provider=BackendProvider.PINECONE,
        backend_class=None,
        default_capabilities=BackendCapabilities(
            supports_vector_search=True,
            supports_sparse_vectors=True,
            supports_hybrid_search=True,
            supports_filtering=True,
            supports_batch_operations=True,
            supported_distance_metrics=[
                DistanceMetric.COSINE,
                DistanceMetric.EUCLIDEAN,
                DistanceMetric.DOT_PRODUCT,
            ],
            supported_index_types=[IndexType.HNSW],
            is_cloud_native=True,
            requires_api_key=True,
            supports_multi_tenancy=True,
            max_vector_dims=20000,
        ),
        description="Pinecone - Fully managed vector database service",
    ),
    BackendProvider.WEAVIATE: ProviderRegistryEntry(
        provider=BackendProvider.WEAVIATE,
        backend_class=None,
        default_capabilities=BackendCapabilities(
            supports_vector_search=True,
            supports_hybrid_search=True,
            supports_filtering=True,
            supports_grouping=True,
            supports_batch_operations=True,
            supports_replication=True,
            supported_distance_metrics=[
                DistanceMetric.COSINE,
                DistanceMetric.EUCLIDEAN,
                DistanceMetric.DOT_PRODUCT,
                DistanceMetric.MANHATTAN,
            ],
            supported_index_types=[IndexType.HNSW, IndexType.FLAT],
            supported_fusion_strategies=[HybridFusionStrategy.RRF, HybridFusionStrategy.LINEAR],
            is_cloud_native=True,
            supports_multi_tenancy=True,
        ),
        description="Weaviate - AI-native vector database with GraphQL API",
    ),
    BackendProvider.CHROMA: ProviderRegistryEntry(
        provider=BackendProvider.CHROMA,
        backend_class=None,
        default_capabilities=BackendCapabilities(
            supports_vector_search=True,
            supports_filtering=True,
            supports_persistence=True,
            supported_distance_metrics=[
                DistanceMetric.COSINE,
                DistanceMetric.EUCLIDEAN,
                DistanceMetric.DOT_PRODUCT,
            ],
            supported_index_types=[IndexType.HNSW],
            storage_type=StorageType.DISK,
        ),
        description="Chroma - Simple, developer-friendly vector database",
    ),
    BackendProvider.FAISS: ProviderRegistryEntry(
        provider=BackendProvider.FAISS,
        backend_class=None,
        default_capabilities=BackendCapabilities(
            supports_vector_search=True,
            supports_exact_search=True,
            supports_batch_operations=True,
            supports_compression=True,
            supports_quantization=True,
            supported_distance_metrics=[DistanceMetric.EUCLIDEAN, DistanceMetric.DOT_PRODUCT],
            supported_index_types=[
                IndexType.FLAT,
                IndexType.IVF_FLAT,
                IndexType.IVF_PQ,
                IndexType.HNSW,
                IndexType.LSH,
            ],
            storage_type=StorageType.MEMORY,
            supports_persistence=False,  # Requires manual save/load
            max_vectors_per_collection=1000000000,  # 1 billion
        ),
        description="FAISS - Facebook's efficient similarity search library",
    ),
    BackendProvider.PGVECTOR: ProviderRegistryEntry(
        provider=BackendProvider.PGVECTOR,
        backend_class=None,
        default_capabilities=BackendCapabilities(
            supports_vector_search=True,
            supports_exact_search=True,
            supports_filtering=True,
            supports_transactions=True,
            supports_persistence=True,
            supported_distance_metrics=[
                DistanceMetric.EUCLIDEAN,
                DistanceMetric.COSINE,
                DistanceMetric.DOT_PRODUCT,
            ],
            supported_index_types=[IndexType.FLAT, IndexType.IVF_FLAT, IndexType.HNSW],
            storage_type=StorageType.DISK,
            consistency_model="strong",
            max_vector_dims=16000,
        ),
        description="pgvector - PostgreSQL extension for vector similarity search",
    ),
    BackendProvider.MILVUS: ProviderRegistryEntry(
        provider=BackendProvider.MILVUS,
        backend_class=None,
        default_capabilities=BackendCapabilities(
            supports_vector_search=True,
            supports_exact_search=True,
            supports_hybrid_search=True,
            supports_sparse_vectors=True,
            supports_filtering=True,
            supports_batch_operations=True,
            supports_sharding=True,
            supports_snapshots=True,
            supports_streaming=True,
            supported_distance_metrics=[
                DistanceMetric.EUCLIDEAN,
                DistanceMetric.COSINE,
                DistanceMetric.DOT_PRODUCT,
            ],
            supported_index_types=[
                IndexType.FLAT,
                IndexType.IVF_FLAT,
                IndexType.IVF_PQ,
                IndexType.HNSW,
                IndexType.ANNOY,
                IndexType.DPG,
            ],
            is_cloud_native=True,
            supports_multi_tenancy=True,
            max_vector_dims=32768,
        ),
        description="Milvus - Highly scalable vector database built for AI",
    ),
    BackendProvider.REDIS: ProviderRegistryEntry(
        provider=BackendProvider.REDIS,
        backend_class=None,
        default_capabilities=BackendCapabilities(
            supports_vector_search=True,
            supports_exact_search=True,
            supports_filtering=True,
            supports_persistence=True,
            supported_distance_metrics=[
                DistanceMetric.EUCLIDEAN,
                DistanceMetric.COSINE,
                DistanceMetric.DOT_PRODUCT,
            ],
            supported_index_types=[IndexType.FLAT, IndexType.HNSW],
            storage_type=StorageType.MEMORY,
            supports_transactions=True,
            max_vector_dims=32768,
        ),
        description="Redis - In-memory data store with vector search capabilities",
    ),
    BackendProvider.ELASTICSEARCH: ProviderRegistryEntry(
        provider=BackendProvider.ELASTICSEARCH,
        backend_class=None,
        default_capabilities=BackendCapabilities(
            supports_vector_search=True,
            supports_exact_search=True,
            supports_hybrid_search=True,
            supports_sparse_vectors=True,
            supports_filtering=True,
            supports_grouping=True,
            supports_batch_operations=True,
            supports_replication=True,
            supports_snapshots=True,
            supported_distance_metrics=[
                DistanceMetric.COSINE,
                DistanceMetric.DOT_PRODUCT,
                DistanceMetric.EUCLIDEAN,
            ],
            supported_index_types=[IndexType.HNSW],
            supported_fusion_strategies=[HybridFusionStrategy.LINEAR, HybridFusionStrategy.RRF],
            is_cloud_native=True,
            supports_multi_tenancy=True,
            consistency_model="eventual",
            max_vector_dims=4096,
        ),
        description="Elasticsearch - Distributed search and analytics engine with vector support",
    ),
    BackendProvider.CUSTOM: ProviderRegistryEntry(
        provider=BackendProvider.CUSTOM,
        backend_class=None,
        default_capabilities=BackendCapabilities(
            # Minimal defaults for custom implementations
            supports_vector_search=True,
            supported_distance_metrics=[DistanceMetric.COSINE],
            supported_index_types=[IndexType.FLAT],
        ),
        description="Custom vector database implementation",
    ),
}


def get_provider_info(provider: BackendProvider) -> ProviderRegistryEntry:
    """Get registry information for a provider.

    Args:
        provider: The backend provider enum

    Returns:
        Registry entry with provider information

    Raises:
        KeyError: If provider is not in registry
    """
    return PROVIDER_REGISTRY[provider]


def register_backend_class(provider: BackendProvider, backend_class: type) -> None:
    """Register a backend implementation class for a provider.

    Args:
        provider: The backend provider enum
        backend_class: The backend implementation class

    Raises:
        KeyError: If provider is not in registry
    """
    if provider not in PROVIDER_REGISTRY:
        raise KeyError(f"Unknown provider: {provider}")

    PROVIDER_REGISTRY[provider].backend_class = backend_class


def get_available_providers() -> list[BackendProvider]:
    """Get list of providers that have implementations available.

    Returns:
        List of available backend providers
    """
    return [provider for provider, entry in PROVIDER_REGISTRY.items() if entry.is_available]
