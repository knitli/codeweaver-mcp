# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Base protocols and data structures for embedding and reranking providers.

Defines universal interfaces that all provider implementations must follow,
enabling seamless integration of different embedding and reranking services.
"""

import enum

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, runtime_checkable


if TYPE_CHECKING:
    from codeweaver.backends.base import VectorBackend
    from codeweaver.backends.factory import AvailableBackend


class CustomBackendCapabilities(TypedDict):
    """Typed dictionary for custom backend capabilities.

    This is used to define arbitrary capabilities for custom backends.
    """

    only_traditional_search: bool
    """Only set this to True if the backend ONLY supports traditional keyword search"""
    is_reranker: bool
    is_embedder: bool
    is_only_traditional_search: bool

    supported_algorithms:
    supports_hybrid_search: bool
    supports_streaming: bool
    supports_transactions: bool
    supports_batch_processing: bool
    supports_sparse_vectors: bool
    supports_vector_indexing: bool
    supports_vector_search: bool
    supports_vector_upsert: bool
    supports_vector_deletion: bool
    supports_vector_metadata: bool
    supports_vector_similarity_search: bool
    supports_vector_filtering: bool

    in_memory_only: bool
    persistent_storage: bool


@dataclass
class RerankResult:
    """Result from a reranking operation."""

    index: int  # Original index in the input documents
    relevance_score: float  # Relevance score from reranker (0.0 to 1.0)
    document: str | None = None  # Optional: the original document text


class BackendResourceProvider(enum.Enum):
    """Enumeration of supported vector database providers."""

    QDRANT = "qdrant"

    # Planned providers
    # These are not yet implemented but will be added in the future
    ANNOY = "annoy"
    CHROMA = "chroma"
    CUSTOM = "custom"  # User-defined provider
    ELASTICSEARCH = "elasticsearch"
    FAISS = "faiss"
    LANCEDB = "lancedb"
    MARQO = "marqo"
    MILVUS = "milvus"
    OPENSEARCH = "opensearch"
    PGVECTOR = "pgvector"
    PINECONE = "pinecone"
    REDIS = "redis"
    SCANN = "scann"
    VESPA = "vespa"
    WEAVIATE = "weaviate"

    @classmethod
    def from_string(cls, value: str) -> "BackendResourceProvider":
        """Create a BackendResourceProvider from a string value."""
        normalized_value = value.strip().lower().replace(" ", "_").replace("-", "_")
        if normalized_value in cls._value2member_map_:
            return cls._value2member_map_[normalized_value]
        # TODO: I don't think this will work... we need a formal way to define custom providers
        # Check if the value matches a custom provider defined in globals()
        if globals().get(value) or globals().get(normalized_value):
            return cls.CUSTOM
        return None

    @property
    def supports_hybrid_search(self) -> bool:
        """Check if the provider supports hybrid search."""
        match self:
            case (
                BackendResourceProvider.ELASTICSEARCH
                | BackendResourceProvider.MARQO
                | BackendResourceProvider.OPENSEARCH
                | BackendResourceProvider.QDRANT
                | BackendResourceProvider.VESPA
                | BackendResourceProvider.WEAVIATE
            ):
                return True
            case (
                BackendResourceProvider.ANNOY
                | BackendResourceProvider.CHROMA
                | BackendResourceProvider.FAISS
                | BackendResourceProvider.LANCEDB
                | BackendResourceProvider.MILVUS
                | BackendResourceProvider.PGVECTOR
                | BackendResourceProvider.PINECONE
                | BackendResourceProvider.REDIS
                | BackendResourceProvider.SCANN
            ):
                return False
            case BackendResourceProvider.CUSTOM:
                if backend := getattr(self, "backend", None):
                    # Check if the custom backend supports hybrid search
                    # This assumes the backend class has a hybrid_search method
                    # and is a subclass of HybridSearchBackend
                    return all(
                        hasattr(backend, method)
                        for method in [
                            "hybrid_search",
                            "create_sparse_index",
                            "update_sparse_vectors",
                        ]
                    )
                return None
            case _:
                raise NotImplementedError(
                    f"Hybrid search not implemented for provider: {self.value}"
                )

    @property
    def supports_streaming(self) -> bool:
        """Check if the provider supports streaming operations."""
        return all(hasattr(self.backend, "stream_upsert"))

    @property
    def supports_transactions(self) -> bool:
        """Check if the provider supports transactions."""
        # This checks if the backend has a method for starting transactions
        return hasattr(self.backend, "begin_transaction")

    @property
    def normalized_name(self) -> str:
        """Get the normalized name of the provider."""
        return self.value.lower()

    @property
    def backend(self) -> type["VectorBackend"]:
        """Get the backend class for this provider."""
        try:
            provider = getattr(
                __import__("codeweaver.backends", fromlist=[self.value]),
                f"{self.value.capitalize()}Backend",
            )
        except ImportError as e:
            raise NotImplementedError(f"Backend not implemented for provider: {self.value}") from e
        else:
            return provider

    def to_available_backend(self) -> "AvailableBackend":
        """Convert to an AvailableBackend dictionary."""
        return {
            "provider": self,
            "backend": self.backend,
            "available": self
            in [BackendResourceProvider.QDRANT, BackendResourceProvider.CUSTOM],
            "supports_hybrid_search": self.supports_hybrid_search,
            "supports_streaming": self.supports_streaming,
            "supports_transactions": self.supports_transactions,
        }


@dataclass
class ProviderInfo:
    """Information about a provider's capabilities and configuration."""

    name: str
    display_name: str
    description: str
    supported_capabilities: list["ProviderCapability"]
    default_models: dict[str, str] | None  # capability -> default model
    supported_models: dict[str, list[str]] | None  # capability -> list of models
    rate_limits: dict[str, int] | None = None  # operation -> limit per minute
    requires_api_key: bool = True
    supports_batch_processing: bool = True
    max_batch_size: int | None = None
    max_input_length: int | None = None
    native_dimensions: dict[str, int] | None = None  # model -> native dimensions


class ProviderCapability(enum.Enum):
    """Capabilities that providers can support."""

    EMBEDDING = "embedding"
    RERANKING = "reranking"
    LOCAL_INFERENCE = "local_inference"
    CUSTOM_DIMENSIONS = "custom_dimensions"
    BATCH_PROCESSING = "batch_processing"


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers.

    All embedding providers must implement this interface to be compatible
    with the CodeWeaver embedding system.
    """

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        ...

    @property
    def model_name(self) -> str:
        """Get the current model name."""
        ...

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        ...

    @property
    def max_batch_size(self) -> int | None:
        """Get the maximum batch size for this provider (None = unlimited)."""
        ...

    @property
    def max_input_length(self) -> int | None:
        """Get the maximum input length in characters (None = unlimited)."""
        ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors, one per input text

        Raises:
            ValueError: If input is invalid or too large
            RuntimeError: If the provider service is unavailable
        """
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector for the query

        Raises:
            ValueError: If input is invalid or too large
            RuntimeError: If the provider service is unavailable
        """
        ...

    def get_provider_info(self) -> ProviderInfo:
        """Get information about this provider's capabilities."""
        ...


@runtime_checkable
class RerankProvider(Protocol):
    """Protocol for reranking providers.

    All reranking providers must implement this interface to be compatible
    with the CodeWeaver reranking system.
    """

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        ...

    @property
    def model_name(self) -> str:
        """Get the current reranking model name."""
        ...

    @property
    def max_documents(self) -> int | None:
        """Get the maximum number of documents that can be reranked (None = unlimited)."""
        ...

    @property
    def max_query_length(self) -> int | None:
        """Get the maximum query length in characters (None = unlimited)."""
        ...

    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[RerankResult]:
        """Rerank documents based on relevance to the query.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Maximum number of results to return (None = all)

        Returns:
            List of rerank results ordered by relevance (highest first)

        Raises:
            ValueError: If input is invalid or too large
            RuntimeError: If the provider service is unavailable
        """
        ...

    def get_provider_info(self) -> ProviderInfo:
        """Get information about this provider's capabilities."""
        ...


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
    ) -> list[RerankResult]:
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


class LocalEmbeddingProvider(EmbeddingProviderBase):
    """Base class for local embedding providers that don't require API keys."""

    def __init__(self, config: dict[str, Any]):
        """Initialize local provider.

        Args:
            config: Configuration dictionary (API key not required)
        """
        # Override to not require API key validation
        self.config = config
        # Skip base class __init__ to avoid API key validation
        self._validate_local_config()

    @abstractmethod
    def _validate_local_config(self) -> None:
        """Validate local provider configuration (no API key required)."""
        ...

    def _validate_config(self) -> None:
        """Implement abstract method but delegate to local validation."""
        self._validate_local_config()

    @property
    def requires_api_key(self) -> bool:
        """Local providers don't require API keys."""
        return False


class CombinedProvider(EmbeddingProviderBase, RerankProviderBase):
    """Base class for providers that support both embedding and reranking."""

    def __init__(self, config: dict[str, Any]):
        """Initialize combined provider.

        Args:
            config: Configuration dictionary containing settings for both capabilities
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate configuration for both embedding and reranking."""
        ...

    def get_embedding_info(self) -> ProviderInfo:
        """Get provider info for embedding capability."""
        info = self.get_provider_info()
        # Filter to only embedding capabilities
        embedding_capabilities = [
            cap for cap in info.supported_capabilities if cap == ProviderCapability.EMBEDDING
        ]
        return ProviderInfo(
            name=info.name,
            display_name=f"{info.display_name} (Embedding)",
            description=f"{info.description} - Embedding only",
            supported_capabilities=embedding_capabilities,
            default_models={k: v for k, v in info.default_models.items() if k == "embedding"},
            supported_models={k: v for k, v in info.supported_models.items() if k == "embedding"},
            rate_limits=info.rate_limits,
            requires_api_key=info.requires_api_key,
            supports_batch_processing=info.supports_batch_processing,
            max_batch_size=info.max_batch_size,
            max_input_length=info.max_input_length,
            native_dimensions=info.native_dimensions,
        )

    def get_reranking_info(self) -> ProviderInfo:
        """Get provider info for reranking capability."""
        info = self.get_provider_info()
        # Filter to only reranking capabilities
        reranking_capabilities = [
            cap for cap in info.supported_capabilities if cap == ProviderCapability.RERANKING
        ]
        return ProviderInfo(
            name=info.name,
            display_name=f"{info.display_name} (Reranking)",
            description=f"{info.description} - Reranking only",
            supported_capabilities=reranking_capabilities,
            default_models={k: v for k, v in info.default_models.items() if k == "reranking"},
            supported_models={k: v for k, v in info.supported_models.items() if k == "reranking"},
            rate_limits=info.rate_limits,
            requires_api_key=info.requires_api_key,
            supports_batch_processing=info.supports_batch_processing,
            max_batch_size=info.max_batch_size,
            max_input_length=info.max_input_length,
            native_dimensions=info.native_dimensions,
        )
