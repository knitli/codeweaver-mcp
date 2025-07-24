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
from typing import Any, Protocol, runtime_checkable


@dataclass
class RerankResult:
    """Result from a reranking operation."""

    index: int  # Original index in the input documents
    relevance_score: float  # Relevance score from reranker (0.0 to 1.0)
    document: str | None = None  # Optional: the original document text


@dataclass
class ProviderInfo:
    """Information about a provider's capabilities and configuration."""

    name: str
    display_name: str
    description: str
    supported_capabilities: list["ProviderCapability"]
    default_models: dict[str, str]  # capability -> default model
    supported_models: dict[str, list[str]]  # capability -> list of models
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
