# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Base protocols and data structures for embedding and reranking providers.

Defines universal interfaces that all provider implementations must follow,
enabling seamless integration of different embedding and reranking services.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from codeweaver._types import EmbeddingProviderInfo, RerankResult


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

    def get_provider_info(self) -> EmbeddingProviderInfo:
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

    def get_provider_info(self) -> EmbeddingProviderInfo:
        """Get information about this provider's capabilities."""
        ...


class EmbeddingProviderBase(ABC):
    """Abstract base class for embedding providers with common functionality."""

    def __init__(self, config: Any):
        """Initialize the provider with configuration.

        Args:
            config: Configuration object (Pydantic model or dict)
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
    def get_provider_info(self) -> EmbeddingProviderInfo:
        """Get information about this provider's capabilities."""
        ...


class RerankProviderBase(ABC):
    """Abstract base class for reranking providers with common functionality."""

    def __init__(self, config: Any):
        """Initialize the provider with configuration.

        Args:
            config: Configuration object (Pydantic model or dict)
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
    def get_provider_info(self) -> EmbeddingProviderInfo:
        """Get information about this provider's capabilities."""
        ...


class LocalEmbeddingProvider(EmbeddingProviderBase):
    """Base class for local embedding providers that don't require API keys."""

    def _validate_config(self) -> None:
        """Local providers typically have minimal validation requirements."""
        # Override in subclasses for specific validation


class CombinedProvider(EmbeddingProviderBase, RerankProviderBase):
    """Base class for providers that support both embedding and reranking."""

    def __init__(self, config: Any):
        """Initialize combined provider.

        Args:
            config: Configuration object containing settings for both capabilities
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate configuration for both embedding and reranking."""
        ...
