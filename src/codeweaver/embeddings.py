# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Embedding and reranking functionality with provider system integration.

Provides backward-compatible factory functions that use the new provider system
while maintaining existing API compatibility. New code should use the provider
system directly via codeweaver.providers.

Legacy Support:
- create_embedder() function for existing code
- VoyageAIReranker class for backward compatibility
- EmbedderBase abstract class (deprecated, use providers.base instead)
"""

import logging
import warnings

from abc import ABC, abstractmethod
from typing import Any

from codeweaver.config import EmbeddingConfig
from codeweaver.providers import EmbeddingProvider, get_provider_factory
from codeweaver.rate_limiter import RateLimiter


logger = logging.getLogger(__name__)


class EmbedderBase(ABC):
    """Abstract base class for embedding providers.

    .. deprecated:: 2.0.0
        Use :class:`codeweaver.providers.base.EmbeddingProviderBase` instead.
        This class is maintained for backward compatibility only.
    """

    def __init__(self, config: EmbeddingConfig, rate_limiter: RateLimiter | None = None):
        """Initialize the embedder with configuration and optional rate limiting.

        Args:
            config: Embedding configuration with model settings
            rate_limiter: Optional rate limiter for API calls
        """
        warnings.warn(
            "EmbedderBase is deprecated. Use codeweaver.providers.base.EmbeddingProviderBase instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.config = config
        self.rate_limiter = rate_limiter
        self.dimension = config.dimension

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for documents (code chunks)."""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for search query."""


class _ProviderAdapter(EmbedderBase):
    """Adapter to make new providers compatible with legacy EmbedderBase interface."""

    def __init__(self, provider: EmbeddingProvider, config: EmbeddingConfig):
        """Initialize adapter with a provider instance."""
        super().__init__(config)
        self.provider = provider
        self.dimension = provider.dimension

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Delegate to provider."""
        return await self.provider.embed_documents(texts)

    async def embed_query(self, text: str) -> list[float]:
        """Delegate to provider."""
        return await self.provider.embed_query(text)


class VoyageAIEmbedder(EmbedderBase):
    """Handles Voyage AI embeddings for code with rate limiting.

    .. deprecated:: 2.0.0
        Use :class:`codeweaver.providers.voyage.VoyageAIProvider` instead.
        This class is maintained for backward compatibility only.
    """

    def __init__(self, config: EmbeddingConfig, rate_limiter: RateLimiter | None = None):
        """Initialize Voyage AI embedder with configuration and rate limiting.

        Args:
            config: Embedding configuration with Voyage AI settings
            rate_limiter: Optional rate limiter for API calls
        """
        super().__init__(config, rate_limiter)

        warnings.warn(
            "VoyageAIEmbedder is deprecated. Use codeweaver.providers.voyage.VoyageAIProvider instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Use the new provider system internally
        try:
            from codeweaver.providers.voyage import VoyageAIProvider

            provider_config = {
                "api_key": config.api_key,
                "model": config.model,
                "dimension": config.dimension,
                "rate_limiter": rate_limiter,
            }
            self._provider = VoyageAIProvider(provider_config)
            self.dimension = self._provider.dimension
        except ImportError as e:
            raise ImportError("Voyage AI library not available. Install with: uv add voyageai") from e

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for documents (code chunks) with rate limiting."""
        return await self._provider.embed_documents(texts)

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for search query with rate limiting."""
        return await self._provider.embed_query(text)


class OpenAIEmbedder(EmbedderBase):
    """Handles OpenAI-compatible embeddings with rate limiting.

    .. deprecated:: 2.0.0
        Use :class:`codeweaver.providers.openai.OpenAIProvider` instead.
        This class is maintained for backward compatibility only.
    """

    def __init__(self, config: EmbeddingConfig, rate_limiter: RateLimiter | None = None):
        """Initialize OpenAI embedder with configuration and rate limiting.

        Args:
            config: Embedding configuration with OpenAI settings
            rate_limiter: Optional rate limiter for API calls
        """
        super().__init__(config, rate_limiter)

        warnings.warn(
            "OpenAIEmbedder is deprecated. Use codeweaver.providers.openai.OpenAIProvider instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Use the new provider system internally
        try:
            from codeweaver.providers.openai import OpenAIProvider

            provider_config = {
                "api_key": config.api_key,
                "model": config.model,
                "dimension": config.dimension,
                "base_url": getattr(config, "base_url", None),
                "custom_headers": getattr(config, "custom_headers", {}),
                "rate_limiter": rate_limiter,
            }
            # Remove None values
            provider_config = {k: v for k, v in provider_config.items() if v is not None}

            self._provider = OpenAIProvider(provider_config)
            self.dimension = self._provider.dimension
        except ImportError as e:
            raise ImportError("OpenAI library not available. Install with: uv add openai") from e

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for documents (code chunks) with rate limiting."""
        return await self._provider.embed_documents(texts)

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for search query with rate limiting."""
        return await self._provider.embed_query(text)


def create_embedder(
    config: EmbeddingConfig, rate_limiter: RateLimiter | None = None
) -> EmbedderBase:
    """Factory function to create the appropriate embedder based on configuration.

    .. deprecated:: 2.0.0
        Use :func:`codeweaver.providers.get_provider_factory().create_embedding_provider()` instead.
        This function is maintained for backward compatibility only.

    Args:
        config: Embedding configuration
        rate_limiter: Optional rate limiter

    Returns:
        EmbedderBase instance (wrapped provider)

    Raises:
        ValueError: If provider is unknown or unavailable
    """
    warnings.warn(
        "create_embedder is deprecated. Use codeweaver.providers.get_provider_factory() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        # Use the new provider system
        factory = get_provider_factory()
        provider = factory.create_embedding_provider(config, rate_limiter)

        # Wrap in adapter for backward compatibility
        return _ProviderAdapter(provider, config)

    except Exception as e:
        # Fallback to legacy implementations for compatibility
        if config.provider.lower() == "voyage":
            return VoyageAIEmbedder(config, rate_limiter)
        if config.provider.lower() == "openai":
            return OpenAIEmbedder(config, rate_limiter)
        raise ValueError(f"Unknown embedding provider: {config.provider}") from e


class VoyageAIReranker:
    """Handles Voyage AI reranking with rate limiting.

    .. deprecated:: 2.0.0
        Use :class:`codeweaver.providers.voyage.VoyageAIProvider` instead.
        This class is maintained for backward compatibility only.
    """

    def __init__(
        self, api_key: str, model: str = "voyage-rerank-2", rate_limiter: RateLimiter | None = None
    ):
        """Initialize Voyage AI reranker with API configuration.

        Args:
            api_key: Voyage AI API key for authentication
            model: Reranking model to use (default: voyage-rerank-2)
            rate_limiter: Optional rate limiter for API calls
        """
        warnings.warn(
            "VoyageAIReranker is deprecated. Use codeweaver.providers.voyage.VoyageAIProvider instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Use the new provider system internally
        try:
            from codeweaver.providers.voyage import VoyageAIProvider

            provider_config = {
                "api_key": api_key,
                "rerank_model": model,
                "rate_limiter": rate_limiter,
            }
            self._provider = VoyageAIProvider(provider_config)
        except ImportError as e:
            raise ImportError("Voyage AI library not available. Install with: uv add voyageai") from e

    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Rerank documents for a query with rate limiting."""
        try:
            results = await self._provider.rerank(query, documents, top_k)
            # Convert to legacy format for backward compatibility
            return [
                {
                    "index": result.index,
                    "relevance_score": result.relevance_score,
                    "document": result.document,
                }
                for result in results
            ]
        except Exception:
            logger.exception("Error reranking")
            raise
