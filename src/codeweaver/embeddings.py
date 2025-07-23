# sourcery skip: avoid-global-variables
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Embedding and reranking functionality using Voyage AI and OpenAI-compatible providers.

Provides semantic embeddings for code chunks and query reranking
using Voyage AI's specialized code models or OpenAI-compatible embeddings
with rate limiting and backoff.
"""

import logging

from abc import ABC, abstractmethod
from typing import Any


try:
    import voyageai

    VOYAGEAI_AVAILABLE = True
except ImportError:
    VOYAGEAI_AVAILABLE = False
    voyageai = None

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

from codeweaver.config import EmbeddingConfig
from codeweaver.rate_limiter import (
    RateLimiter,
    calculate_embedding_tokens,
    calculate_rerank_tokens,
    rate_limited,
)


logger = logging.getLogger(__name__)


class EmbedderBase(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self, config: EmbeddingConfig, rate_limiter: RateLimiter | None = None):
        """Initialize the embedder with configuration and optional rate limiting.

        Args:
            config: Embedding configuration with model settings
            rate_limiter: Optional rate limiter for API calls
        """
        self.config = config
        self.rate_limiter = rate_limiter
        self.dimension = config.dimension

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for documents (code chunks)."""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for search query."""


class VoyageAIEmbedder(EmbedderBase):
    """Handles Voyage AI embeddings for code with rate limiting."""

    def __init__(self, config: EmbeddingConfig, rate_limiter: RateLimiter | None = None):
        """Initialize Voyage AI embedder with configuration and rate limiting.

        Args:
            config: Embedding configuration with Voyage AI settings
            rate_limiter: Optional rate limiter for API calls
        """
        super().__init__(config, rate_limiter)

        if not VOYAGEAI_AVAILABLE:
            raise ImportError("Voyage AI library not available. Install with: uv add voyageai")

        self.client = voyageai.Client(api_key=config.api_key)
        self.model = config.model

    @rate_limited("voyage_embed_documents", calculate_embedding_tokens)
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for documents (code chunks) with rate limiting."""
        try:
            result = self.client.embed(
                texts=texts,
                model=self.model,
                input_type="document",
                output_dimension=self.dimension,
            )
        except Exception:
            logger.exception("Error generating embeddings")
            raise
        else:
            return result.embeddings

    @rate_limited("voyage_embed_query", lambda self, text, **kwargs: max(1, len(text) // 4))
    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for search query with rate limiting."""
        try:
            result = self.client.embed(
                texts=[text], model=self.model, input_type="query", output_dimension=self.dimension
            )
            return result.embeddings[0]
        except Exception:
            logger.exception("Error generating query embedding")
            raise


class OpenAIEmbedder(EmbedderBase):
    """Handles OpenAI-compatible embeddings with rate limiting."""

    def __init__(self, config: EmbeddingConfig, rate_limiter: RateLimiter | None = None):
        """Initialize OpenAI embedder with configuration and rate limiting.

        Args:
            config: Embedding configuration with OpenAI settings
            rate_limiter: Optional rate limiter for API calls
        """
        super().__init__(config, rate_limiter)

        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: uv add openai")

        # Initialize OpenAI client with custom base URL if provided
        client_kwargs = {"api_key": config.api_key}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url

        self.client = openai.AsyncOpenAI(**client_kwargs)
        self.model = config.model

        # Set default models if not specified
        if self.model == "voyage-code-3":  # Default from config
            self.model = "text-embedding-3-small"

    @rate_limited("openai_embed_documents", calculate_embedding_tokens)
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for documents (code chunks) with rate limiting."""
        try:
            # OpenAI supports batching up to 2048 inputs
            batch_size = min(2048, len(texts))
            embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                embedding_kwargs = {"input": batch_texts, "model": self.model}

                # Add dimensions parameter if specified and not native
                if self.dimension and self.dimension != self._get_native_dimensions():
                    embedding_kwargs["dimensions"] = self.dimension

                response = await self.client.embeddings.create(**embedding_kwargs)

                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
        except Exception:
            logger.exception("Error generating OpenAI embeddings")
            raise
        else:
            return embeddings

    @rate_limited("openai_embed_query", lambda self, text, **kwargs: max(1, len(text) // 4))
    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for search query with rate limiting."""
        try:
            embedding_kwargs = {"input": [text], "model": self.model}

            # Add dimensions parameter if specified and not native
            if self.dimension and self.dimension != self._get_native_dimensions():
                embedding_kwargs["dimensions"] = self.dimension

            response = await self.client.embeddings.create(**embedding_kwargs)
            return response.data[0].embedding
        except Exception:
            logger.exception("Error generating OpenAI query embedding")
            raise

    def _get_native_dimensions(self) -> int:
        """Get the native dimensions for the model."""
        if self.model == "text-embedding-3-small":
            return 1536
        return 3072 if self.model == "text-embedding-3-large" else self.dimension


def create_embedder(
    config: EmbeddingConfig, rate_limiter: RateLimiter | None = None
) -> EmbedderBase:
    """Factory function to create the appropriate embedder based on configuration."""
    if config.provider.lower() == "voyage":
        return VoyageAIEmbedder(config, rate_limiter)
    if config.provider.lower() == "openai":
        return OpenAIEmbedder(config, rate_limiter)
    raise ValueError(f"Unknown embedding provider: {config.provider}")


class VoyageAIReranker:
    """Handles Voyage AI reranking with rate limiting."""

    def __init__(
        self, api_key: str, model: str = "voyage-rerank-2", rate_limiter: RateLimiter | None = None
    ):
        """Initialize Voyage AI reranker with API configuration.

        Args:
            api_key: Voyage AI API key for authentication
            model: Reranking model to use (default: voyage-rerank-2)
            rate_limiter: Optional rate limiter for API calls
        """
        self.client = voyageai.Client(api_key=api_key)
        self.model = model
        self.rate_limiter = rate_limiter

    @rate_limited("voyage_rerank", calculate_rerank_tokens)
    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Rerank documents for a query with rate limiting."""
        try:
            result = self.client.rerank(
                query=query, documents=documents, model=self.model, top_k=top_k
            )
        except Exception:
            logger.exception("Error reranking")
            raise
        else:
            return result.results
