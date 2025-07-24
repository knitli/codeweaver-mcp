# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
VoyageAI provider implementation for embeddings and reranking.

Provides VoyageAI's specialized code embeddings and reranking using the unified provider interface.
Supports both embedding generation and document reranking with rate limiting.
"""

import logging

from typing import Any, ClassVar

from codeweaver.providers.base import (
    CombinedProvider,
    ProviderCapability,
    ProviderInfo,
    RerankResult,
)
from codeweaver.rate_limiter import (
    calculate_embedding_tokens,
    calculate_rerank_tokens,
    rate_limited,
)


try:
    import voyageai

    VOYAGEAI_AVAILABLE = True
except ImportError:
    VOYAGEAI_AVAILABLE = False
    voyageai = None


logger = logging.getLogger(__name__)


class VoyageAIProvider(CombinedProvider):
    """VoyageAI provider supporting both embeddings and reranking."""

    # Provider metadata
    PROVIDER_NAME: ClassVar[str] = "voyage"
    DISPLAY_NAME: ClassVar[str] = "Voyage AI"
    DESCRIPTION: ClassVar[str] = (
        "Best-in-class code embeddings and reranking with specialized models"
    )

    # Supported models
    EMBEDDING_MODELS: ClassVar[dict[str, int]] = {
        "voyage-code-3": 1024,
        "voyage-3": 1024,
        "voyage-3-lite": 512,
        "voyage-large-2": 1536,
        "voyage-2": 1024,
    }

    RERANKING_MODELS: ClassVar[list[str]] = ["voyage-rerank-2", "voyage-rerank-lite-1"]

    # Rate limits (requests per minute)
    RATE_LIMITS: ClassVar[dict[str, int]] = {
        "embed_requests": 100,
        "embed_tokens": 1000000,
        "rerank_requests": 100,
        "rerank_tokens": 100000,
    }

    def __init__(self, config: dict[str, Any]):
        """Initialize VoyageAI provider.

        Args:
            config: Configuration dictionary with keys:
                - api_key: VoyageAI API key
                - model: Embedding model name (default: voyage-code-3)
                - rerank_model: Reranking model name (default: voyage-rerank-2)
                - dimension: Custom dimension (optional)
                - rate_limiter: RateLimiter instance (optional)
        """
        super().__init__(config)

        if not VOYAGEAI_AVAILABLE:
            raise ImportError("VoyageAI library not available. Install with: uv add voyageai")

        self.client = voyageai.Client(api_key=self.config["api_key"])
        self.rate_limiter = self.config.get("rate_limiter")

        # Embedding configuration
        self._embedding_model = self.config.get("model", "voyage-code-3")
        self._dimension = self.config.get("dimension")
        if self._dimension is None:
            self._dimension = self.EMBEDDING_MODELS.get(self._embedding_model, 1024)

        # Reranking configuration
        self._rerank_model = self.config.get("rerank_model", "voyage-rerank-2")

    def _validate_config(self) -> None:
        """Validate VoyageAI configuration."""
        if not self.config.get("api_key"):
            raise ValueError("VoyageAI API key is required")

        embedding_model = self.config.get("model", "voyage-code-3")
        if embedding_model not in self.EMBEDDING_MODELS:
            available = ", ".join(self.EMBEDDING_MODELS.keys())
            raise ValueError(
                f"Unknown VoyageAI embedding model: {embedding_model}. Available: {available}"
            )

        rerank_model = self.config.get("rerank_model", "voyage-rerank-2")
        if rerank_model not in self.RERANKING_MODELS:
            available = ", ".join(self.RERANKING_MODELS)
            raise ValueError(
                f"Unknown VoyageAI reranking model: {rerank_model}. Available: {available}"
            )

    # EmbeddingProvider implementation

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self.PROVIDER_NAME

    @property
    def model_name(self) -> str:
        """Get the current embedding model name."""
        return self._embedding_model

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension

    @property
    def max_batch_size(self) -> int | None:
        """VoyageAI supports batch processing with reasonable limits."""
        return 128  # Conservative batch size for API stability

    @property
    def max_input_length(self) -> int | None:
        """VoyageAI has input length limits."""
        return 32000  # Characters limit for VoyageAI

    @rate_limited("voyage_embed_documents", calculate_embedding_tokens)
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for documents with rate limiting."""
        try:
            result = self.client.embed(
                texts=texts,
                model=self._embedding_model,
                input_type="document",
                output_dimension=self._dimension,
            )
            return result.embeddings
        except Exception:
            logger.exception("Error generating VoyageAI embeddings")
            raise

    @rate_limited("voyage_embed_query", lambda self, text, **kwargs: max(1, len(text) // 4))
    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for search query with rate limiting."""
        try:
            result = self.client.embed(
                texts=[text],
                model=self._embedding_model,
                input_type="query",
                output_dimension=self._dimension,
            )
            return result.embeddings[0]
        except Exception:
            logger.exception("Error generating VoyageAI query embedding")
            raise

    # RerankProvider implementation

    @property
    def max_documents(self) -> int | None:
        """VoyageAI reranking has document limits."""
        return 1000  # VoyageAI limit for reranking

    @property
    def max_query_length(self) -> int | None:
        """VoyageAI has query length limits for reranking."""
        return 8000  # Characters limit for reranking queries

    @rate_limited("voyage_rerank", calculate_rerank_tokens)
    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[RerankResult]:
        """Rerank documents using VoyageAI with rate limiting."""
        try:
            # Validate input limits
            if len(documents) > (self.max_documents or float("inf")):
                raise ValueError(f"Too many documents: {len(documents)} > {self.max_documents}")

            if len(query) > (self.max_query_length or float("inf")):
                raise ValueError(f"Query too long: {len(query)} > {self.max_query_length}")

            result = self.client.rerank(
                query=query, documents=documents, model=self._rerank_model, top_k=top_k
            )

            # Convert to our format
            rerank_results = []
            for item in result.results:
                rerank_results.append(
                    RerankResult(
                        index=item.index,
                        relevance_score=item.relevance_score,
                        document=item.document if hasattr(item, "document") else None,
                    )
                )

            return rerank_results

        except Exception:
            logger.exception("Error reranking with VoyageAI")
            raise

    # Provider info methods

    def get_provider_info(self) -> ProviderInfo:
        """Get information about VoyageAI capabilities."""
        return ProviderInfo(
            name=self.PROVIDER_NAME,
            display_name=self.DISPLAY_NAME,
            description=self.DESCRIPTION,
            supported_capabilities=[
                ProviderCapability.EMBEDDING,
                ProviderCapability.RERANKING,
                ProviderCapability.BATCH_PROCESSING,
                ProviderCapability.CUSTOM_DIMENSIONS,
            ],
            default_models={"embedding": "voyage-code-3", "reranking": "voyage-rerank-2"},
            supported_models={
                "embedding": list(self.EMBEDDING_MODELS.keys()),
                "reranking": self.RERANKING_MODELS,
            },
            rate_limits=self.RATE_LIMITS,
            requires_api_key=True,
            supports_batch_processing=True,
            max_batch_size=self.max_batch_size,
            max_input_length=self.max_input_length,
            native_dimensions=self.EMBEDDING_MODELS,
        )

    @classmethod
    def get_static_provider_info(cls) -> ProviderInfo:
        """Get static provider information without instantiation."""
        return ProviderInfo(
            name=cls.PROVIDER_NAME,
            display_name=cls.DISPLAY_NAME,
            description=cls.DESCRIPTION,
            supported_capabilities=[
                ProviderCapability.EMBEDDING,
                ProviderCapability.RERANKING,
                ProviderCapability.BATCH_PROCESSING,
                ProviderCapability.CUSTOM_DIMENSIONS,
            ],
            default_models={"embedding": "voyage-code-3", "reranking": "voyage-rerank-2"},
            supported_models={
                "embedding": list(cls.EMBEDDING_MODELS.keys()),
                "reranking": cls.RERANKING_MODELS,
            },
            rate_limits=cls.RATE_LIMITS,
            requires_api_key=True,
            supports_batch_processing=True,
            max_batch_size=128,
            max_input_length=32000,
            native_dimensions=cls.EMBEDDING_MODELS,
        )

    @classmethod
    def check_availability(cls, capability: ProviderCapability) -> tuple[bool, str | None]:
        """Check if VoyageAI is available for the given capability."""
        if not VOYAGEAI_AVAILABLE:
            return False, "voyageai package not installed (install with: uv add voyageai)"

        # Both embedding and reranking are supported
        if capability in [ProviderCapability.EMBEDDING, ProviderCapability.RERANKING]:
            return True, None

        return False, f"Capability {capability.value} not supported by VoyageAI"
