# sourcery skip: avoid-global-variables
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Cohere provider implementation for embeddings and reranking.

Provides Cohere's multilingual embeddings and reranking using the unified provider interface.
Supports both embedding generation and document reranking with batch processing.
"""

import logging

from typing import Any, ClassVar

from codeweaver.providers.base import (
    CombinedProvider,
    ProviderCapability,
    ProviderInfo,
    RerankResult,
)


try:
    import cohere  # type: ignore

    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    cohere = None


logger = logging.getLogger(__name__)


class CohereProvider(CombinedProvider):
    """Cohere provider supporting both embeddings and reranking."""

    # Provider metadata
    PROVIDER_NAME: ClassVar[str] = "cohere"
    DISPLAY_NAME: ClassVar[str] = "Cohere"
    DESCRIPTION: ClassVar[str] = (
        "Multilingual embeddings and reranking with strong semantic understanding"
    )

    # Supported models
    EMBEDDING_MODELS: ClassVar[dict[str, int]] = {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-english-light-v3.0": 384,
        "embed-multilingual-light-v3.0": 384,
    }

    RERANKING_MODELS: ClassVar[list[str]] = [
        "rerank-english-v3.0",
        "rerank-multilingual-v3.0",
        "rerank-english-v2.0",
        "rerank-multilingual-v2.0",
    ]

    # Rate limits (requests per minute)
    RATE_LIMITS: ClassVar[dict[str, int]] = {
        "embed_requests": 1000,
        "embed_tokens": 1000000,
        "rerank_requests": 1000,
        "rerank_tokens": 100000,
    }

    def __init__(self, config: dict[str, Any]):
        """Initialize Cohere provider.

        Args:
            config: Configuration dictionary with keys:
                - api_key: Cohere API key
                - model: Embedding model name (default: embed-english-v3.0)
                - rerank_model: Reranking model name (default: rerank-english-v3.0)
                - rate_limiter: RateLimiter instance (optional)
        """
        super().__init__(config)

        if not COHERE_AVAILABLE:
            raise ImportError("Cohere library not available. Install with: uv add cohere")

        self.client = cohere.Client(api_key=self.config["api_key"])
        self.rate_limiter = self.config.get("rate_limiter")

        # Model configuration
        self._embedding_model = self.config.get("model", "embed-english-v3.0")
        self._rerank_model = self.config.get("rerank_model", "rerank-english-v3.0")
        self._dimension = self.EMBEDDING_MODELS.get(self._embedding_model, 1024)

    def _validate_config(self) -> None:
        """Validate Cohere configuration."""
        if not self.config.get("api_key"):
            raise ValueError("Cohere API key is required")

        embedding_model = self.config.get("model", "embed-english-v3.0")
        if embedding_model not in self.EMBEDDING_MODELS:
            available = ", ".join(self.EMBEDDING_MODELS.keys())
            raise ValueError(
                f"Unknown Cohere embedding model: {embedding_model}. Available: {available}"
            )

        rerank_model = self.config.get("rerank_model", "rerank-english-v3.0")
        if rerank_model not in self.RERANKING_MODELS:
            available = ", ".join(self.RERANKING_MODELS)
            raise ValueError(
                f"Unknown Cohere reranking model: {rerank_model}. Available: {available}"
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
        """Cohere supports batch processing."""
        return 96  # Cohere's batch limit

    @property
    def max_input_length(self) -> int | None:
        """Cohere has input length limits."""
        return 500000  # Estimated character limit

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for documents."""
        try:
            # Note: This is a placeholder implementation
            # In a real implementation, you would:
            # 1. Handle rate limiting
            # 2. Process in batches
            # 3. Use appropriate input_type

            response = self.client.embed(
                texts=texts, model=self._embedding_model, input_type="search_document"
            )
        except Exception:
            logger.exception("Error generating Cohere embeddings")
            raise
        else:
            return response.embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for search query."""
        try:
            response = self.client.embed(
                texts=[text], model=self._embedding_model, input_type="search_query"
            )
            return response.embeddings[0]

        except Exception:
            logger.exception("Error generating Cohere query embedding")
            raise

    # RerankProvider implementation

    @property
    def max_documents(self) -> int | None:
        """Cohere reranking has document limits."""
        return 1000  # Cohere's limit

    @property
    def max_query_length(self) -> int | None:
        """Cohere has query length limits for reranking."""
        return 10000  # Estimated character limit

    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[RerankResult]:
        """Rerank documents using Cohere."""

        def _raise_value_error(msg: str) -> None:
            raise ValueError(msg)

        try:
            # Validate input limits
            if len(documents) > (self.max_documents or float("inf")):
                _raise_value_error(f"Too many documents: {len(documents)} > {self.max_documents}")

            if len(query) > (self.max_query_length or float("inf")):
                _raise_value_error(f"Query too long: {len(query)} > {self.max_query_length}")

            response = self.client.rerank(
                query=query, documents=documents, model=self._rerank_model, top_k=top_k
            )

            # Convert to our format
            rerank_results = []
            rerank_results.extend(
                RerankResult(
                    index=item.index,
                    relevance_score=item.relevance_score,
                    document=getattr(item, "document", None),
                )
                for item in response.results
            )

        except Exception:
            logger.exception("Error reranking with Cohere")
            raise

        else:
            return rerank_results

    # Provider info methods

    def get_provider_info(self) -> ProviderInfo:
        """Get information about Cohere capabilities."""
        return self.get_static_provider_info()

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
            ],
            default_models={"embedding": "embed-english-v3.0", "reranking": "rerank-english-v3.0"},
            supported_models={
                "embedding": list(cls.EMBEDDING_MODELS.keys()),
                "reranking": cls.RERANKING_MODELS,
            },
            rate_limits=cls.RATE_LIMITS,
            requires_api_key=True,
            supports_batch_processing=True,
            max_batch_size=96,
            max_input_length=500000,
            native_dimensions=cls.EMBEDDING_MODELS,
        )

    @classmethod
    def check_availability(cls, capability: ProviderCapability) -> tuple[bool, str | None]:
        """Check if Cohere is available for the given capability."""
        if not COHERE_AVAILABLE:
            return False, "cohere package not installed (install with: uv add cohere)"

        # Both embedding and reranking are supported
        if capability in [ProviderCapability.EMBEDDING, ProviderCapability.RERANKING]:
            return True, None

        return False, f"Capability {capability.value} not supported by Cohere"
