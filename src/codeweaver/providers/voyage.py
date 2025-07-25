# sourcery skip: avoid-global-variables
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

from typing import Any

from codeweaver._types import (
    EmbeddingProviderInfo,
    ProviderCapability,
    ProviderType,
    RerankResult,
    get_provider_registry_entry,
    register_provider_class,
)
from codeweaver.providers.base import CombinedProvider
from codeweaver.providers.config import VoyageConfig
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

    def __init__(self, config: VoyageConfig | dict[str, Any]):
        """Initialize VoyageAI provider.

        Args:
            config: VoyageConfig instance or configuration dictionary
        """
        super().__init__(config)

        if not VOYAGEAI_AVAILABLE:
            raise ImportError("VoyageAI library not available. Install with: uv add voyageai")

        self.client = voyageai.Client(api_key=self.config["api_key"])
        self.rate_limiter = self.config.get("rate_limiter")

        # Get provider registry info
        self._registry_entry = get_provider_registry_entry(ProviderType.VOYAGE_AI)
        self._capabilities = self._registry_entry.capabilities

        # Embedding configuration
        self._embedding_model = self.config.get("model", self._capabilities.default_embedding_model)
        self._dimension = self.config.get("dimension")
        if self._dimension is None:
            self._dimension = self._capabilities.native_dimensions.get(self._embedding_model, 1024)

        # Reranking configuration
        self._rerank_model = self.config.get("rerank_model", self._capabilities.default_reranking_model)

    def _validate_config(self) -> None:
        """Validate VoyageAI configuration."""
        if not self.config.get("api_key"):
            raise ValueError("VoyageAI API key is required")

        embedding_model = self.config.get("model", self._capabilities.default_embedding_model)
        if embedding_model not in self._capabilities.supported_embedding_models:
            available = ", ".join(self._capabilities.supported_embedding_models)
            raise ValueError(
                f"Unknown VoyageAI embedding model: {embedding_model}. Available: {available}"
            )

        rerank_model = self.config.get("rerank_model", self._capabilities.default_reranking_model)
        if rerank_model not in self._capabilities.supported_reranking_models:
            available = ", ".join(self._capabilities.supported_reranking_models)
            raise ValueError(
                f"Unknown VoyageAI reranking model: {rerank_model}. Available: {available}"
            )

    # EmbeddingProvider implementation

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return ProviderType.VOYAGE_AI.value

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
        except Exception:
            logger.exception("Error generating VoyageAI embeddings")
            raise

        else:
            return result.embeddings

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
        def _raise_value_error(msg: str) -> None:
            raise ValueError(msg)
        try:
            # Validate input limits
            if len(documents) > (self.max_documents or float("inf")):
                _raise_value_error(f"Too many documents: {len(documents)} > {self.max_documents}")

            if len(query) > (self.max_query_length or float("inf")):
                _raise_value_error(f"Query too long: {len(query)} > {self.max_query_length}")

            result = self.client.rerank(
                query=query, documents=documents, model=self._rerank_model, top_k=top_k
            )

            # Convert to our format
            rerank_results = []
            for item in result.results:
                rerank_results.append(  # noqa: PERF401
                    RerankResult(
                        index=item.index,
                        relevance_score=item.relevance_score,
                        document=item.document if hasattr(item, "document") else None,
                    )
                )

        except Exception:
            logger.exception("Error reranking with VoyageAI")
            raise

        else:
            return rerank_results

    def get_provider_info(self) -> EmbeddingProviderInfo:
        """Get information about VoyageAI capabilities from centralized registry."""
        registry_entry = get_provider_registry_entry(ProviderType.VOYAGE_AI)
        capabilities = registry_entry.capabilities

        return EmbeddingProviderInfo(
            name=ProviderType.VOYAGE_AI.value,
            display_name=registry_entry.display_name,
            description=registry_entry.description,
            supported_capabilities=[
                ProviderCapability.EMBEDDING,
                ProviderCapability.RERANKING,
                ProviderCapability.BATCH_PROCESSING,
                ProviderCapability.CUSTOM_DIMENSIONS,
            ],
            capabilities=capabilities,
            default_models={"embedding": capabilities.default_embedding_model, "reranking": capabilities.default_reranking_model},
            supported_models={
                "embedding": capabilities.supported_embedding_models,
                "reranking": capabilities.supported_reranking_models,
            },
            rate_limits={"requests_per_minute": capabilities.requests_per_minute, "tokens_per_minute": capabilities.tokens_per_minute},
            requires_api_key=capabilities.requires_api_key,
            max_batch_size=capabilities.max_batch_size,
            max_input_length=capabilities.max_input_length,
            native_dimensions=capabilities.native_dimensions,
        )

    @classmethod
    def get_static_provider_info(cls) -> EmbeddingProviderInfo:
        """Get static provider information from centralized registry."""
        registry_entry = get_provider_registry_entry(ProviderType.VOYAGE_AI)
        capabilities = registry_entry.capabilities

        return EmbeddingProviderInfo(
            name=ProviderType.VOYAGE_AI.value,
            display_name=registry_entry.display_name,
            description=registry_entry.description,
            supported_capabilities=[
                ProviderCapability.EMBEDDING,
                ProviderCapability.RERANKING,
                ProviderCapability.BATCH_PROCESSING,
                ProviderCapability.CUSTOM_DIMENSIONS,
            ],
            capabilities=capabilities,
            default_models={"embedding": capabilities.default_embedding_model, "reranking": capabilities.default_reranking_model},
            supported_models={
                "embedding": capabilities.supported_embedding_models,
                "reranking": capabilities.supported_reranking_models,
            },
            rate_limits={"requests_per_minute": capabilities.requests_per_minute, "tokens_per_minute": capabilities.tokens_per_minute},
            requires_api_key=capabilities.requires_api_key,
            max_batch_size=capabilities.max_batch_size,
            max_input_length=capabilities.max_input_length,
            native_dimensions=capabilities.native_dimensions,
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


# Register the VoyageAI provider in the registry
register_provider_class(ProviderType.VOYAGE_AI, VoyageAIProvider)
