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
from codeweaver.providers.config import CohereConfig


try:
    import cohere  # type: ignore

    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    cohere = None


logger = logging.getLogger(__name__)


class CohereProvider(CombinedProvider):
    """Cohere provider supporting both embeddings and reranking."""

    def __init__(self, config: CohereConfig | dict[str, Any]):
        """Initialize Cohere provider.

        Args:
            config: CohereConfig instance or configuration dictionary
        """
        super().__init__(config)

        if not COHERE_AVAILABLE:
            raise ImportError("Cohere library not available. Install with: uv add cohere")

        self.client = cohere.Client(api_key=self.config["api_key"])
        self.rate_limiter = self.config.get("rate_limiter")

        # Get provider registry info
        self._registry_entry = get_provider_registry_entry(ProviderType.COHERE)
        self._capabilities = self._registry_entry.capabilities

        # Model configuration
        self._embedding_model = self.config.get("model", self._capabilities.default_embedding_model)
        self._rerank_model = self.config.get("rerank_model", self._capabilities.default_reranking_model)
        self._dimension = self._capabilities.native_dimensions.get(self._embedding_model, 1024)

    def _validate_config(self) -> None:
        """Validate Cohere configuration."""
        if not self.config.get("api_key"):
            raise ValueError("Cohere API key is required")

        embedding_model = self.config.get("model", self._capabilities.default_embedding_model)
        if embedding_model not in self._capabilities.supported_embedding_models:
            available = ", ".join(self._capabilities.supported_embedding_models)
            raise ValueError(
                f"Unknown Cohere embedding model: {embedding_model}. Available: {available}"
            )

        rerank_model = self.config.get("rerank_model", self._capabilities.default_reranking_model)
        if rerank_model not in self._capabilities.supported_reranking_models:
            available = ", ".join(self._capabilities.supported_reranking_models)
            raise ValueError(
                f"Unknown Cohere reranking model: {rerank_model}. Available: {available}"
            )

    # EmbeddingProvider implementation

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return ProviderType.COHERE.value

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

    def get_provider_info(self) -> EmbeddingProviderInfo:
        """Get information about Cohere capabilities from centralized registry."""
        registry_entry = get_provider_registry_entry(ProviderType.COHERE)
        capabilities = registry_entry.capabilities

        return EmbeddingProviderInfo(
            name=ProviderType.COHERE.value,
            display_name=registry_entry.display_name,
            description=registry_entry.description,
            supported_capabilities=[
                ProviderCapability.EMBEDDING,
                ProviderCapability.RERANKING,
                ProviderCapability.BATCH_PROCESSING,
            ],
            capabilities=capabilities,
            default_models={
                "embedding": capabilities.default_embedding_model,
                "reranking": capabilities.default_reranking_model,
            },
            supported_models={
                "embedding": capabilities.supported_embedding_models,
                "reranking": capabilities.supported_reranking_models,
            },
            rate_limits={
                "requests_per_minute": capabilities.requests_per_minute,
                "tokens_per_minute": capabilities.tokens_per_minute,
            },
            requires_api_key=capabilities.requires_api_key,
            max_batch_size=capabilities.max_batch_size,
            max_input_length=capabilities.max_input_length,
            native_dimensions=capabilities.native_dimensions,
        )

    @classmethod
    def get_static_provider_info(cls) -> EmbeddingProviderInfo:
        """Get static provider information from centralized registry."""
        registry_entry = get_provider_registry_entry(ProviderType.COHERE)
        capabilities = registry_entry.capabilities

        return EmbeddingProviderInfo(
            name=ProviderType.COHERE.value,
            display_name=registry_entry.display_name,
            description=registry_entry.description,
            supported_capabilities=[
                ProviderCapability.EMBEDDING,
                ProviderCapability.RERANKING,
                ProviderCapability.BATCH_PROCESSING,
            ],
            capabilities=capabilities,
            default_models={
                "embedding": capabilities.default_embedding_model,
                "reranking": capabilities.default_reranking_model,
            },
            supported_models={
                "embedding": capabilities.supported_embedding_models,
                "reranking": capabilities.supported_reranking_models,
            },
            rate_limits={
                "requests_per_minute": capabilities.requests_per_minute,
                "tokens_per_minute": capabilities.tokens_per_minute,
            },
            requires_api_key=capabilities.requires_api_key,
            max_batch_size=capabilities.max_batch_size,
            max_input_length=capabilities.max_input_length,
            native_dimensions=capabilities.native_dimensions,
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


# Register the Cohere provider in the registry
register_provider_class(ProviderType.COHERE, CohereProvider)
