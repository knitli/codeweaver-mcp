# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
VoyageAI provider implementation for embeddings and reranking.

Provides VoyageAI's specialized code embeddings and reranking using the unified provider interface.
Supports both embedding generation and document reranking with rate limiting.
"""

import asyncio
import logging
import time

from typing import Any

from codeweaver.cw_types import (
    EmbeddingProviderError,
    EmbeddingProviderInfo,
    ProviderCapability,
    ProviderCompatibilityError,
    ProviderConfigurationError,
    ProviderType,
    RerankResult,
    get_provider_registry_entry,
    register_provider_class,
)
from codeweaver.providers.base import CombinedProvider
from codeweaver.providers.config import VoyageConfig


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
            raise EmbeddingProviderError(
                "VoyageAI library not available",
                provider_name="voyage_ai",
                operation="initialization",
                recovery_suggestions=["Install with: uv add voyageai"]
            )
        self.client = voyageai.Client(api_key=self.config["api_key"])
        self._last_request_time = 0.0
        self._min_request_interval = 0.1
        self._registry_entry = get_provider_registry_entry(ProviderType.VOYAGE_AI)
        self._capabilities = self._registry_entry.capabilities
        self._embedding_model = self.config.get("model", self._capabilities.default_embedding_model)
        self._dimension = self.config.get("dimension")
        if self._dimension is None:
            self._dimension = self._capabilities.native_dimensions.get(self._embedding_model, 1024)
        self._rerank_model = self.config.get(
            "rerank_model", self._capabilities.default_reranking_model
        )

    def _validate_config(self) -> None:
        """Validate VoyageAI configuration."""
        if not self.config.get("api_key"):
            raise ProviderConfigurationError(
                "VoyageAI API key is required",
                provider_type="embedding",
                provider_name="voyage_ai",
                operation="validation",
                recovery_suggestions=["Set CW_EMBEDDING_API_KEY environment variable or provide api_key in config"]
            )
        embedding_model = self.config.get("model", self._capabilities.default_embedding_model)
        if embedding_model not in self._capabilities.supported_embedding_models:
            available = ", ".join(self._capabilities.supported_embedding_models)
            raise ProviderCompatibilityError(
                f"Unknown VoyageAI embedding model: {embedding_model}",
                provider_type="embedding",
                provider_name="voyage_ai",
                operation="model_validation",
                recovery_suggestions=[
                    f"Use one of the supported models: {available}",
                    "Check VoyageAI documentation for latest model list"
                ]
            )
        rerank_model = self.config.get("rerank_model", self._capabilities.default_reranking_model)
        if rerank_model not in self._capabilities.supported_reranking_models:
            available = ", ".join(self._capabilities.supported_reranking_models)
            raise ProviderCompatibilityError(
                f"Unknown VoyageAI reranking model: {rerank_model}",
                provider_type="embedding",
                provider_name="voyage_ai",
                operation="rerank_model_validation",
                recovery_suggestions=[
                    f"Use one of the supported reranking models: {available}",
                    "Check VoyageAI documentation for latest reranking model list"
                ]
            )

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
        return 128

    @property
    def max_input_length(self) -> int | None:
        """VoyageAI has input length limits."""
        return 32000

    async def _apply_rate_limit(self) -> None:
        """Apply basic rate limiting between API requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)
        self._last_request_time = time.time()

    async def embed_documents(
        self, texts: list[str], context: dict[str, Any] | None = None
    ) -> list[list[float]]:
        """Generate embeddings for documents with service layer integration."""
        context = context or {}
        cache_service = context.get("caching_service")
        if cache_service:
            cache_key = {
                "provider": "voyage_ai",
                "model": self._embedding_model,
                "texts": texts,
                "input_type": "document",
                "dimension": self._dimension,
            }
            cached_result = await cache_service.get(cache_key)
            if cached_result:
                logger.debug("Cache hit for %s VoyageAI embeddings", len(texts))
                return cached_result
        if rate_limiter := context.get("rate_limiting_service"):
            await rate_limiter.acquire("voyage_ai", len(texts))
        else:
            await self._rate_limit()
        try:
            result = self.client.embed(
                texts=texts,
                model=self._embedding_model,
                input_type="document",
                output_dimension=self._dimension,
            )
            embeddings = result.embeddings
            if cache_service:
                await cache_service.set(cache_key, embeddings, ttl=3600)
                logger.debug("Cached %s VoyageAI embeddings", len(texts))
        except Exception as e:
            logger.exception("Error generating VoyageAI embeddings")
            raise EmbeddingProviderError(
                "Failed to generate VoyageAI embeddings",
                provider_name="voyage_ai",
                operation="embed_documents",
                model_name=self._embedding_model,
                original_error=e,
                recovery_suggestions=[
                    "Check API key validity and network connectivity",
                    "Verify input text length is within limits",
                    "Check VoyageAI service status"
                ]
            ) from e
        return embeddings

    async def embed_query(self, text: str, context: dict[str, Any] | None = None) -> list[float]:
        """Generate embedding for search query with service layer integration."""
        context = context or {}
        cache_service = context.get("caching_service")
        if cache_service:
            cache_key = {
                "provider": "voyage_ai",
                "model": self._embedding_model,
                "text": text,
                "input_type": "query",
                "dimension": self._dimension,
            }
            cached_result = await cache_service.get(cache_key)
            if cached_result:
                logger.debug("Cache hit for VoyageAI query embedding")
                return cached_result
        if rate_limiter := context.get("rate_limiting_service"):
            await rate_limiter.acquire("voyage_ai", 1)
        else:
            await self._rate_limit()
        try:
            result = self.client.embed(
                texts=[text],
                model=self._embedding_model,
                input_type="query",
                output_dimension=self._dimension,
            )
            embedding = result.embeddings[0]
            if cache_service:
                await cache_service.set(cache_key, embedding, ttl=3600)
                logger.debug("Cached VoyageAI query embedding")
        except Exception as e:
            logger.exception("Error generating VoyageAI query embedding")
            raise EmbeddingProviderError(
                "Failed to generate VoyageAI query embedding",
                provider_name="voyage_ai",
                operation="embed_query",
                model_name=self._embedding_model,
                original_error=e,
                recovery_suggestions=[
                    "Check API key validity and network connectivity",
                    "Verify query text length is within limits",
                    "Check VoyageAI service status"
                ]
            ) from e
        return embedding

    @property
    def max_documents(self) -> int | None:
        """VoyageAI reranking has document limits."""
        return 1000

    @property
    def max_query_length(self) -> int | None:
        """VoyageAI has query length limits for reranking."""
        return 8000

    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[RerankResult]:
        """Rerank documents using VoyageAI with basic rate limiting."""

        def _raise_value_error(msg: str) -> None:
            raise ProviderConfigurationError(
                msg,
                provider_type="embedding",
                provider_name="voyage_ai",
                operation="rerank_validation",
                recovery_suggestions=[
                    "Reduce number of documents or query length",
                    "Check VoyageAI reranking limits documentation"
                ]
            )

        try:
            if len(documents) > (self.max_documents or float("inf")):
                _raise_value_error(f"Too many documents: {len(documents)} > {self.max_documents}")
            if len(query) > (self.max_query_length or float("inf")):
                _raise_value_error(f"Query too long: {len(query)} > {self.max_query_length}")
            result = self.client.rerank(
                query=query, documents=documents, model=self._rerank_model, top_k=top_k
            )
            rerank_results = [
                RerankResult(
                    index=item.index,
                    relevance_score=item.relevance_score,
                    document=item.document if hasattr(item, "document") else None,
                )
                for item in result.results
            ]
        except Exception as e:
            logger.exception("Error reranking with VoyageAI")
            raise EmbeddingProviderError(
                "Failed to rerank documents with VoyageAI",
                provider_name="voyage_ai",
                operation="rerank",
                model_name=self._rerank_model,
                original_error=e,
                recovery_suggestions=[
                    "Check API key validity and network connectivity",
                    "Verify document count and query length are within limits",
                    "Check VoyageAI service status"
                ]
            ) from e
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

    async def health_check(self) -> bool:
        """Check provider health by attempting a minimal API call.

        Returns:
            True if provider is healthy and operational, False otherwise
        """
        try:
            await self.embed_query("health_check")
            logger.debug("VoyageAI health check passed")
        except Exception as e:
            logger.exception("VoyageAI health check failed")
            return False
        else:
            return True

    @classmethod
    def check_availability(cls, capability: ProviderCapability) -> tuple[bool, str | None]:
        """Check if VoyageAI is available for the given capability."""
        if not VOYAGEAI_AVAILABLE:
            return (False, "voyageai package not installed (install with: uv add voyageai)")
        if capability in [ProviderCapability.EMBEDDING, ProviderCapability.RERANKING]:
            return (True, None)
        return (False, f"Capability {capability.value} not supported by VoyageAI")


register_provider_class(ProviderType.VOYAGE_AI, VoyageAIProvider)
