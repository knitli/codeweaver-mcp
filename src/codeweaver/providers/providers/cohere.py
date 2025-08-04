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
from codeweaver.providers.config import CohereConfig
from codeweaver.utils.decorators import feature_flag_required


try:
    import cohere

    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    cohere = None
logger = logging.getLogger(__name__)


@feature_flag_required("cohere", dependencies=["cohere"])
class CohereProvider(CombinedProvider):
    """Cohere provider supporting both embeddings and reranking."""

    def __init__(self, config: CohereConfig | dict[str, Any]):
        """Initialize Cohere provider.

        Args:
            config: CohereConfig instance or configuration dictionary
        """
        super().__init__(config)
        if not COHERE_AVAILABLE:
            raise EmbeddingProviderError(
                "Cohere library not available",
                provider_name="cohere",
                operation="initialization",
                recovery_suggestions=["Install with: uv add cohere"],
            )
        self.client = cohere.Client(api_key=self.config["api_key"])
        self.rate_limiter = self.config.get("rate_limiter")
        self._registry_entry = get_provider_registry_entry(ProviderType.COHERE)
        self._capabilities = self._registry_entry.capabilities
        self._embedding_model = self.config.get("model", self._capabilities.default_embedding_model)
        self._rerank_model = self.config.get(
            "rerank_model", self._capabilities.default_reranking_model
        )
        self._dimension = self._capabilities.native_dimensions.get(self._embedding_model, 1024)

    def _validate_config(self) -> None:
        """Validate Cohere configuration."""
        if not self.config.get("api_key"):
            raise ProviderConfigurationError(
                "Cohere API key is required",
                provider_type="embedding",
                provider_name="cohere",
                operation="validation",
                recovery_suggestions=[
                    "Set CW_EMBEDDING_API_KEY environment variable or provide api_key in config",
                    "Get API key from https://dashboard.cohere.ai/api-keys",
                ],
            )
        embedding_model = self.config.get("model", self._capabilities.default_embedding_model)
        if embedding_model not in self._capabilities.supported_embedding_models:
            available = ", ".join(self._capabilities.supported_embedding_models)
            raise ProviderCompatibilityError(
                f"Unknown Cohere embedding model: {embedding_model}",
                provider_type="embedding",
                provider_name="cohere",
                operation="model_validation",
                recovery_suggestions=[
                    f"Use one of the supported models: {available}",
                    "Check Cohere documentation for latest model list",
                ],
            )
        rerank_model = self.config.get("rerank_model", self._capabilities.default_reranking_model)
        if rerank_model not in self._capabilities.supported_reranking_models:
            available = ", ".join(self._capabilities.supported_reranking_models)
            raise ProviderCompatibilityError(
                f"Unknown Cohere reranking model: {rerank_model}",
                provider_type="embedding",
                provider_name="cohere",
                operation="rerank_model_validation",
                recovery_suggestions=[
                    f"Use one of the supported reranking models: {available}",
                    "Check Cohere documentation for latest reranking model list",
                ],
            )

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
        return 96

    @property
    def max_input_length(self) -> int | None:
        """Cohere has input length limits."""
        return 500000

    async def embed_documents(
        self, texts: list[str], context: dict[str, Any] | None = None
    ) -> list[list[float]]:
        """Generate embeddings for documents with service layer integration."""
        context = context or {}
        cache_service = context.get("caching_service")
        if cache_service:
            cache_key = {
                "provider": "cohere",
                "model": self._embedding_model,
                "texts": texts,
                "input_type": "search_document",
            }
            cached_result = await cache_service.get(cache_key)
            if cached_result:
                logger.debug("Cache hit for %s Cohere embeddings", len(texts))
                return cached_result
        if rate_limiter := context.get("rate_limiting_service"):
            await rate_limiter.acquire("cohere", len(texts))
        try:
            response = self.client.embed(
                texts=texts, model=self._embedding_model, input_type="search_document"
            )
            embeddings = response.embeddings
            if cache_service:
                await cache_service.set(cache_key, embeddings, ttl=3600)
                logger.debug("Cached %s Cohere embeddings", len(texts))
        except Exception as e:
            logger.exception("Error generating Cohere embeddings")
            raise EmbeddingProviderError(
                "Failed to generate Cohere embeddings",
                provider_name="cohere",
                operation="embed_documents",
                model_name=self._embedding_model,
                original_error=e,
                recovery_suggestions=[
                    "Check API key validity and network connectivity",
                    "Verify input text length is within limits",
                    "Check Cohere service status",
                ],
            ) from e
        else:
            return embeddings

    async def embed_query(self, text: str, context: dict[str, Any] | None = None) -> list[float]:
        """Generate embedding for search query with service layer integration."""
        context = context or {}
        cache_service = context.get("caching_service")
        if cache_service:
            cache_key = {
                "provider": "cohere",
                "model": self._embedding_model,
                "text": text,
                "input_type": "search_query",
            }
            cached_result = await cache_service.get(cache_key)
            if cached_result:
                logger.debug("Cache hit for Cohere query embedding")
                return cached_result
        if rate_limiter := context.get("rate_limiting_service"):
            await rate_limiter.acquire("cohere", 1)
        try:
            response = self.client.embed(
                texts=[text], model=self._embedding_model, input_type="search_query"
            )
            embedding = response.embeddings[0]
            if cache_service:
                await cache_service.set(cache_key, embedding, ttl=3600)
                logger.debug("Cached Cohere query embedding")
        except Exception as e:
            logger.exception("Error generating Cohere query embedding")
            raise EmbeddingProviderError(
                "Failed to generate Cohere query embedding",
                provider_name="cohere",
                operation="embed_query",
                model_name=self._embedding_model,
                original_error=e,
                recovery_suggestions=[
                    "Check API key validity and network connectivity",
                    "Verify query text length is within limits",
                    "Check Cohere service status",
                ],
            ) from e
        else:
            return embedding

    @property
    def max_documents(self) -> int | None:
        """Cohere reranking has document limits."""
        return 1000

    @property
    def max_query_length(self) -> int | None:
        """Cohere has query length limits for reranking."""
        return 10000

    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[RerankResult]:
        """Rerank documents using Cohere."""

        def _raise_value_error(msg: str) -> None:
            raise ProviderConfigurationError(
                msg,
                provider_type="embedding",
                provider_name="cohere",
                operation="rerank_validation",
                recovery_suggestions=[
                    "Reduce number of documents or query length",
                    "Check Cohere reranking limits documentation",
                ],
            )

        try:
            if len(documents) > (self.max_documents or float("inf")):
                _raise_value_error(f"Too many documents: {len(documents)} > {self.max_documents}")
            if len(query) > (self.max_query_length or float("inf")):
                _raise_value_error(f"Query too long: {len(query)} > {self.max_query_length}")
            response = self.client.rerank(
                query=query, documents=documents, model=self._rerank_model, top_k=top_k
            )
            rerank_results = []
            rerank_results.extend(
                RerankResult(
                    index=item.index,
                    relevance_score=item.relevance_score,
                    document=getattr(item, "document", None),
                )
                for item in response.results
            )
        except Exception as e:
            logger.exception("Error reranking with Cohere")
            raise EmbeddingProviderError(
                "Failed to rerank documents with Cohere",
                provider_name="cohere",
                operation="rerank",
                model_name=self._rerank_model,
                original_error=e,
                recovery_suggestions=[
                    "Check API key validity and network connectivity",
                    "Verify document count and query length are within limits",
                    "Check Cohere service status",
                ],
            ) from e
        else:
            return rerank_results

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

    async def health_check(self) -> bool:
        """Check provider health by attempting a minimal API call.

        Returns:
            True if provider is healthy and operational, False otherwise
        """
        try:
            await self.embed_query("health_check")
            logger.debug("Cohere health check passed")
        except Exception:
            logger.exception("Cohere health check failed")
            return False
        else:
            return True

    @classmethod
    def check_availability(cls, capability: ProviderCapability) -> tuple[bool, str | None]:
        """Check if Cohere is available for the given capability."""
        if not COHERE_AVAILABLE:
            return (False, "cohere package not installed (install with: uv add cohere)")
        if capability in [ProviderCapability.EMBEDDING, ProviderCapability.RERANKING]:
            return (True, None)
        return (False, f"Capability {capability.value} not supported by Cohere")


register_provider_class(ProviderType.COHERE, CohereProvider)
