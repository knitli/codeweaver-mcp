# sourcery skip: avoid-global-variables
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
OpenAI-compatible provider implementation for embeddings.

Provides a flexible provider that works with any OpenAI-compatible API endpoint,
including OpenAI, OpenRouter, Together AI, local deployments, and custom services.
Supports arbitrary models, dynamic capability discovery, and service-agnostic configuration.
"""

import asyncio
import logging
import time

from typing import Any

from codeweaver.cw_types import (
    EmbeddingProviderError,
    EmbeddingProviderInfo,
    OpenAIModel,
    ProviderCapability,
    ProviderCompatibilityError,
    ProviderConfigurationError,
    ProviderType,
    register_provider_class,
)
from codeweaver.providers.base import EmbeddingProviderBase
from codeweaver.providers.config import OpenAICompatibleConfig, OpenAIConfig
from codeweaver.utils.decorators import feature_flag_required


try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None
logger = logging.getLogger(__name__)


@feature_flag_required("openai", dependencies=["openai"])
class OpenAICompatibleProvider(EmbeddingProviderBase):
    """OpenAI-compatible provider for embeddings.

    Works with any service that implements the OpenAI embeddings API,
    including OpenAI, OpenRouter, Together AI, and custom deployments.
    Supports dynamic model discovery and arbitrary model configurations.
    """

    def __init__(self, config: OpenAICompatibleConfig | dict[str, Any]):
        """Initialize OpenAI-compatible provider.

        Args:
            config: OpenAICompatibleConfig instance or configuration dictionary
        """
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise EmbeddingProviderError(
                "OpenAI library not available",
                provider_name="openai_compatible",
                operation="initialization",
                recovery_suggestions=["Install with: uv add openai"],
            )
        self._service_name = self.config.get("service_name", "OpenAI-Compatible Service")
        self._base_url = self.config.get("base_url", "https://api.openai.com/v1")
        client_kwargs = {"api_key": self.config["api_key"], "base_url": self._base_url}
        if self.config.get("custom_headers"):
            client_kwargs["default_headers"] = self.config["custom_headers"]
        self.client = openai.AsyncOpenAI(**client_kwargs)
        self._last_request_time = 0.0
        self._min_request_interval = 0.1
        self._model = self.config.get("model", OpenAIModel.TEXT_EMBEDDING_3_SMALL)
        self._dimension = self.config.get("dimension")
        self._auto_discover_dimensions = self.config.get("auto_discover_dimensions", True)
        self._max_batch_size = self.config.get("max_batch_size")
        self._max_input_length = self.config.get("max_input_length")
        self._initialize_dimensions()

    def _validate_config(self) -> None:
        """Validate OpenAI-compatible configuration."""
        if not self.config.get("api_key"):
            raise ProviderConfigurationError(
                "API key is required for OpenAI-compatible provider",
                provider_type="embedding",
                provider_name="openai_compatible",
                operation="validation",
                recovery_suggestions=[
                    "Set CW_EMBEDDING_API_KEY environment variable or provide api_key in config",
                    "For OpenAI: Get API key from https://platform.openai.com/api-keys",
                ],
            )
        logger.info(
            "Initializing %s with model '%s' at %s", self._service_name, self._model, self._base_url
        )

    def _initialize_dimensions(self) -> None:
        """Initialize embedding dimensions for the configured model."""
        if self._dimension is not None:
            return
        if isinstance(self._model, str) and self._model in OpenAIModel.get_values():
            self._model = OpenAIModel.from_string(self._model)
        if isinstance(self._model, OpenAIModel):
            self._dimension = OpenAIModel.dimensions
        if self._auto_discover_dimensions:
            try:
                self._dimension = self._discover_model_dimensions()
                logger.info(
                    "Auto-discovered dimensions for model '%s': %d", self._model, self._dimension
                )
            except Exception as e:
                logger.warning(
                    "Failed to auto-discover dimensions for model '%s': %s. Using default 1536.",
                    self._model,
                    e,
                )
                self._dimension = 1536
        else:
            self._dimension = 1536
            logger.info(
                "Using default dimensions for unknown model '%s': %d", self._model, self._dimension
            )

    def _discover_model_dimensions(self) -> int:
        """Attempt to discover model dimensions by making a test embedding call.

        Returns:
            The discovered dimension count

        Raises:
            Exception: If discovery fails
        """
        import asyncio

        async def _discover():
            response = await self.client.embeddings.create(input=["test"], model=self._model)
            return len(response.data[0].embedding)

        import contextlib

        loop = None
        with contextlib.suppress(RuntimeError):
            loop = asyncio.get_event_loop()
        if loop and loop.is_running():
            raise ProviderCompatibilityError(
                "Cannot auto-discover dimensions in async context",
                provider_type="embedding",
                provider_name="openai_compatible",
                operation="dimension_discovery",
                recovery_suggestions=[
                    "Set dimension explicitly in configuration",
                    "Disable auto_discover_dimensions in config",
                    "Use synchronous initialization context",
                ],
            )
        return asyncio.run(_discover())

    async def _apply_rate_limit(self) -> None:
        """Apply basic rate limiting between API requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)
        self._last_request_time = time.time()

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return ProviderType.OPENAI_COMPATIBLE.value

    @property
    def model_name(self) -> str:
        """Get the current model name."""
        return self._model

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension

    @property
    def max_batch_size(self) -> int | None:
        """Get maximum batch size for this service."""
        if self._max_batch_size is not None:
            return self._max_batch_size
        if "openai.com" in self._base_url:
            return 2048
        if "openrouter.ai" in self._base_url:
            return 128
        return 256 if "together.ai" in self._base_url else 128

    @property
    def max_input_length(self) -> int | None:
        """Get maximum input length for this service."""
        if self._max_input_length is not None:
            return self._max_input_length
        return 500000 if "openai.com" in self._base_url else 32000

    async def embed_documents(
        self, texts: list[str], context: dict[str, Any] | None = None
    ) -> list[list[float]]:
        """Generate embeddings for documents with service layer integration."""
        context = context or {}
        cache_service = context.get("caching_service")
        if cache_service:
            cache_key = {
                "provider": "openai",
                "model": self._model,
                "texts": texts,
                "dimension": self._dimension,
            }
            cached_result = await cache_service.get(cache_key)
            if cached_result:
                logger.debug("Cache hit for %s OpenAI embeddings", len(texts))
                return cached_result
        if rate_limiter := context.get("rate_limiting_service"):
            await rate_limiter.acquire("openai", len(texts))
        else:
            await self._apply_rate_limit()
        try:
            batch_size = min(self.max_batch_size or len(texts), len(texts))
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                embedding_kwargs = {"input": batch_texts, "model": self._model}
                if self._dimension and self._supports_custom_dimensions():
                    embedding_kwargs["dimensions"] = self._dimension
                response = await self.client.embeddings.create(**embedding_kwargs)
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
            if cache_service:
                await cache_service.set(cache_key, embeddings, ttl=3600)
                logger.debug("Cached %s OpenAI embeddings", len(texts))
        except Exception as e:
            logger.exception("Error generating embeddings from %s", self._service_name)
            raise EmbeddingProviderError(
                f"Failed to generate embeddings from {self._service_name}",
                provider_name="openai_compatible",
                operation="embed_documents",
                model_name=self._model,
                original_error=e,
                recovery_suggestions=[
                    "Check API key validity and service availability",
                    "Verify base_url is correct for your service",
                    "Check input text length and batch size limits",
                    "Ensure model name is supported by the service",
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
                "provider": "openai",
                "model": self._model,
                "text": text,
                "dimension": self._dimension,
            }
            cached_result = await cache_service.get(cache_key)
            if cached_result:
                logger.debug("Cache hit for OpenAI query embedding")
                return cached_result
        if rate_limiter := context.get("rate_limiting_service"):
            await rate_limiter.acquire("openai", 1)
        else:
            await self._apply_rate_limit()
        try:
            embedding_kwargs = {"input": [text], "model": self._model}
            if self._dimension and self._supports_custom_dimensions():
                embedding_kwargs["dimensions"] = self._dimension
            response = await self.client.embeddings.create(**embedding_kwargs)
            embedding = response.data[0].embedding
            if cache_service:
                await cache_service.set(cache_key, embedding, ttl=3600)
                logger.debug("Cached OpenAI query embedding")
        except Exception as e:
            logger.exception("Error generating query embedding from %s", self._service_name)
            raise EmbeddingProviderError(
                f"Failed to generate query embedding from {self._service_name}",
                provider_name="openai_compatible",
                operation="embed_query",
                model_name=self._model,
                original_error=e,
                recovery_suggestions=[
                    "Check API key validity and service availability",
                    "Verify base_url is correct for your service",
                    "Check query text length limits",
                    "Ensure model name is supported by the service",
                ],
            ) from e
        else:
            return embedding

    def _supports_custom_dimensions(self) -> bool:
        """Check if the current model/service supports custom dimensions.

        Returns:
            True if custom dimensions are supported
        """
        return self._model.startswith("text-embedding-3-")

    def get_provider_info(self) -> EmbeddingProviderInfo:
        """Get information about this OpenAI-compatible provider instance."""
        return EmbeddingProviderInfo(
            name=ProviderType.OPENAI_COMPATIBLE.value,
            display_name=f"{self._service_name} (OpenAI-Compatible)",
            description=f"OpenAI-compatible embeddings via {self._service_name} at {self._base_url}",
            supported_capabilities=[
                ProviderCapability.EMBEDDING,
                ProviderCapability.BATCH_PROCESSING,
            ],
            capabilities=None,
            default_models={"embedding": self._model},
            supported_models={"embedding": [self._model]},
            rate_limits=None,
            requires_api_key=True,
            max_batch_size=self.max_batch_size,
            max_input_length=self.max_input_length,
            native_dimensions={self._model: self._dimension},
        )

    @classmethod
    def get_static_provider_info(cls) -> EmbeddingProviderInfo:
        """Get static provider information for OpenAI-compatible providers."""
        return EmbeddingProviderInfo(
            name=ProviderType.OPENAI_COMPATIBLE.value,
            display_name="OpenAI-Compatible Provider",
            description="Flexible provider for any OpenAI-compatible embedding API",
            supported_capabilities=[
                ProviderCapability.EMBEDDING,
                ProviderCapability.BATCH_PROCESSING,
            ],
            capabilities=None,
            default_models={"embedding": "text-embedding-3-small"},
            supported_models={"embedding": []},
            rate_limits=None,
            requires_api_key=True,
            max_batch_size=None,
            max_input_length=None,
            native_dimensions={},
        )

    async def health_check(self) -> bool:
        """Check provider health by attempting a minimal API call.

        Returns:
            True if provider is healthy and operational, False otherwise
        """
        try:
            await self.embed_query("health_check")
            logger.debug("OpenAI-compatible provider health check passed")
        except Exception:
            logger.exception("OpenAI-compatible provider health check failed")
            return False
        else:
            return True

    @classmethod
    def check_availability(cls, capability: ProviderCapability) -> tuple[bool, str | None]:
        """Check if OpenAI-compatible provider is available for the given capability."""
        if not OPENAI_AVAILABLE:
            return (False, "openai package not installed (install with: uv add openai)")
        if capability == ProviderCapability.EMBEDDING:
            return (True, None)
        return (False, f"Capability {capability.value} not supported by OpenAI-compatible provider")


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI provider preconfigured for the official OpenAI API.

    This provider simplifies configuration by setting OpenAI defaults.
    """

    def __init__(self, config: OpenAIConfig | dict[str, Any]):
        """Initialize OpenAI provider with OpenAI-specific defaults."""
        openai_config = {
            "base_url": "https://api.openai.com/v1",
            "service_name": "OpenAI",
            **config,
        }
        super().__init__(openai_config)

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return ProviderType.OPENAI.value


register_provider_class(ProviderType.OPENAI, OpenAIProvider)
register_provider_class(ProviderType.OPENAI_COMPATIBLE, OpenAICompatibleProvider)
