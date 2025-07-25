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

import logging

from typing import Any

from codeweaver._types.provider_enums import ProviderCapability, ProviderType
from codeweaver._types.provider_registry import EmbeddingProviderInfo, register_provider_class
from codeweaver.providers.base import EmbeddingProviderBase
from codeweaver.providers.config import OpenAICompatibleConfig, OpenAIConfig
from codeweaver.rate_limiter import calculate_embedding_tokens, rate_limited


try:
    import openai  # type: ignore

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None


logger = logging.getLogger(__name__)


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
            raise ImportError("OpenAI library not available. Install with: uv add openai")

        # Service configuration
        self._service_name = self.config.get("service_name", "OpenAI-Compatible Service")
        self._base_url = self.config.get("base_url", "https://api.openai.com/v1")

        # Initialize OpenAI client
        client_kwargs = {
            "api_key": self.config["api_key"],
            "base_url": self._base_url,
        }
        if self.config.get("custom_headers"):
            client_kwargs["default_headers"] = self.config["custom_headers"]

        self.client = openai.AsyncOpenAI(**client_kwargs)
        self.rate_limiter = self.config.get("rate_limiter")

        # Model configuration
        self._model = self.config.get("model", "text-embedding-3-small")
        self._dimension = self.config.get("dimension")
        self._auto_discover_dimensions = self.config.get("auto_discover_dimensions", True)

        # Override service-specific limits if provided
        self._max_batch_size = self.config.get("max_batch_size")
        self._max_input_length = self.config.get("max_input_length")

        # Initialize dimensions
        self._initialize_dimensions()

    def _validate_config(self) -> None:
        """Validate OpenAI-compatible configuration."""
        if not self.config.get("api_key"):
            raise ValueError("API key is required for OpenAI-compatible provider")

        # Log service information for debugging
        logger.info(
            "Initializing %s with model '%s' at %s",
            self._service_name,
            self._model,
            self._base_url,
        )

    def _initialize_dimensions(self) -> None:
        """Initialize embedding dimensions for the configured model."""
        if self._dimension is not None:
            # User explicitly set dimension
            return

        # Try to get dimension from config or known models
        # For now, use a simplified lookup - this could be enhanced with a registry
        known_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        if self._model in known_dimensions:
            self._dimension = known_dimensions[self._model]
            logger.info(
                "Using known dimensions for model '%s': %d",
                self._model,
                self._dimension,
            )
            return

        # Auto-discover dimensions if enabled
        if self._auto_discover_dimensions:
            try:
                self._dimension = self._discover_model_dimensions()
                logger.info(
                    "Auto-discovered dimensions for model '%s': %d",
                    self._model,
                    self._dimension,
                )
            except Exception as e:
                logger.warning(
                    "Failed to auto-discover dimensions for model '%s': %s. Using default 1536.",
                    self._model,
                    e,
                )
                self._dimension = 1536
        else:
            # Default fallback
            self._dimension = 1536
            logger.info(
                "Using default dimensions for unknown model '%s': %d",
                self._model,
                self._dimension,
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
            response = await self.client.embeddings.create(
                input=["test"],
                model=self._model,
            )
            return len(response.data[0].embedding)

        # Run the async discovery in a sync context
        import contextlib

        loop = None
        with contextlib.suppress(RuntimeError):
            loop = asyncio.get_event_loop()

        if loop and loop.is_running():
            # If we're already in an async context, this is tricky
            # For now, just return a reasonable default
            raise RuntimeError("Cannot auto-discover dimensions in async context")
        return asyncio.run(_discover())

    # EmbeddingProvider implementation

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

        # Service-specific defaults based on base URL
        if "openai.com" in self._base_url:
            return 2048  # OpenAI's maximum
        if "openrouter.ai" in self._base_url:
            return 128   # OpenRouter's typical limit
        return 256 if "together.ai" in self._base_url else 128

    @property
    def max_input_length(self) -> int | None:
        """Get maximum input length for this service."""
        if self._max_input_length is not None:
            return self._max_input_length

        # Service-specific defaults based on base URL
        return 500000 if "openai.com" in self._base_url else 32000

    @rate_limited("openai_embed_documents", calculate_embedding_tokens)
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for documents with rate limiting."""
        try:
            # Process in batches if needed
            batch_size = min(self.max_batch_size or len(texts), len(texts))
            embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                embedding_kwargs = {"input": batch_texts, "model": self._model}

                # Add dimensions parameter if custom dimension specified and different from discovered
                if self._dimension and self._supports_custom_dimensions():
                    embedding_kwargs["dimensions"] = self._dimension

                response = await self.client.embeddings.create(**embedding_kwargs)

                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)

        except Exception:
            logger.exception("Error generating embeddings from %s", self._service_name)
            raise

        else:
            return embeddings

    @rate_limited("openai_embed_query", lambda self, text, **kwargs: max(1, len(text) // 4))
    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for search query with rate limiting."""
        try:
            embedding_kwargs = {"input": [text], "model": self._model}

            # Add dimensions parameter if custom dimension specified and different from discovered
            if self._dimension and self._supports_custom_dimensions():
                embedding_kwargs["dimensions"] = self._dimension

            response = await self.client.embeddings.create(**embedding_kwargs)
            return response.data[0].embedding

        except Exception:
            logger.exception("Error generating query embedding from %s", self._service_name)
            raise

    def _supports_custom_dimensions(self) -> bool:
        """Check if the current model/service supports custom dimensions.

        Returns:
            True if custom dimensions are supported
        """
        # OpenAI's v3 models support custom dimensions
        # Most other services don't support custom dimensions
        # This could be made configurable in the future
        return self._model.startswith("text-embedding-3-")

    # Provider info methods

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
            capabilities=None,  # Dynamic capabilities
            default_models={"embedding": self._model},
            supported_models={"embedding": [self._model]},  # Only the configured model
            rate_limits=None,  # Service-specific, not known generically
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
            capabilities=None,  # Dynamic capabilities
            default_models={"embedding": "text-embedding-3-small"},
            supported_models={"embedding": []},  # Any model supported by the endpoint
            rate_limits=None,  # Service-specific
            requires_api_key=True,
            max_batch_size=None,  # Service-specific
            max_input_length=None,  # Service-specific
            native_dimensions={},  # Dynamic discovery
        )

    @classmethod
    def check_availability(cls, capability: ProviderCapability) -> tuple[bool, str | None]:
        """Check if OpenAI-compatible provider is available for the given capability."""
        if not OPENAI_AVAILABLE:
            return False, "openai package not installed (install with: uv add openai)"

        # Only embedding is supported
        if capability == ProviderCapability.EMBEDDING:
            return True, None

        return False, f"Capability {capability.value} not supported by OpenAI-compatible provider"


# OpenAI provider (configured for official OpenAI service)
class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI provider preconfigured for the official OpenAI API.

    This provider simplifies configuration by setting OpenAI defaults.
    """

    def __init__(self, config: OpenAIConfig | dict[str, Any]):
        """Initialize OpenAI provider with OpenAI-specific defaults."""
        # Ensure we use OpenAI defaults if not specified
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


# Register both providers in the registry
register_provider_class(ProviderType.OPENAI, OpenAIProvider)
register_provider_class(ProviderType.OPENAI_COMPATIBLE, OpenAICompatibleProvider)
