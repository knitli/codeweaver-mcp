# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
OpenAI provider implementation for embeddings.

Provides OpenAI embeddings and OpenAI-compatible API endpoints using the unified provider interface.
Supports custom dimensions, batch processing, and alternative base URLs for local/custom deployments.
"""

import logging

from typing import Any, ClassVar

from codeweaver.providers.base import EmbeddingProviderBase, ProviderCapability, ProviderInfo
from codeweaver.rate_limiter import calculate_embedding_tokens, rate_limited


try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None


logger = logging.getLogger(__name__)


class OpenAIProvider(EmbeddingProviderBase):
    """OpenAI provider for embeddings with support for custom endpoints."""

    # Provider metadata
    PROVIDER_NAME: ClassVar[str] = "openai"
    DISPLAY_NAME: ClassVar[str] = "OpenAI"
    DESCRIPTION: ClassVar[str] = (
        "OpenAI embeddings with support for custom dimensions and endpoints"
    )

    # Native model dimensions
    NATIVE_DIMENSIONS: ClassVar[dict[str, int]] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    # Supported models
    SUPPORTED_MODELS: ClassVar[list[str]] = list(NATIVE_DIMENSIONS.keys())

    # Rate limits (requests per minute)
    RATE_LIMITS: ClassVar[dict[str, int]] = {"embed_requests": 5000, "embed_tokens": 1000000}

    def __init__(self, config: dict[str, Any]):
        """Initialize OpenAI provider.

        Args:
            config: Configuration dictionary with keys:
                - api_key: OpenAI API key
                - model: Model name (default: text-embedding-3-small)
                - dimension: Custom dimension (optional, uses native if not specified)
                - base_url: Custom base URL for OpenAI-compatible APIs (optional)
                - custom_headers: Additional headers for requests (optional)
                - rate_limiter: RateLimiter instance (optional)
        """
        super().__init__(config)

        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: uv add openai")

        # Initialize OpenAI client
        client_kwargs = {"api_key": self.config["api_key"]}
        if self.config.get("base_url"):
            client_kwargs["base_url"] = self.config["base_url"]
        if self.config.get("custom_headers"):
            client_kwargs["default_headers"] = self.config["custom_headers"]

        self.client = openai.AsyncOpenAI(**client_kwargs)
        self.rate_limiter = self.config.get("rate_limiter")

        # Model configuration
        self._model = self.config.get("model", "text-embedding-3-small")
        self._dimension = self.config.get("dimension")

        # Use native dimensions if custom not specified
        if self._dimension is None:
            self._dimension = self.NATIVE_DIMENSIONS.get(self._model, 1536)

        # Set default model for legacy config compatibility
        if self._model == "voyage-code-3":  # Legacy default from old config
            self._model = "text-embedding-3-small"

    def _validate_config(self) -> None:
        """Validate OpenAI configuration."""
        if not self.config.get("api_key"):
            raise ValueError("OpenAI API key is required")

        model = self.config.get("model", "text-embedding-3-small")
        if model != "voyage-code-3" and model not in self.SUPPORTED_MODELS:
            # Allow voyage-code-3 for legacy compatibility (will be converted)
            available = ", ".join(self.SUPPORTED_MODELS)
            logger.warning(
                "Unknown OpenAI model: %s. Supported models: %s. "
                "Using text-embedding-3-small as fallback.",
                model,
                available,
            )

    # EmbeddingProvider implementation

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self.PROVIDER_NAME

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
        """OpenAI supports batch processing."""
        return 2048  # OpenAI's maximum batch size

    @property
    def max_input_length(self) -> int | None:
        """OpenAI has input length limits."""
        return 500000  # Conservative estimate in characters

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

                # Add dimensions parameter if custom dimension specified and model supports it
                if self._dimension and self._dimension != self._get_native_dimensions():
                    embedding_kwargs["dimensions"] = self._dimension

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
            embedding_kwargs = {"input": [text], "model": self._model}

            # Add dimensions parameter if custom dimension specified and model supports it
            if self._dimension and self._dimension != self._get_native_dimensions():
                embedding_kwargs["dimensions"] = self._dimension

            response = await self.client.embeddings.create(**embedding_kwargs)
            return response.data[0].embedding

        except Exception:
            logger.exception("Error generating OpenAI query embedding")
            raise

    def _get_native_dimensions(self) -> int:
        """Get the native dimensions for the current model."""
        return self.NATIVE_DIMENSIONS.get(self._model, self._dimension)

    # Provider info methods

    def get_provider_info(self) -> ProviderInfo:
        """Get information about OpenAI capabilities."""
        return ProviderInfo(
            name=self.PROVIDER_NAME,
            display_name=self.DISPLAY_NAME,
            description=self.DESCRIPTION,
            supported_capabilities=[
                ProviderCapability.EMBEDDING,
                ProviderCapability.BATCH_PROCESSING,
                ProviderCapability.CUSTOM_DIMENSIONS,
            ],
            default_models={"embedding": "text-embedding-3-small"},
            supported_models={"embedding": self.SUPPORTED_MODELS},
            rate_limits=self.RATE_LIMITS,
            requires_api_key=True,
            supports_batch_processing=True,
            max_batch_size=self.max_batch_size,
            max_input_length=self.max_input_length,
            native_dimensions=self.NATIVE_DIMENSIONS,
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
                ProviderCapability.BATCH_PROCESSING,
                ProviderCapability.CUSTOM_DIMENSIONS,
            ],
            default_models={"embedding": "text-embedding-3-small"},
            supported_models={"embedding": cls.SUPPORTED_MODELS},
            rate_limits=cls.RATE_LIMITS,
            requires_api_key=True,
            supports_batch_processing=True,
            max_batch_size=2048,
            max_input_length=500000,
            native_dimensions=cls.NATIVE_DIMENSIONS,
        )

    @classmethod
    def check_availability(cls, capability: ProviderCapability) -> tuple[bool, str | None]:
        """Check if OpenAI is available for the given capability."""
        if not OPENAI_AVAILABLE:
            return False, "openai package not installed (install with: uv add openai)"

        # Only embedding is supported
        if capability == ProviderCapability.EMBEDDING:
            return True, None

        return False, f"Capability {capability.value} not supported by OpenAI provider"
