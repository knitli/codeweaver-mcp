# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
SentenceTransformers provider implementation for local embeddings.

Provides local embedding generation using SentenceTransformers models with no API key required.
Supports GPU acceleration and model caching for improved performance.
"""

import logging

from typing import Any, ClassVar

from codeweaver.providers.base import LocalEmbeddingProvider, ProviderCapability, ProviderInfo


try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


logger = logging.getLogger(__name__)


class SentenceTransformersProvider(LocalEmbeddingProvider):
    """SentenceTransformers provider for local embeddings."""

    # Provider metadata
    PROVIDER_NAME: ClassVar[str] = "sentence-transformers"
    DISPLAY_NAME: ClassVar[str] = "SentenceTransformers"
    DESCRIPTION: ClassVar[str] = (
        "Local embedding models with no API key required, supports GPU acceleration"
    )

    # Popular code/text embedding models
    SUPPORTED_MODELS: ClassVar[dict[str, int]] = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "multi-qa-MiniLM-L6-cos-v1": 384,
        "multi-qa-mpnet-base-dot-v1": 768,
        "all-distilroberta-v1": 768,
        "msmarco-distilbert-base-tas-b": 768,
        "paraphrase-MiniLM-L6-v2": 384,
        "paraphrase-mpnet-base-v2": 768,
    }

    def __init__(self, config: dict[str, Any]):
        """Initialize SentenceTransformers provider.

        Args:
            config: Configuration dictionary with keys:
                - model: Model name (default: all-MiniLM-L6-v2)
                - device: Device to use ('cpu', 'cuda', 'auto') (default: auto)
                - normalize_embeddings: Whether to normalize embeddings (default: True)
                - batch_size: Batch size for processing (default: 32)
        """
        super().__init__(config)

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "SentenceTransformers library not available. "
                "Install with: uv add sentence-transformers"
            )

        # Model configuration
        self._model_name = self.config.get("model", "all-MiniLM-L6-v2")
        self._device = self.config.get("device", "auto")
        self._normalize_embeddings = self.config.get("normalize_embeddings", True)
        self._batch_size = self.config.get("batch_size", 32)

        # Load model
        self._model = SentenceTransformer(self._model_name, device=self._device)

        # Get actual dimension from model
        self._dimension = self._model.get_sentence_embedding_dimension()

    def _validate_local_config(self) -> None:
        """Validate SentenceTransformers configuration."""
        model = self.config.get("model", "all-MiniLM-L6-v2")
        if model not in self.SUPPORTED_MODELS:
            # Allow any model name, just warn about popular ones
            logger.info(
                "Using custom SentenceTransformers model: %s. Popular code models: %s",
                model,
                ", ".join(list(self.SUPPORTED_MODELS.keys())[:5]),
            )

        batch_size = self.config.get("batch_size", 32)
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")

    # EmbeddingProvider implementation

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self.PROVIDER_NAME

    @property
    def model_name(self) -> str:
        """Get the current model name."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension

    @property
    def max_batch_size(self) -> int | None:
        """SentenceTransformers can handle large batches, limited by memory."""
        return self._batch_size

    @property
    def max_input_length(self) -> int | None:
        """SentenceTransformers models have token limits, roughly estimate characters."""
        # Most models have 512 token limit, roughly 4 chars per token
        return 2000

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for documents."""
        try:
            # SentenceTransformers encode is synchronous, but we run in async context
            embeddings = self._model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize_embeddings,
                convert_to_numpy=True,
            )

            # Convert numpy arrays to lists
            return [embedding.tolist() for embedding in embeddings]

        except Exception:
            logger.exception("Error generating SentenceTransformers embeddings")
            raise

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for search query."""
        try:
            embedding = self._model.encode(
                [text],
                batch_size=1,
                normalize_embeddings=self._normalize_embeddings,
                convert_to_numpy=True,
            )

            return embedding[0].tolist()

        except Exception:
            logger.exception("Error generating SentenceTransformers query embedding")
            raise

    # Provider info methods

    def get_provider_info(self) -> ProviderInfo:
        """Get information about SentenceTransformers capabilities."""
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
                ProviderCapability.LOCAL_INFERENCE,
                ProviderCapability.BATCH_PROCESSING,
            ],
            default_models={"embedding": "all-MiniLM-L6-v2"},
            supported_models={"embedding": list(cls.SUPPORTED_MODELS.keys())},
            rate_limits=None,  # No rate limits for local models
            requires_api_key=False,
            supports_batch_processing=True,
            max_batch_size=32,
            max_input_length=2000,
            native_dimensions=cls.SUPPORTED_MODELS,
        )

    @classmethod
    def check_availability(cls, capability: ProviderCapability) -> tuple[bool, str | None]:
        """Check if SentenceTransformers is available for the given capability."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return (
                False,
                "sentence-transformers package not installed (install with: uv add sentence-transformers)",
            )

        # Only embedding is supported
        if capability == ProviderCapability.EMBEDDING:
            return True, None

        return False, f"Capability {capability.value} not supported by SentenceTransformers"
