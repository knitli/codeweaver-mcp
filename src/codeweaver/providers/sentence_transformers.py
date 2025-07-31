"""
SentenceTransformers provider implementation for local embeddings.

Provides local embedding generation using SentenceTransformers models with no API key required.
Supports GPU acceleration and model caching for improved performance.
"""

import logging

from typing import Any

from codeweaver.providers.base import LocalEmbeddingProvider
from codeweaver.providers.config import SentenceTransformersConfig
from codeweaver.types import (
    EmbeddingProviderInfo,
    ProviderCapability,
    ProviderType,
    get_provider_registry_entry,
    register_provider_class,
)
from codeweaver.utils.decorators import feature_flag_required


try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
logger = logging.getLogger(__name__)


@feature_flag_required("sentence-transformers", dependencies=["sentence-transformers"])
class SentenceTransformersProvider(LocalEmbeddingProvider):
    """SentenceTransformers provider for local embeddings."""

    def __init__(self, config: dict[str, Any] | SentenceTransformersConfig):
        """Initialize SentenceTransformers provider.

        Args:
            config: Configuration dictionary or SentenceTransformersConfig instance with settings for:
                - model: Model name
                - device: Device to use ('cpu', 'cuda', 'auto')
                - normalize_embeddings: Whether to normalize embeddings
                - batch_size: Batch size for processing
        """
        super().__init__(config)
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "SentenceTransformers library not available. Install with: uv add sentence-transformers"
            )
        self._registry_entry = get_provider_registry_entry(ProviderType.SENTENCE_TRANSFORMERS)
        if isinstance(config, dict):
            if "model" not in config:
                config["model"] = self._registry_entry.capabilities.default_embedding_model
            self._config = SentenceTransformersConfig(**config)
        else:
            self._config = config
        if self._config.model not in self._registry_entry.capabilities.supported_embedding_models:
            logger.info(
                "Using custom SentenceTransformers model: %s. Popular models: %s",
                self._config.model,
                ", ".join(list(self._registry_entry.capabilities.supported_embedding_models)[:5]),
            )
        self._model_name = self._config.model
        self._device = getattr(self._config, "device", "cpu")
        self._normalize_embeddings = self._config.normalize_embeddings
        self._batch_size = self._config.batch_size
        self._model = SentenceTransformer(self._model_name, device=self._device)
        self._dimension = self._model.get_sentence_embedding_dimension()

    def _validate_local_config(self) -> None:
        """Validate SentenceTransformers configuration."""

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return ProviderType.SENTENCE_TRANSFORMERS.value

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
        return 2000

    async def embed_documents(
        self, texts: list[str], *, context: dict | None = None
    ) -> list[list[float]]:
        """Generate embeddings for documents."""
        try:
            embeddings = self._model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize_embeddings,
                convert_to_numpy=True,
            )
        except Exception:
            logger.exception("Error generating SentenceTransformers embeddings")
            raise
        else:
            return [embedding.tolist() for embedding in embeddings]

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for search query."""
        try:
            embedding = self._model.encode(
                [text],
                batch_size=1,
                normalize_embeddings=self._normalize_embeddings,
                convert_to_numpy=True,
            )
        except Exception:
            logger.exception("Error generating SentenceTransformers query embedding")
            raise
        else:
            return embedding[0].tolist()

    def get_provider_info(self) -> EmbeddingProviderInfo:
        """Get information about SentenceTransformers capabilities from centralized registry."""
        capabilities = self._registry_entry.capabilities
        return EmbeddingProviderInfo(
            name=ProviderType.SENTENCE_TRANSFORMERS.value,
            display_name=self._registry_entry.display_name,
            description=self._registry_entry.description,
            supported_capabilities=[
                ProviderCapability.EMBEDDING,
                ProviderCapability.LOCAL_INFERENCE,
                ProviderCapability.BATCH_PROCESSING,
            ],
            capabilities=capabilities,
            default_models={"embedding": capabilities.default_embedding_model},
            supported_models={"embedding": capabilities.supported_embedding_models},
            rate_limits=None,
            requires_api_key=False,
            max_batch_size=capabilities.max_batch_size,
            max_input_length=capabilities.max_input_length,
            native_dimensions=capabilities.native_dimensions,
        )

    @classmethod
    def get_static_provider_info(cls) -> EmbeddingProviderInfo:
        """Get static provider information from centralized registry."""
        registry_entry = get_provider_registry_entry(ProviderType.SENTENCE_TRANSFORMERS)
        capabilities = registry_entry.capabilities
        return EmbeddingProviderInfo(
            name=ProviderType.SENTENCE_TRANSFORMERS.value,
            display_name=registry_entry.display_name,
            description=registry_entry.description,
            supported_capabilities=[
                ProviderCapability.EMBEDDING,
                ProviderCapability.LOCAL_INFERENCE,
                ProviderCapability.BATCH_PROCESSING,
            ],
            capabilities=capabilities,
            default_models={"embedding": capabilities.default_embedding_model},
            supported_models={"embedding": capabilities.supported_embedding_models},
            rate_limits=None,
            requires_api_key=False,
            max_batch_size=capabilities.max_batch_size,
            max_input_length=capabilities.max_input_length,
            native_dimensions=capabilities.native_dimensions,
        )

    @classmethod
    def check_availability(cls, capability: ProviderCapability) -> tuple[bool, str | None]:
        """Check if SentenceTransformers is available for the given capability."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return (
                False,
                "sentence-transformers package not installed (install with: uv add sentence-transformers)",
            )
        supported_capabilities = {
            ProviderCapability.EMBEDDING,
            ProviderCapability.LOCAL_INFERENCE,
            ProviderCapability.BATCH_PROCESSING,
        }
        if capability in supported_capabilities:
            return (True, None)
        return (False, f"Capability {capability.value} not supported by SentenceTransformers")


register_provider_class(ProviderType.SENTENCE_TRANSFORMERS, SentenceTransformersProvider)
