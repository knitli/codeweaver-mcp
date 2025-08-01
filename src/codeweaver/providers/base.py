# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Base protocols and data structures for embedding, reranking, and NLP providers.

Defines universal interfaces that all provider implementations must follow,
enabling seamless integration of different embedding, reranking, and NLP services.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from codeweaver.types import EmbeddingProviderInfo, IntentType, RerankResult
from codeweaver.utils.decorators import require_implementation


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers.

    All embedding providers must implement this interface to be compatible
    with the CodeWeaver embedding system.
    """

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        ...

    @property
    def model_name(self) -> str:
        """Get the current model name."""
        ...

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        ...

    @property
    def max_batch_size(self) -> int | None:
        """Get the maximum batch size for this provider (None = unlimited)."""
        ...

    @property
    def max_input_length(self) -> int | None:
        """Get the maximum input length in characters (None = unlimited)."""
        ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors, one per input text

        Raises:
            ValueError: If input is invalid or too large
            RuntimeError: If the provider service is unavailable
        """
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector for the query

        Raises:
            ValueError: If input is invalid or too large
            RuntimeError: If the provider service is unavailable
        """
        ...

    def get_provider_info(self) -> EmbeddingProviderInfo:
        """Get information about this provider's capabilities."""
        ...


@runtime_checkable
class RerankProvider(Protocol):
    """Protocol for reranking providers.

    All reranking providers must implement this interface to be compatible
    with the CodeWeaver reranking system.
    """

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        ...

    @property
    def model_name(self) -> str:
        """Get the current reranking model name."""
        ...

    @property
    def max_documents(self) -> int | None:
        """Get the maximum number of documents that can be reranked (None = unlimited)."""
        ...

    @property
    def max_query_length(self) -> int | None:
        """Get the maximum query length in characters (None = unlimited)."""
        ...

    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[RerankResult]:
        """Rerank documents based on relevance to the query.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Maximum number of results to return (None = all)

        Returns:
            List of rerank results ordered by relevance (highest first)

        Raises:
            ValueError: If input is invalid or too large
            RuntimeError: If the provider service is unavailable
        """
        ...

    def get_provider_info(self) -> EmbeddingProviderInfo:
        """Get information about this provider's capabilities."""
        ...


class EmbeddingProviderBase(ABC):
    """Abstract base class for embedding providers with common functionality."""

    @require_implementation("embed_documents", "embed_query", "_validate_config")
    def __init__(self, config: Any):
        """Initialize the provider with configuration.

        Args:
            config: Configuration object (Pydantic model or dict)
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the provider configuration.

        Raises:
            ValueError: If configuration is invalid or missing required values
        """
        ...

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple documents."""
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the current model name."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the embedding dimension."""
        ...

    @property
    def max_batch_size(self) -> int | None:
        """Get the maximum batch size (default: None = unlimited)."""
        return None

    @property
    def max_input_length(self) -> int | None:
        """Get the maximum input length (default: None = unlimited)."""
        return None

    @abstractmethod
    def get_provider_info(self) -> EmbeddingProviderInfo:
        """Get information about this provider's capabilities."""
        ...


class RerankProviderBase(ABC):
    """Abstract base class for reranking providers with common functionality."""

    @require_implementation("rerank", "_validate_config")
    def __init__(self, config: Any):
        """Initialize the provider with configuration.

        Args:
            config: Configuration object (Pydantic model or dict)
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the provider configuration.

        Raises:
            ValueError: If configuration is invalid or missing required values
        """
        ...

    @abstractmethod
    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[RerankResult]:
        """Rerank documents based on relevance to the query."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the current reranking model name."""
        ...

    @property
    def max_documents(self) -> int | None:
        """Get the maximum number of documents (default: None = unlimited)."""
        return None

    @property
    def max_query_length(self) -> int | None:
        """Get the maximum query length (default: None = unlimited)."""
        return None

    @abstractmethod
    def get_provider_info(self) -> EmbeddingProviderInfo:
        """Get information about this provider's capabilities."""
        ...


class LocalEmbeddingProvider(EmbeddingProviderBase):
    """Base class for local embedding providers that don't require API keys."""

    def _validate_config(self) -> None:
        """Local providers typically have minimal validation requirements."""
        # Override in subclasses for specific validation


class CombinedProvider(EmbeddingProviderBase, RerankProviderBase):
    """Base class for providers that support both embedding and reranking."""

    def __init__(self, config: Any):
        """Initialize combined provider.

        Args:
            config: Configuration object containing settings for both capabilities
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate configuration for both embedding and reranking."""
        ...


@runtime_checkable
class NLPProvider(Protocol):
    """Protocol for natural language processing providers.

    All NLP providers must implement this interface to be compatible
    with the CodeWeaver NLP system.
    """

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        ...

    @property
    def model_name(self) -> str:
        """Get the current NLP model name."""
        ...

    @property
    def supported_languages(self) -> list[str]:
        """Get list of supported languages."""
        ...

    @property
    def max_text_length(self) -> int | None:
        """Get the maximum text length in characters (None = unlimited)."""
        ...

    async def process_text(
        self, text: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Process text and extract intent, entities, and embeddings.

        Args:
            text: Text to process
            context: Optional context information

        Returns:
            Dictionary containing:
            - intent_type: Detected IntentType or None
            - confidence: Confidence score for intent detection
            - entities: List of extracted entities
            - primary_target: Primary target extracted from text
            - embeddings: Text embeddings if available
            - metadata: Processing metadata

        Raises:
            ValueError: If input is invalid or too large
            RuntimeError: If the provider service is unavailable
        """
        ...

    async def classify_intent(self, text: str) -> tuple[IntentType | None, float]:
        """Classify intent with confidence score.

        Args:
            text: Text to classify

        Returns:
            Tuple of (intent_type, confidence_score)

        Raises:
            ValueError: If input is invalid or too large
            RuntimeError: If the provider service is unavailable
        """
        ...

    async def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract named entities from text.

        Args:
            text: Text to process

        Returns:
            List of entity dictionaries with keys:
            - text: Entity text
            - label: Entity label/type
            - start: Start character position
            - end: End character position
            - confidence: Confidence score

        Raises:
            ValueError: If input is invalid or too large
            RuntimeError: If the provider service is unavailable
        """
        ...

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get text embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors, one per input text

        Raises:
            ValueError: If input is invalid or too large
            RuntimeError: If the provider service is unavailable
        """
        ...

    async def get_embedding(self, text: str) -> list[float]:
        """Get text embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector for the text

        Raises:
            ValueError: If input is invalid or too large
            RuntimeError: If the provider service is unavailable
        """
        ...

    def get_available_models(self) -> list[dict[str, Any]]:
        """Get list of available NLP models.

        Returns:
            List of model dictionaries with keys:
            - name: Model name
            - language: Language code
            - capabilities: List of capabilities
            - model_size: Model size category
            - requires_download: Whether model requires download
            - description: Model description
        """
        ...

    async def switch_model(self, model_name: str) -> bool:
        """Switch to different model at runtime.

        Args:
            model_name: Name of model to switch to

        Returns:
            True if switch was successful, False otherwise

        Raises:
            ValueError: If model name is invalid
            RuntimeError: If model switching fails
        """
        ...

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary with model information:
            - name: Model name
            - language: Language code
            - capabilities: List of capabilities
            - pipeline: Pipeline components
            - metadata: Additional metadata
        """
        ...

    def get_provider_info(self) -> EmbeddingProviderInfo:
        """Get information about this provider's capabilities."""
        ...


class NLPProviderBase(ABC):
    """Abstract base class for NLP providers with common functionality."""

    @require_implementation(
        "process_text", "classify_intent", "extract_entities", "_validate_config"
    )
    def __init__(self, config: Any):
        """Initialize the provider with configuration.

        Args:
            config: Configuration object (Pydantic model or dict)
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the provider configuration.

        Raises:
            ValueError: If configuration is invalid or missing required values
        """
        ...

    @abstractmethod
    async def process_text(
        self, text: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Process text and extract intent, entities, and embeddings."""
        ...

    @abstractmethod
    async def classify_intent(self, text: str) -> tuple[IntentType | None, float]:
        """Classify intent with confidence score."""
        ...

    @abstractmethod
    async def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract named entities from text."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the current NLP model name."""
        ...

    @property
    def supported_languages(self) -> list[str]:
        """Get list of supported languages (default: English only)."""
        return ["en"]

    @property
    def max_text_length(self) -> int | None:
        """Get the maximum text length (default: None = unlimited)."""
        return None

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get text embeddings for multiple texts (default: empty embeddings)."""
        return [[0.0] * 300 for _ in texts]  # Default spaCy vector size

    async def get_embedding(self, text: str) -> list[float]:
        """Get text embedding for a single text (default: empty embedding)."""
        embeddings = await self.get_embeddings([text])
        return embeddings[0]

    def get_available_models(self) -> list[dict[str, Any]]:
        """Get list of available NLP models (default: current model only)."""
        return [
            {
                "name": self.model_name,
                "language": "en",
                "capabilities": [],
                "model_size": "unknown",
                "requires_download": False,
                "description": f"Current model: {self.model_name}",
            }
        ]

    async def switch_model(self, model_name: str) -> bool:
        """Switch to different model at runtime (default: not supported)."""
        return False

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model."""
        return {
            "name": self.model_name,
            "language": "en",
            "capabilities": [],
            "pipeline": [],
            "metadata": {},
        }

    @abstractmethod
    def get_provider_info(self) -> EmbeddingProviderInfo:
        """Get information about this provider's capabilities."""
        ...


class LocalNLPProvider(NLPProviderBase):
    """Base class for local NLP providers that don't require API keys."""

    def _validate_config(self) -> None:
        """Local providers typically have minimal validation requirements."""
        # Override in subclasses for specific validation
