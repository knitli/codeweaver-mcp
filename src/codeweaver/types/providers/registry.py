# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Provider registry with centralized metadata and capabilities.

Consolidates all provider information in one place, eliminating hardcoded
attributes scattered across provider implementations.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeweaver.types.providers.capabilities import ProviderCapabilities
from codeweaver.types.providers.enums import ModelFamily, ProviderCapability, ProviderType


if TYPE_CHECKING:
    from codeweaver.providers.base import EmbeddingProvider, RerankProvider


@dataclass
class EmbeddingProviderInfo:
    """Information about an embedding/reranking provider's capabilities and configuration.

    This is separate from the backend ProviderInfo class which is for vector databases.
    """

    name: str
    display_name: str
    description: str
    supported_capabilities: list[ProviderCapability]
    capabilities: ProviderCapabilities
    default_models: dict[str, str] | None = None  # capability -> default model
    supported_models: dict[str, list[str]] | None = None  # capability -> list of models
    rate_limits: dict[str, int] | None = None  # operation -> limit per minute
    requires_api_key: bool = True
    max_batch_size: int | None = None
    max_input_length: int | None = None
    native_dimensions: dict[str, int] | None = None  # model -> native dimensions


@dataclass
class ProviderRegistryEntry:
    """Information about a provider.

    Contains all metadata and capabilities for a specific provider,
    serving as the single source of truth for provider information.
    """

    provider_class: type["EmbeddingProvider | RerankProvider"] | None
    capabilities: ProviderCapabilities
    provider_type: ProviderType
    display_name: str
    description: str
    supported_models: dict[ModelFamily, list[str]] | None = None
    implemented: bool = True

    @property
    def is_available(self) -> bool:
        """Check if provider implementation is available."""
        return self.implemented and self.provider_class is not None


# Registry with full capability information
PROVIDER_REGISTRY: dict[ProviderType, ProviderRegistryEntry] = {
    ProviderType.VOYAGE_AI: ProviderRegistryEntry(
        provider_class=None,  # Will be set when VoyageAIProvider is imported
        capabilities=ProviderCapabilities(
            supports_embedding=True,
            supports_reranking=True,
            supports_batch_processing=True,
            supports_rate_limiting=True,
            supports_custom_dimensions=True,
            max_batch_size=128,
            max_input_length=32000,
            max_concurrent_requests=10,
            requires_api_key=True,
            requests_per_minute=100,
            tokens_per_minute=1000000,
            required_dependencies=["voyageai"],
            default_embedding_model="voyage-code-3",
            default_reranking_model="voyage-rerank-2",
            supported_embedding_models=[
                "voyage-code-3",
                "voyage-3",
                "voyage-3-lite",
                "voyage-large-2",
                "voyage-2",
            ],
            supported_reranking_models=["voyage-rerank-2", "voyage-rerank-lite-1"],
            native_dimensions={
                "voyage-code-3": 1024,
                "voyage-3": 1024,
                "voyage-3-lite": 512,
                "voyage-large-2": 1536,
                "voyage-2": 1024,
            },
        ),
        provider_type=ProviderType.VOYAGE_AI,
        display_name="Voyage AI",
        description="Best-in-class code embeddings and reranking",
        supported_models={
            ModelFamily.CODE_EMBEDDING: ["voyage-code-3", "voyage-3"],
            ModelFamily.TEXT_EMBEDDING: ["voyage-3-lite", "voyage-large-2", "voyage-2"],
            ModelFamily.RERANKING: ["voyage-rerank-2", "voyage-rerank-lite-1"],
        },
    ),
    ProviderType.OPENAI: ProviderRegistryEntry(
        provider_class=None,  # Will be set when OpenAIProvider is imported
        capabilities=ProviderCapabilities(
            supports_embedding=True,
            supports_reranking=False,
            supports_batch_processing=True,
            supports_rate_limiting=True,
            supports_custom_dimensions=True,
            max_batch_size=2048,
            max_input_length=8191,
            max_concurrent_requests=50,
            requires_api_key=True,
            requests_per_minute=3000,
            tokens_per_minute=1000000,
            required_dependencies=["openai"],
            default_embedding_model="text-embedding-3-small",
            supported_embedding_models=[
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ],
            supported_reranking_models=[],
            native_dimensions={
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536,
            },
        ),
        provider_type=ProviderType.OPENAI,
        display_name="OpenAI",
        description="OpenAI's text embedding models",
        supported_models={
            ModelFamily.TEXT_EMBEDDING: [
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ]
        },
    ),
    ProviderType.OPENAI_COMPATIBLE: ProviderRegistryEntry(
        provider_class=None,  # Will be set when OpenAICompatibleProvider is imported
        capabilities=ProviderCapabilities(
            supports_embedding=True,
            supports_reranking=False,
            supports_batch_processing=True,
            supports_rate_limiting=True,
            supports_custom_dimensions=False,  # Depends on the specific service
            max_batch_size=None,  # Service-specific
            max_input_length=None,  # Service-specific
            max_concurrent_requests=10,  # Conservative default
            requires_api_key=True,
            requests_per_minute=None,  # Service-specific
            tokens_per_minute=None,  # Service-specific
            required_dependencies=["openai"],
            default_embedding_model=None,  # Configurable
            supported_embedding_models=[],  # Any model supported by the endpoint
            supported_reranking_models=[],
            native_dimensions={},  # Auto-discovered or user-configured
        ),
        provider_type=ProviderType.OPENAI_COMPATIBLE,
        display_name="OpenAI-Compatible",
        description="Flexible provider for any OpenAI-compatible embedding API (OpenRouter, Together AI, etc.)",
        supported_models={},  # Dynamic based on endpoint
    ),
    ProviderType.COHERE: ProviderRegistryEntry(
        provider_class=None,  # Will be set when CohereProvider is imported
        capabilities=ProviderCapabilities(
            supports_embedding=True,
            supports_reranking=True,
            supports_batch_processing=True,
            supports_rate_limiting=True,
            supports_custom_dimensions=False,
            max_batch_size=96,
            max_input_length=2048,
            max_concurrent_requests=10,
            requires_api_key=True,
            requests_per_minute=100,
            tokens_per_minute=10000,
            required_dependencies=["cohere"],
            default_embedding_model="embed-english-v3.0",
            default_reranking_model="rerank-3",
            supported_embedding_models=[
                "embed-english-v3.0",
                "embed-multilingual-v3.0",
                "embed-english-light-v3.0",
                "embed-multilingual-light-v3.0",
            ],
            supported_reranking_models=["rerank-3", "rerank-multilingual-3", "rerank-english-3"],
            native_dimensions={
                "embed-english-v3.0": 1024,
                "embed-multilingual-v3.0": 1024,
                "embed-english-light-v3.0": 384,
                "embed-multilingual-light-v3.0": 384,
            },
        ),
        provider_type=ProviderType.COHERE,
        display_name="Cohere",
        description="Cohere's embedding and reranking models",
        supported_models={
            ModelFamily.TEXT_EMBEDDING: [
                "embed-english-v3.0",
                "embed-multilingual-v3.0",
                "embed-english-light-v3.0",
                "embed-multilingual-light-v3.0",
            ],
            ModelFamily.RERANKING: ["rerank-3", "rerank-multilingual-3", "rerank-english-3"],
        },
    ),
    ProviderType.HUGGINGFACE: ProviderRegistryEntry(
        provider_class=None,  # Will be set when HuggingFaceProvider is imported
        capabilities=ProviderCapabilities(
            supports_embedding=True,
            supports_reranking=False,
            supports_batch_processing=True,
            supports_rate_limiting=True,
            supports_local_inference=True,
            max_batch_size=32,
            max_input_length=512,
            max_concurrent_requests=5,
            requires_api_key=False,
            required_dependencies=["transformers", "torch"],
            optional_dependencies=["accelerate", "optimum"],
            default_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            supported_embedding_models=[
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
            ],
            supported_reranking_models=[],
            native_dimensions={
                "sentence-transformers/all-MiniLM-L6-v2": 384,
                "sentence-transformers/all-mpnet-base-v2": 768,
            },
        ),
        provider_type=ProviderType.HUGGINGFACE,
        display_name="Hugging Face",
        description="Hugging Face transformers with local inference",
        supported_models={
            ModelFamily.TEXT_EMBEDDING: [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
            ]
        },
    ),
    ProviderType.SENTENCE_TRANSFORMERS: ProviderRegistryEntry(
        provider_class=None,  # Will be set when SentenceTransformersProvider is imported
        capabilities=ProviderCapabilities(
            supports_embedding=True,
            supports_reranking=False,
            supports_batch_processing=True,
            supports_rate_limiting=False,
            supports_local_inference=True,
            max_batch_size=64,
            max_input_length=512,
            max_concurrent_requests=1,  # Local inference typically single-threaded
            requires_api_key=False,
            required_dependencies=["sentence-transformers"],
            default_embedding_model="all-MiniLM-L6-v2",
            supported_embedding_models=[
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "all-distilroberta-v1",
            ],
            supported_reranking_models=[],
            native_dimensions={
                "all-MiniLM-L6-v2": 384,
                "all-mpnet-base-v2": 768,
                "all-distilroberta-v1": 768,
            },
        ),
        provider_type=ProviderType.SENTENCE_TRANSFORMERS,
        display_name="Sentence Transformers",
        description="Local sentence transformers with no API requirements",
        supported_models={
            ModelFamily.TEXT_EMBEDDING: [
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "all-distilroberta-v1",
            ]
        },
    ),
    ProviderType.CUSTOM: ProviderRegistryEntry(
        provider_class=None,
        capabilities=ProviderCapabilities(
            supports_embedding=True,  # Default assumption for custom providers
            requires_api_key=False,  # Let custom providers decide
        ),
        provider_type=ProviderType.CUSTOM,
        display_name="Custom Provider",
        description="Custom provider implementation",
        supported_models={},
    ),
}


def get_provider_registry_entry(provider_type: ProviderType) -> ProviderRegistryEntry:
    """Get registry entry for a provider type.

    Args:
        provider_type: The provider type enum

    Returns:
        Registry entry with provider information

    Raises:
        KeyError: If provider type is not in registry
    """
    return PROVIDER_REGISTRY[provider_type]


def register_provider_class(provider_type: ProviderType, provider_class: type) -> None:
    """Register a provider implementation class.

    Args:
        provider_type: The provider type enum
        provider_class: The provider implementation class

    Raises:
        KeyError: If provider type is not in registry
    """
    if provider_type not in PROVIDER_REGISTRY:
        raise KeyError(f"Unknown provider type: {provider_type}")

    PROVIDER_REGISTRY[provider_type].provider_class = provider_class


def get_available_providers() -> list[ProviderType]:
    """Get list of providers that have implementations available.

    Returns:
        List of available provider types
    """
    return [
        provider_type for provider_type, entry in PROVIDER_REGISTRY.items() if entry.is_available
    ]
