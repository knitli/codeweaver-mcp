# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Provider enums for embedding and reranking capabilities.

This module defines all enums related to provider capabilities, types, and model families.
Replaces string literals and TypedDict usage with proper type-safe enums.
"""

from dataclasses import dataclass

from codeweaver.types.base_enum import BaseEnum


@dataclass
class RerankResult:
    """Result from a reranking operation."""

    index: int
    relevance_score: float
    document: str | None = None


class ProviderCapability(BaseEnum):
    """Provider capability types.

    This enum defines the different capabilities that providers can support.
    Replaces the broken TypedDict that was causing runtime failures.
    """

    EMBEDDING = "embedding"  # Fixed naming to follow singular pattern
    RERANKING = "reranking"
    BATCH_PROCESSING = "batch_processing"
    RATE_LIMITING = "rate_limiting"
    STREAMING = "streaming"
    CUSTOM_DIMENSIONS = "custom_dimensions"
    LOCAL_INFERENCE = "local_inference"


class ProviderKind(BaseEnum):
    """Provider kind/role classification.

    Defines what kind of provider this is based on capabilities.
    """

    EMBEDDING = "embedding"  # Embedding-only provider
    RERANKING = "reranking"  # Reranking-only provider
    COMBINED = "combined"  # Both embedding and reranking
    LOCAL = "local"  # Local inference provider


class ProviderType(BaseEnum):
    """Provider types.

    Enum of all supported provider implementations.
    """

    VOYAGE_AI = "voyage-ai"  # Fixed naming for consistency
    OPENAI = "openai"
    OPENAI_COMPATIBLE = "openai-compatible"  # Generic OpenAI-compatible provider
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    CUSTOM = "custom"

    # Test/Mock providers
    MOCK_EMBEDDING = "mock_embedding"
    MOCK_RERANK = "mock_rerank"


class ModelFamily(BaseEnum):
    """Model families across providers.

    Categorizes models by their primary use case and capabilities.
    """

    CODE_EMBEDDING = "code_embedding"
    TEXT_EMBEDDING = "text_embedding"
    RERANKING = "reranking"
    MULTIMODAL = "multimodal"


class VoyageModels(BaseEnum):
    """VoyageAI supported models."""

    CODE_3 = "voyage-code-3"
    VOYAGE_3 = "voyage-3"
    VOYAGE_3_LITE = "voyage-3-lite"
    VOYAGE_LARGE_2 = "voyage-large-2"
    VOYAGE_2 = "voyage-2"

    @property
    def dimensions(self) -> int:
        """Get native dimensions for this model."""
        return {
            self.CODE_3: 1024,
            self.VOYAGE_3: 1024,
            self.VOYAGE_3_LITE: 512,
            self.VOYAGE_LARGE_2: 1536,
            self.VOYAGE_2: 1024,
        }[self]

    @property
    def model_family(self) -> ModelFamily:
        """Get model family."""
        if "code" in self.value:
            return ModelFamily.CODE_EMBEDDING
        return ModelFamily.TEXT_EMBEDDING


class VoyageRerankModels(BaseEnum):
    """VoyageAI reranking models."""

    RERANK_2 = "voyage-rerank-2"
    RERANK_LITE_1 = "voyage-rerank-lite-1"

    @property
    def model_family(self) -> ModelFamily:
        """Get model family."""
        return ModelFamily.RERANKING


class OpenAIModels(BaseEnum):
    """OpenAI supported models."""

    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"

    @property
    def dimensions(self) -> int:
        """Get native dimensions for this model."""
        return {
            self.TEXT_EMBEDDING_3_SMALL: 1536,
            self.TEXT_EMBEDDING_3_LARGE: 3072,
            self.TEXT_EMBEDDING_ADA_002: 1536,
        }[self]

    @property
    def model_family(self) -> ModelFamily:
        """Get model family."""
        return ModelFamily.TEXT_EMBEDDING


class CohereModels(BaseEnum):
    """Cohere supported models."""

    EMBED_ENGLISH_V3 = "embed-english-v3.0"
    EMBED_MULTILINGUAL_V3 = "embed-multilingual-v3.0"
    EMBED_ENGLISH_LIGHT_V3 = "embed-english-light-v3.0"
    EMBED_MULTILINGUAL_LIGHT_V3 = "embed-multilingual-light-v3.0"

    @property
    def dimensions(self) -> int:
        """Get native dimensions for this model."""
        return {
            self.EMBED_ENGLISH_V3: 1024,
            self.EMBED_MULTILINGUAL_V3: 1024,
            self.EMBED_ENGLISH_LIGHT_V3: 384,
            self.EMBED_MULTILINGUAL_LIGHT_V3: 384,
        }[self]

    @property
    def model_family(self) -> ModelFamily:
        """Get model family."""
        return ModelFamily.TEXT_EMBEDDING


class CohereRerankModels(BaseEnum):
    """Cohere reranking models."""

    RERANK_3 = "rerank-3"
    RERANK_MULTILINGUAL_3 = "rerank-multilingual-3"
    RERANK_ENGLISH_3 = "rerank-english-3"

    @property
    def model_family(self) -> ModelFamily:
        """Get model family."""
        return ModelFamily.RERANKING
