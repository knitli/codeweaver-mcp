# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Provider system for embeddings and reranking.

Provides extensible abstractions for multiple embedding and reranking providers
with unified interfaces, registry system, and backward compatibility.
"""

from codeweaver.cw_types import (
    EmbeddingProviderInfo,
    ProviderCapabilities,
    ProviderCapability,
    ProviderType,
    RerankResult,
)
from codeweaver.providers.base import EmbeddingProvider, RerankProvider
from codeweaver.providers.custom import (
    EnhancedProviderRegistry,
    ProviderSDK,
    ValidationResult,
    register_combined_provider,
    register_embedding_provider,
    register_reranking_provider,
)
from codeweaver.providers.factory import ProviderFactory, ProviderRegistry, get_provider_factory
from codeweaver.providers.nlp import SpaCyProvider
from codeweaver.providers.providers import (
    CohereProvider,
    HuggingFaceProvider,
    OpenAICompatibleProvider,
    SentenceTransformersProvider,
    VoyageAIProvider,
)


__all__ = [
    "CohereProvider",
    "EmbeddingProvider",
    "EmbeddingProviderInfo",
    "EnhancedProviderRegistry",
    "HuggingFaceProvider",
    "OpenAICompatibleProvider",
    "ProviderCapabilities",
    "ProviderCapability",
    "ProviderFactory",
    "ProviderRegistry",
    "ProviderSDK",
    "ProviderType",
    "RerankProvider",
    "RerankResult",
    "SentenceTransformersProvider",
    "SpaCyProvider",
    "ValidationResult",
    "VoyageAIProvider",
    "get_provider_factory",
    "register_combined_provider",
    "register_embedding_provider",
    "register_reranking_provider",
]
