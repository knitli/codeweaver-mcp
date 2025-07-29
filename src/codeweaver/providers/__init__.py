# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Provider system for embeddings and reranking.

Provides extensible abstractions for multiple embedding and reranking providers
with unified interfaces, registry system, and backward compatibility.
"""

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
from codeweaver.types import (
    EmbeddingProviderInfo,
    ProviderCapabilities,
    ProviderCapability,
    ProviderType,
    RerankResult,
)


__all__ = [
    "EmbeddingProvider",
    "EmbeddingProviderInfo",
    "EnhancedProviderRegistry",
    "ProviderCapabilities",
    "ProviderCapability",
    "ProviderFactory",
    "ProviderRegistry",
    "ProviderSDK",
    "ProviderType",
    "RerankProvider",
    "RerankResult",
    "ValidationResult",
    "get_provider_factory",
    "register_combined_provider",
    "register_embedding_provider",
    "register_reranking_provider",
]
