# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Provider system for embeddings and reranking.

Provides extensible abstractions for multiple embedding and reranking providers
with unified interfaces, registry system, and backward compatibility.
"""

from codeweaver.providers.base import (
    EmbeddingProvider,
    ProviderCapability,
    ProviderInfo,
    RerankProvider,
    RerankResult,
)
from codeweaver.providers.factory import ProviderFactory, ProviderRegistry, get_provider_factory


__all__ = [
    "EmbeddingProvider",
    "ProviderCapability",
    "ProviderFactory",
    "ProviderInfo",
    "ProviderRegistry",
    "RerankProvider",
    "RerankResult",
    "get_provider_factory",
]
