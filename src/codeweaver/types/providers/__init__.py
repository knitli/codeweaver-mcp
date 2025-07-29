# SPDX-FileCopyrightText: 2025 Knitli Inc.
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Types for embedding and reranking providers.
"""

from codeweaver.types.providers.capabilities import ProviderCapabilities
from codeweaver.types.providers.enums import (
    CohereModels,
    CohereRerankModels,
    ModelFamily,
    OpenAIModels,
    ProviderCapability,
    ProviderKind,
    ProviderType,
    RerankResult,
    VoyageModels,
    VoyageRerankModels,
)
from codeweaver.types.providers.registry import (
    PROVIDER_REGISTRY,
    EmbeddingProviderInfo,
    ProviderRegistryEntry,
    get_available_providers,
    get_provider_registry_entry,
    register_provider_class,
)


__all__ = [
    "PROVIDER_REGISTRY",
    "CohereModels",
    "CohereRerankModels",
    "EmbeddingProviderInfo",
    "ModelFamily",
    "OpenAIModels",
    "ProviderCapabilities",
    "ProviderCapability",
    "ProviderKind",
    "ProviderRegistryEntry",
    "ProviderType",
    "RerankResult",
    "VoyageModels",
    "VoyageRerankModels",
    "get_available_providers",
    "get_provider_registry_entry",
    "register_provider_class",
]
