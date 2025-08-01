# SPDX-FileCopyrightText: 2025 Knitli Inc.
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Types for embedding and reranking providers.
"""

from codeweaver.cw_types.providers.capabilities import ProviderCapabilities
from codeweaver.cw_types.providers.enums import (
    CohereModel,
    CohereRerankModel,
    ModelFamily,
    NLPCapability,
    NLPModelSize,
    OpenAIModel,
    ProviderCapability,
    ProviderKind,
    ProviderType,
    VoyageModel,
    VoyageRerankModel,
)
from codeweaver.cw_types.providers.registry import (
    PROVIDER_REGISTRY,
    EmbeddingProviderInfo,
    ProviderRegistryEntry,
    RerankResult,
    get_available_providers,
    get_provider_registry_entry,
    register_provider_class,
)


__all__ = [
    "PROVIDER_REGISTRY",
    "CohereModel",
    "CohereRerankModel",
    "EmbeddingProviderInfo",
    "ModelFamily",
    "NLPCapability",
    "NLPModelSize",
    "OpenAIModel",
    "ProviderCapabilities",
    "ProviderCapability",
    "ProviderKind",
    "ProviderRegistryEntry",
    "ProviderType",
    "RerankResult",
    "VoyageModel",
    "VoyageRerankModel",
    "get_available_providers",
    "get_provider_registry_entry",
    "register_provider_class",
]
