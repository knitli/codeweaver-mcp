# SPDX-FileCopyrightText: 2025 Knitli Inc.
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Common types for vector database backends."""

from codeweaver.types.backends.base import (
    CollectionInfo,
    FilterCondition,
    SearchFilter,
    SearchResult,
    VectorPoint,
)
from codeweaver.types.backends.capabilities import BackendCapabilities
from codeweaver.types.backends.enums import (
    BackendProvider,
    DistanceMetric,
    FilterOperator,
    HybridFusionStrategy,
    HybridStrategy,
    IndexType,
    SparseIndexType,
    StorageType,
)
from codeweaver.types.backends.providers import PROVIDER_REGISTRY as BACKEND_PROVIDER_REGISTRY
from codeweaver.types.backends.providers import (
    EmbeddingProviderBase,
    ProviderInfo,
    RerankProviderBase,
)


__all__ = [
    "BACKEND_PROVIDER_REGISTRY",
    "BackendCapabilities",
    "BackendProvider",
    "CollectionInfo",
    "DistanceMetric",
    "EmbeddingProviderBase",
    "FilterCondition",
    "FilterOperator",
    "HybridFusionStrategy",
    "HybridStrategy",
    "IndexType",
    "ProviderInfo",
    "RerankProviderBase",
    "SearchFilter",
    "SearchResult",
    "SparseIndexType",
    "StorageType",
    "VectorPoint",
]
