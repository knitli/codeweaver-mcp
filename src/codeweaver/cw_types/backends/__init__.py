# SPDX-FileCopyrightText: 2025 Knitli Inc.
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Common types for vector database backends."""

from codeweaver.cw_types.backends.base import (
    CollectionInfo,
    FilterCondition,
    SearchFilter,
    SearchResult,
    VectorPoint,
)
from codeweaver.cw_types.backends.capabilities import (
    BackendCapabilities,
    get_all_backend_capabilities,
)
from codeweaver.cw_types.backends.enums import (
    BackendProvider,
    DistanceMetric,
    FilterOperator,
    HybridFusionStrategy,
    HybridStrategy,
    IndexType,
    SparseIndexType,
    StorageType,
)
from codeweaver.cw_types.backends.providers import PROVIDER_REGISTRY as BACKEND_PROVIDER_REGISTRY
from codeweaver.cw_types.backends.providers import (
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
    "get_all_backend_capabilities",
]
