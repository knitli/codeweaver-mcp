# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Vector database backends for CodeWeaver.

This module provides comprehensive abstractions for 15+ vector databases
with support for hybrid search, streaming operations, and transactions.
"""

from codeweaver.backends.base import (
    BackendAuthError,
    BackendCollectionNotFoundError,
    BackendConnectionError,
    BackendError,
    BackendResourceProvider,
    BackendUnsupportedOperationError,
    BackendVectorDimensionMismatchError,
    CollectionInfo,
    DistanceMetric,
    FilterCondition,
    HybridSearchBackend,
    HybridStrategy,
    SearchFilter,
    SearchResult,
    StreamingBackend,
    TransactionalBackend,
    VectorBackend,
    VectorPoint,
)
from codeweaver.backends.config import (
    EXAMPLE_CONFIGS,
    BackendConfig,
    BackendConfigExtended,
    create_backend_config_from_env,
    create_backend_config_from_legacy,
    get_provider_specific_config,
    migrate_config_to_toml,
)
from codeweaver.backends.factory import BackendFactory, create_backend
from codeweaver.backends.qdrant import QdrantBackend, QdrantHybridBackend


__all__ = [
    "EXAMPLE_CONFIGS",
    "BackendAuthError",
    "BackendCollectionNotFoundError",
    # Configuration
    "BackendConfig",
    "BackendConfigExtended",
    "BackendConnectionError",
    # Exceptions
    "BackendError",
    # Factory
    "BackendFactory",
    # Enums
    "BackendResourceProvider",
    "BackendUnsupportedOperationError",
    "BackendVectorDimensionMismatchError",
    "CollectionInfo",
    "DistanceMetric",
    "FilterCondition",
    "HybridSearchBackend",
    "HybridStrategy",
    # Implementations
    "QdrantBackend",
    "QdrantHybridBackend",
    "SearchFilter",
    "SearchResult",
    "StreamingBackend",
    "TransactionalBackend",
    # Core protocols
    "VectorBackend",
    # Data structures
    "VectorPoint",
    "create_backend",
    "create_backend_config_from_env",
    "create_backend_config_from_legacy",
    "get_provider_specific_config",
    "migrate_config_to_toml",
]
