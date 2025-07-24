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
    AuthenticationError,
    BackendError,
    CollectionInfo,
    CollectionNotFoundError,
    ConnectionError,
    DimensionMismatchError,
    DistanceMetric,
    FilterCondition,
    HybridSearchBackend,
    HybridStrategy,
    SearchFilter,
    SearchResult,
    StreamingBackend,
    TransactionalBackend,
    UnsupportedOperationError,
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
    "AuthenticationError",
    # Configuration
    "BackendConfig",
    "BackendConfigExtended",
    # Exceptions
    "BackendError",
    # Factory
    "BackendFactory",
    "CollectionInfo",
    "CollectionNotFoundError",
    "ConnectionError",
    "DimensionMismatchError",
    # Enums
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
    "UnsupportedOperationError",
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
