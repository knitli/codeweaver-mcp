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
    HybridSearchBackend,
    StreamingBackend,
    TransactionalBackend,
    VectorBackend,
)
from codeweaver.backends.config import (
    EXAMPLE_CONFIGS,
    BackendConfigExtended,
    create_backend_config_from_env,
    get_provider_specific_config,
)
from codeweaver.backends.factory import BackendConfig, BackendFactory
from codeweaver.backends.providers import QdrantBackend, QdrantHybridBackend
from codeweaver.types import (
    BackendAuthError,
    BackendCollectionNotFoundError,
    BackendConnectionError,
    BackendError,
    BackendUnsupportedOperationError,
    BackendVectorDimensionMismatchError,
    CollectionInfo,
    DistanceMetric,
    FilterCondition,
    SearchResult,
    VectorPoint,
)


# Try to import DocArray backends (optional)
try:
    from codeweaver.backends.providers.docarray import factory as docarray_factory  # noqa: F401

    _DOCARRAY_AVAILABLE = True
except ImportError:
    _DOCARRAY_AVAILABLE = False


__all__ = [
    "EXAMPLE_CONFIGS",
    "BackendAuthError",
    "BackendCollectionNotFoundError",
    "BackendConfig",
    "BackendConfigExtended",
    "BackendConnectionError",
    "BackendError",
    "BackendFactory",
    "BackendUnsupportedOperationError",
    "BackendVectorDimensionMismatchError",
    "CollectionInfo",
    "DistanceMetric",
    "FilterCondition",
    "HybridSearchBackend",
    "QdrantBackend",
    "QdrantHybridBackend",
    "SearchResult",
    "StreamingBackend",
    "TransactionalBackend",
    "VectorBackend",
    "VectorPoint",
    "create_backend_config_from_env",
    "get_provider_specific_config",
]
