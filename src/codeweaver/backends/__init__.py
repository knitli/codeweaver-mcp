# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# sourcery skip: use-contextlib-suppress
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
from codeweaver.backends.base_config import BackendConfig
from codeweaver.backends.config import (
    EXAMPLE_CONFIGS,
    BackendConfigExtended,
    create_backend_config_from_env,
    get_provider_specific_config,
)
from codeweaver.backends.factory import BackendFactory
from codeweaver.backends.providers import DOCARRAY_AVAILABLE, QdrantBackend, QdrantHybridBackend
from codeweaver.cw_types import (
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


# Core exports that are always available
__all__ = [
    "DOCARRAY_AVAILABLE",
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

# Conditional docarray imports
if DOCARRAY_AVAILABLE:
    try:
        from codeweaver.backends.providers import (
            BaseDocArrayAdapter,
            DocArrayConfigFactory,
            DocArrayHybridAdapter,
            DocumentSchemaGenerator,
            QdrantDocArrayBackend,
            SchemaConfig,
            SchemaTemplates,
            create_docarray_backend,
            register_docarray_backends,
        )

        # Add docarray exports to __all__
        __all__ += [
            "BaseDocArrayAdapter",
            "DocArrayConfigFactory",
            "DocArrayHybridAdapter",
            "DocumentSchemaGenerator",
            "QdrantDocArrayBackend",
            "SchemaConfig",
            "SchemaTemplates",
            "create_docarray_backend",
            "register_docarray_backends",
        ]

        # Try to import DocArray backends factory (optional)
        try:  # noqa: SIM105
            from codeweaver.backends.providers.docarray import (
                factory as docarray_factory,  # noqa: F401
            )
        except ImportError:
            pass

    except ImportError:
        # DocArray imports failed despite DOCARRAY_AVAILABLE being True
        pass
