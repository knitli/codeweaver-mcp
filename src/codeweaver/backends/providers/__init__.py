# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Backend providers for CodeWeaver."""

from codeweaver.backends.providers.qdrant import QdrantBackend, QdrantHybridBackend


# Check if docarray is available first
try:
    from codeweaver.backends.providers.docarray.adapter import DOCARRAY_AVAILABLE
except ImportError:
    DOCARRAY_AVAILABLE = False

# Conditional imports for optional docarray backends
if DOCARRAY_AVAILABLE:
    try:
        from codeweaver.backends.providers.docarray import (
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

        __all__ = (
            "DOCARRAY_AVAILABLE",
            "BaseDocArrayAdapter",
            "DocArrayConfigFactory",
            "DocArrayHybridAdapter",
            "DocumentSchemaGenerator",
            "QdrantBackend",
            "QdrantDocArrayBackend",
            "QdrantHybridBackend",
            "SchemaConfig",
            "SchemaTemplates",
            "create_docarray_backend",
            "register_docarray_backends",
        )
    except ImportError:
        DOCARRAY_AVAILABLE = False
        __all__ = ("DOCARRAY_AVAILABLE", "QdrantBackend", "QdrantHybridBackend")
else:
    __all__ = ("DOCARRAY_AVAILABLE", "QdrantBackend", "QdrantHybridBackend")
