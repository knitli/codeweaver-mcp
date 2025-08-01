# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Backend providers for CodeWeaver."""

from codeweaver.backends.providers.qdrant import QdrantBackend, QdrantHybridBackend

# Conditional imports for optional docarray backends
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
    DOCARRAY_AVAILABLE = True
    
    __all__ = (
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
    
    __all__ = (
        "QdrantBackend",
        "QdrantHybridBackend",
    )
