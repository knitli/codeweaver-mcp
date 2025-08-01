# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Backend providers for CodeWeaver."""

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
from codeweaver.backends.providers.qdrant import QdrantBackend, QdrantHybridBackend


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
