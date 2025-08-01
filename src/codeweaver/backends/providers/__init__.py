# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Backend providers for CodeWeaver."""
from codeweaver.backends.providers import QdrantBackend, QdrantHybridBackend
from codeweaver.backends.providers.docarray import (
    BaseDocArrayAdapter,
    DocArrayConfigFactory,
    DocArrayHybridAdapter,
    DocumentSchemaGenerator,
    SchemaConfig,
    SchemaTemplates,
)


__all__ = (
    "BaseDocArrayAdapter",
    "DocArrayConfigFactory",
    "DocArrayHybridAdapter",
    "DocumentSchemaGenerator",
    "QdrantBackend",
    "QdrantHybridBackend",
    "SchemaConfig",
    "SchemaTemplates",
)
