# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
DocArray backend integration for CodeWeaver.

Provides unified vector database interface supporting 10+ backends through DocArray,
maintaining full compatibility with CodeWeaver's VectorBackend and HybridSearchBackend protocols.
"""

from codeweaver.backends.providers.docarray.adapter import (
    BaseDocArrayAdapter,
    DocArrayHybridAdapter,
)
from codeweaver.backends.providers.docarray.config import (
    DocArrayBackendConfig,
    DocArrayConfigFactory,
    DocArraySchemaConfig,
)
from codeweaver.backends.providers.docarray.schema import (
    DocumentSchemaGenerator,
    SchemaConfig,
    SchemaTemplates,
)


# Re-export main components
__all__ = [
    "BaseDocArrayAdapter",
    "DocArrayBackendConfig",
    "DocArrayConfigFactory",
    "DocArrayHybridAdapter",
    "DocArraySchemaConfig",
    "DocumentSchemaGenerator",
    "SchemaConfig",
    "SchemaTemplates",
]
