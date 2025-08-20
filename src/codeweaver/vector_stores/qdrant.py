# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Qdrant provider for vector and hybrid search/store."""

from codeweaver.vector_stores.base import VectorStoreProvider


QdrantClient = None
try:
    from qdrant_client import AsyncQdrantClient

    QdrantClient = AsyncQdrantClient
except ImportError:
    QdrantClient = None


class QdrantVectorStore(VectorStoreProvider[QdrantClient]):
    """Qdrant vector store provider."""
