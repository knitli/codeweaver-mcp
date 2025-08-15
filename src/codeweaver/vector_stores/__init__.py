# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Provider interfaces and implementations for CodeWeaver."""

from typing import TYPE_CHECKING

from codeweaver.vector_stores.base import EmbeddingProvider, SearchResult, VectorStoreProvider
from codeweaver.vector_stores.memory import InMemoryVectorStore
from codeweaver._settings import Provider


def get_store(provider: Provider) -> VectorStoreProvider:
    """Get the vector store provider."""
    if provider == Provider.IN_MEMORY:
        return InMemoryVectorStore()
    if provider == Provider.QDRANT:
        from codeweaver.vector_stores.qdrant import QdrantVectorStore

        return QdrantVectorStore()
    raise TypeError(f"Expected VectorStoreProvider, got {type(provider)}")


__all__ = ["EmbeddingProvider", "InMemoryVectorStore", "SearchResult", "VectorStoreProvider"]
