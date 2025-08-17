# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Provider interfaces and implementations for CodeWeaver."""

from typing import Any

from codeweaver._settings import Provider
from codeweaver.vector_stores.base import SearchResult, VectorStoreProvider
from codeweaver.vector_stores.memory import InMemoryVectorStoreProvider


def get_store(provider: Provider) -> type[VectorStoreProvider[Any]]:
    """Get the vector store provider."""
    if provider == Provider.FASTEMBED_VECTORSTORE:
        return InMemoryVectorStoreProvider
    if provider == Provider.QDRANT:
        from codeweaver.vector_stores.qdrant import QdrantVectorStore

        return QdrantVectorStore
    raise TypeError(f"Expected VectorStoreProvider, got {type(provider)}")


__all__ = ["InMemoryVectorStoreProvider", "SearchResult", "VectorStoreProvider"]
