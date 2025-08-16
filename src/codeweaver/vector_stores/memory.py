# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""In-memory provider implementations."""
# uses `fastembed_vectorstore` for in-memory vector storage.
# fastembed_vectorstore is a combined implementation of an in-memory vector store using `fastembed` for embeddings.
# It can persist vectors with json serialization and supports searching for similar vectors.
# TODO: We should consider a few things to improve this implementation:
#  - should we use a context manager to ensure the state is saved before exiting? (if it has a path in settings)
#  - should we implement a way to delete vectors by file path?
#  - Once we have a provider registry, we can register this provider with the registry.
#  - The 'search' method takes a string query because `fastembed_vectorstore` expects a string so that it can embed it.
#    - that's NOT how we planned the overall API for vector stores. One approach might be to use dependency injection to give vector stores an embedding provider, and then the store becomes the primary interface for embedding.

from __future__ import annotations

import contextlib

from pathlib import Path

from codeweaver._settings import EmbeddingModelSettings
from codeweaver.vector_stores.base import CodeChunk, SearchResult, VectorStoreProvider


try:
    import fastembed_vectorstore

    from fastembed_vectorstore import FastembedEmbeddingModel, FastembedVectorstore
except ImportError:
    # noop stubs
    if not hasattr(__builtins__, "FastembedVectorstore"):
        type FastembedVectorstore = None
    else:
        type FastembedVectorstore = fastembed_vectorstore.FastembedVectorstore
    if not hasattr(__builtins__, "FastembedEmbeddingModel"):
        type FastembedEmbeddingModel = None
    else:
        type FastembedEmbeddingModel = fastembed_vectorstore.FastembedEmbeddingModel


class FastembedVectorstoreProvider(VectorStoreProvider[FastembedVectorstore]):
    """In-memory vector store for code chunks."""

    _client: FastembedVectorstore | None = None
    _model_settings: EmbeddingModelSettings | None = None
    _store: FastembedVectorstore | None = None

    def __init__(
        self, embedding_model_settings: EmbeddingModelSettings, path: Path | None = None
    ) -> None:
        """Initialize the in-memory vector store."""
        if not isinstance(FastembedVectorstore, type) or not isinstance(
            FastembedEmbeddingModel, type
        ):  # type: ignore
            raise TypeError("fastembed_vectorstore is not installed or not available.")
        self._client = FastembedVectorstore(embedding_model_settings.model)
        self._model_settings = embedding_model_settings
        self.path = path
        if self.path and self.path.exists():
            with contextlib.suppress(OSError):
                self._store = FastembedVectorstore.load(self.path)

    async def search(self, query: list[str]) -> list[SearchResult] | None:
        """Search for similar vectors.

        Args:
            query: Query text

        Returns:
            List of search results
        """
        return self._client.search(query, n=self._model_settings.search_limit)

    async def upsert_chunks(self, chunks: list[CodeChunk]) -> None:
        """Insert or update code chunks in the vector store.

        Args:
            chunks: List of code chunks to store
        """
        serialized_chunks = [chunk.model_dump_json() for chunk in chunks]
        self._client.embed_documents(serialized_chunks)

    async def delete_by_file(self, file_path: Path) -> None:
        """Delete all chunks for a specific file.

        Args:
            file_path: Path of file to remove from index
        """
        # Implement delete logic in memory

    async def save(self) -> None:
        """Save the vector store to disk."""
        if self.path:
            with contextlib.suppress(OSError):
                self._store.save(self.path)
