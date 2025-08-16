# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""In-memory provider implementations for Phase 1 development."""
# TODO: Transition to fastembed-vectorstore
# It has a simple api --`load`, `embed_documents`, `search`, `save`  -- load and save are for persistence with json and take a string path.
# The constructor takes a provider, which is an Enum, `FastembedEmbeddingModel`.
# Instantiate with `FastembedVectorStore(FastembedEmbeddingModel.model)` -- for dev, use something lightweight like `FastembedEmbeddingModel.BGESmallENV15Q` (bge small quantized)
# uses cosine similarity for search.
# Note that it is both an embedding provider and a vector store provider, so it can be used for both embedding and searching.

from __future__ import annotations

import contextlib

from pathlib import Path

from codeweaver._settings import EmbeddingModelSettings
from codeweaver.vector_stores.base import CodeChunk, SearchResult, VectorStoreProvider


try:
    from fastembed_vectorstore import FastembedEmbeddingModel, FastembedVectorStore
except ImportError:
    # noop stubs
    type FastembedVectorStore = None
    type FastembedEmbeddingModel = None


class FastembedVectorstoreProvider(VectorStoreProvider[FastembedVectorStore]):
    """In-memory vector store for code chunks."""

    _client: FastembedVectorStore | None = None
    _model_settings: EmbeddingModelSettings | None = None
    _store: FastembedVectorStore | None = None

    def __init__(
        self, embedding_model_settings: EmbeddingModelSettings, path: Path | None = None
    ) -> None:
        """Initialize the in-memory vector store."""
        self._client = FastembedVectorStore(embedding_model_settings.model)
        self._model_settings = embedding_model_settings
        self.path = path
        if self.path and self.path.exists():
            with contextlib.suppress(OSError):
                self._store = FastembedVectorStore.load(self.path)

    async def search(self, vector: list[float]) -> list[SearchResult] | None:
        """Search for similar vectors.

        Args:
            vector: Query vector

        Returns:
            List of search results
        """
        return self._client.search(vector, limit=self._model_settings.search_limit)

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
