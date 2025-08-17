# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""In-memory vector store implementation using FastEmbed."""

from __future__ import annotations

import contextlib
import json

from pathlib import Path
from typing import Any

from pydantic import UUID4

from codeweaver._settings import Provider
from codeweaver.exceptions import ConfigurationError
from codeweaver.services._filter import Filter
from codeweaver.vector_stores.base import CodeChunk, SearchResult, VectorStoreProvider


try:
    from fastembed_vectorstore import FastembedEmbeddingModel, FastembedVectorstore

except ImportError as e:
    raise ImportError(
        "fastembed_vectorstore is required for FastembedVectorstoreProvider. Install it with: pip install fastembed-vectorstore"
    ) from e

type FastembedResult = tuple[str, float]
type FastembedResults = list[FastembedResult]
"""Fastembed search result type. A list of tuples, where each inner tuple contains the document text and its similarity score."""


class FastembedVectorstoreProvider(VectorStoreProvider[FastembedVectorstore | None]):
    """In-memory vector store using FastEmbed for embeddings and storage."""

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        path: Path | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the in-memory vector store.

        Args:
            embedding_model: Name of the embedding model to use
            path: Optional path for persistence
            **kwargs: Additional configuration arguments
        """
        self._embedding_model = embedding_model
        self._path = path
        self._client: FastembedVectorstore | None = None

        # Initialize the vector store
        self._initialize_store()
        if not self._client:
            raise ConfigurationError(
                "Failed to initialize FastembedVectorstore. Ensure the embedding model is valid and fastembed_vectorstore is installed."
            )

    def _initialize_store(self) -> None:
        """Initialize the FastEmbed vector store."""
        try:
            # Try to load existing store if path is provided
            if (
                (
                    model := next(
                        (
                            model
                            for model in FastembedEmbeddingModel.__members__.values()
                            if model.name.lower() == self._embedding_model.lower()
                        ),
                        None,
                    )
                )
                and self._client
                and self._path
                and self._path.exists()
            ):
                with contextlib.suppress(Exception):
                    client = FastembedVectorstore(model)
                    self._client = client.load(model, str(self._path))
                    return

            if model:
                self._client = FastembedVectorstore(model)

        except Exception as e:
            # Fallback to None if initialization fails
            self._client = None
            raise RuntimeError(f"Failed to initialize FastembedVectorstore: {e}") from e

    @property
    def name(self) -> Provider:
        """Get the provider name."""
        return Provider.FASTEMBED_VECTORSTORE

    @property
    def base_url(self) -> str | None:
        """Get the base URL (not applicable for in-memory store)."""
        return None

    @property
    def collection(self) -> str | None:
        """Get the collection name (not applicable for FastEmbed)."""
        return None

    def list_collections(self) -> list[str] | None:
        """List collections (not applicable for FastEmbed)."""
        return None

    async def search(
        self, vector: list[float], query_filter: Filter | None = None
    ) -> list[SearchResult]:
        """Search for similar vectors.

        Args:
            vector: Query vector to search with
            query_filter: Optional filter to apply (not supported by FastEmbed)

        Returns:
            List of search results
        """
        if not self._client:
            return []

        try:
            # FastembedVectorstore doesn't support vector search directly,
            # it expects string queries. For now, we'll return empty results
            # and implement proper vector search in a future enhancement.
            return []
        except Exception:
            return []

    async def search_by_text(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search using text query (FastEmbed's native interface).

        Args:
            query: Text query to search with
            limit: Maximum number of results to return

        Returns:
            List of search results
        """
        if not self._client:
            return []

        try:
            results: FastembedResults = self._client.search(query, n=limit)

            search_results: list[SearchResult] = []
            for result in results:
                # Convert FastEmbed result to our SearchResult format
                try:
                    # Parse the stored JSON content back to CodeChunk
                    chunk_data = (
                        json.loads(result.content)
                        if isinstance(result.content, str)
                        else result.content
                    )

                    search_result = SearchResult(
                        file_path=Path(chunk_data.get("file_path", "")),
                        content=chunk_data.get("content", ""),
                        score=float(result.score) if hasattr(result, "score") else 0.0,
                        metadata=chunk_data.get("metadata"),
                    )
                    search_results.append(search_result)
                except (json.JSONDecodeError, KeyError, TypeError):
                    # Skip malformed results
                    continue

        except Exception:
            return []
        else:
            return search_results

    async def upsert(self, chunks: list[CodeChunk]) -> None:
        """Insert or update code chunks in the vector store.

        Args:
            chunks: List of code chunks to store
        """
        if not self._client or not chunks:
            return

        try:
            # Serialize chunks for storage
            documents = []
            for chunk in chunks:
                # Create a document that combines content with searchable metadata
                document_content = (
                    f"{chunk.content}\n\nFile: {chunk.file_path}\nLanguage: {chunk.language}"
                )
                documents.append(document_content)

            # Embed and store documents
            self._client.embed_documents(documents)

            # Save to disk if path is configured
            await self._save_if_configured()

        except Exception as e:
            raise RuntimeError(f"Failed to upsert chunks: {e}") from e

    async def delete_by_file(self, file_path: Path) -> None:
        """Delete all chunks for a specific file.

        Args:
            file_path: Path of file to remove from index
        """
        # FastembedVectorstore doesn't support selective deletion
        # This would require rebuilding the entire index
        # For now, we'll implement this as a no-op

    async def delete_by_id(self, ids: list[UUID4]) -> None:
        """Delete specific code chunks by their unique identifiers.

        Args:
            ids: List of chunk IDs to delete
        """
        # FastembedVectorstore doesn't support selective deletion by ID
        # This would require rebuilding the entire index
        # For now, we'll implement this as a no-op

    async def delete_by_name(self, names: list[str]) -> None:
        """Delete specific code chunks by their unique names.

        Args:
            names: List of chunk names to delete
        """
        # FastembedVectorstore doesn't support selective deletion by name
        # This would require rebuilding the entire index
        # For now, we'll implement this as a no-op

    async def save(self) -> None:
        """Save the vector store to disk."""
        await self._save_if_configured()

    async def _save_if_configured(self) -> None:
        # sourcery skip: use-contextlib-suppress
        """Save the store to disk if a path is configured."""
        if self._client and self._path:
            try:
                # Ensure parent directory exists
                self._path.parent.mkdir(parents=True, exist_ok=True)

                # Save the vector store
                self._client.save(str(self._path))

            except Exception:
                # Log the error but don't fail the operation
                # TODO: Use proper logging when available
                pass

    def clear(self) -> None:
        """Clear all vectors from the store."""
        if self._client:
            # Reinitialize the store to clear it
            self._initialize_store()


class InMemoryVectorStoreProvider(VectorStoreProvider[dict[str, Any]]):
    """Simple in-memory vector store implementation using Python dictionaries.

    This is a fallback implementation when fastembed_vectorstore is not available.
    It provides basic storage but no semantic search capabilities.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the simple in-memory vector store."""
        self._client: dict[str, Any] = {"chunks": [], "metadata": {}}

    @property
    def name(self) -> Provider:
        """Get the provider name."""
        return Provider.FASTEMBED_VECTORSTORE  # Use the same enum value

    @property
    def base_url(self) -> str | None:
        """Get the base URL (not applicable)."""
        return None

    @property
    def collection(self) -> str | None:
        """Get the collection name."""
        return "default"

    def list_collections(self) -> list[str] | None:
        """List collections."""
        return ["default"]

    async def search(
        self, vector: list[float], query_filter: Filter | None = None
    ) -> list[SearchResult]:
        """Search for similar vectors (basic keyword matching).

        Args:
            vector: Query vector (ignored in this implementation)
            query_filter: Optional filter to apply

        Returns:
            List of search results
        """
        # Simple implementation that returns all chunks
        # Real semantic search would require vector similarity
        results = []
        for chunk_data in self._client["chunks"]:
            try:
                result = SearchResult(
                    file_path=Path(chunk_data["file_path"])
                    if chunk_data.get("file_path")
                    else Path(),
                    content=chunk_data.get("content", ""),
                    score=1.0,  # Default score
                    metadata=chunk_data.get("metadata"),
                )
                results.append(result)
            except (KeyError, TypeError):
                continue

        return results

    async def upsert(self, chunks: list[CodeChunk]) -> None:
        """Insert or update code chunks.

        Args:
            chunks: List of code chunks to store
        """
        for chunk in chunks:
            chunk_data = {
                "chunk_id": str(chunk.chunk_id),
                "content": chunk.content,
                "file_path": str(chunk.file_path) if chunk.file_path else None,
                "language": chunk.language,
                "chunk_type": chunk.chunk_type,
                "timestamp": chunk.timestamp,
                "metadata": chunk.metadata,
                "line_range": {
                    "start": chunk.line_range.start if hasattr(chunk.line_range, "start") else 0,
                    "end": chunk.line_range.end if hasattr(chunk.line_range, "end") else 0,
                },
            }

            # Simple upsert: remove existing and add new
            self._client["chunks"] = [
                c for c in self._client["chunks"] if c.get("chunk_id") != chunk_data["chunk_id"]
            ]
            self._client["chunks"].append(chunk_data)

    async def delete_by_file(self, file_path: Path) -> None:
        """Delete all chunks for a specific file.

        Args:
            file_path: Path of file to remove from index
        """
        file_path_str = str(file_path)
        self._client["chunks"] = [
            chunk for chunk in self._client["chunks"] if chunk.get("file_path") != file_path_str
        ]

    async def delete_by_id(self, ids: list[UUID4]) -> None:
        """Delete specific code chunks by their unique identifiers.

        Args:
            ids: List of chunk IDs to delete
        """
        ids_str = {str(id_) for id_ in ids}
        self._client["chunks"] = [
            chunk for chunk in self._client["chunks"] if chunk.get("chunk_id") not in ids_str
        ]

    async def delete_by_name(self, names: list[str]) -> None:
        """Delete specific code chunks by their unique names.

        Args:
            names: List of chunk names to delete
        """
        names_set = set(names)
        self._client["chunks"] = [
            chunk
            for chunk in self._client["chunks"]
            if chunk.get("metadata", {}).get("name") not in names_set
        ]
