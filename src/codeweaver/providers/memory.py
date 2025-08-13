# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""In-memory provider implementations for Phase 1 development."""

from __future__ import annotations

import math

from pathlib import Path
from typing import Any, TypeGuard

from codeweaver.providers.base import (
    CodeChunk,
    EmbeddingProvider,
    SearchResult,
    VectorStoreProvider,
)


def is_similarity_tuple(item: tuple[float, int] | None) -> TypeGuard[tuple[float, int]]:
    """Check if the item is a similarity tuple."""
    return item is not None and len(item) == 2 and isinstance(item[0], float)


class InMemoryVectorStore(VectorStoreProvider):
    """Simple in-memory vector store for Phase 1 development.

    This is a basic implementation for Phase 1 that stores vectors
    in memory with simple cosine similarity search.
    """

    def __init__(self) -> None:
        """Initialize in-memory vector store."""
        self.chunks: list[CodeChunk] = []
        self.vectors: list[list[float]] = []
        self.metadata: list[dict[str, Any]] = []

    async def search(self, vector: list[float], limit: int = 10) -> list[SearchResult]:
        """Search for similar vectors using cosine similarity.

        Args:
            vector: Query vector
            limit: Maximum number of results

        Returns:
            List of search results sorted by similarity
        """
        if not self.vectors:
            return []

        # Calculate cosine similarity with all stored vectors
        similarities: list[tuple[float, int]] = []
        for i, stored_vector in enumerate(self.vectors):
            if similarity := self._cosine_similarity(vector, stored_vector):
                similarities.append((similarity, i))

        filtered_similarities = sorted(
            filter(is_similarity_tuple, similarities), key=lambda x: x[0], reverse=True
        )
        similarities = filtered_similarities[:limit]

        # Build search results
        results: list[SearchResult] = []
        for score, index in similarities:
            chunk = self.chunks[index]
            result = SearchResult(
                file_path=chunk.file_path,
                content=chunk.content,
                score=score,
                metadata={
                    "line_range": chunk.line_range,
                    "language": chunk.language,
                    "chunk_type": chunk.chunk_type,
                    **chunk.metadata,
                },
            )
            results.append(result)

        return results

    async def upsert_chunks(self, chunks: list[CodeChunk]) -> None:
        """Store code chunks in memory.

        Note: For Phase 1, this doesn't generate embeddings.
        That will be added in Phase 2 with actual embedding providers.

        Args:
            chunks: List of code chunks to store
        """
        for chunk in chunks:
            # For Phase 1, create dummy vectors based on content hash
            # This allows the system to work without actual embeddings
            dummy_vector = self._create_dummy_vector(chunk.content)

            self.chunks.append(chunk)
            self.vectors.append(dummy_vector)
            self.metadata.append({
                "file_path": str(chunk.file_path),
                "line_range": chunk.line_range,
                "language": chunk.language,
                "chunk_type": chunk.chunk_type,
            })

    async def delete_by_file(self, file_path: Path) -> None:
        """Delete all chunks for a specific file.

        Args:
            file_path: Path of file to remove from index
        """
        # Find indices of chunks to remove
        indices_to_remove: list[int] = []
        indices_to_remove.extend(
            i for i, chunk in enumerate(self.chunks) if chunk.file_path.samefile(file_path)
        )
        # Remove in reverse order to maintain correct indices
        for i in reversed(indices_to_remove):
            del self.chunks[i]
            del self.vectors[i]
            del self.metadata[i]

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score between 0 and 1
        """
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _create_dummy_vector(self, content: str, dimension: int = 128) -> list[float]:
        """Create a dummy vector based on content hash.

        This is a placeholder for Phase 1 to enable basic functionality
        without actual embedding generation.

        Args:
            content: Text content to create vector for
            dimension: Vector dimension

        Returns:
            Dummy embedding vector
        """
        # Simple hash-based vector generation for Phase 1
        content_hash = hash(content)
        vector: list[float] = [
            abs(content_hash + i * 31) % 1000 / 1000.0 - 0.5 for i in range(dimension)
        ]
        # Normalize vector
        magnitude = math.sqrt(sum(x * x for x in vector))
        if magnitude > 0:
            vector = [x / magnitude for x in vector]

        return vector


class DummyEmbeddingProvider(EmbeddingProvider):
    """Dummy embedding provider for Phase 1 development.

    This creates simple hash-based embeddings to enable basic
    functionality without external API dependencies.
    """

    def __init__(self, dimension: int = 128) -> None:
        """Initialize dummy embedding provider.

        Args:
            dimension: Embedding vector dimension
        """
        super().__init__()
        self.dimension = dimension

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate dummy embeddings for texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of dummy embedding vectors
        """
        return [self._create_dummy_embedding(text) for text in texts]

    async def embed_query(self, query: str) -> list[float]:
        """Generate dummy embedding for query.

        Args:
            query: Query text

        Returns:
            Dummy embedding vector
        """
        return self._create_dummy_embedding(query)

    def _create_dummy_embedding(self, text: str) -> list[float]:
        """Create dummy embedding based on text hash.

        Args:
            text: Input text

        Returns:
            Normalized dummy embedding vector
        """
        text_hash = hash(text.lower())
        vector = [abs(text_hash + i * 17) % 1000 / 1000.0 - 0.5 for i in range(self.dimension)]
        # Normalize
        magnitude = math.sqrt(sum(x * x for x in vector))
        if magnitude > 0:
            vector = [x / magnitude for x in vector]

        return vector
