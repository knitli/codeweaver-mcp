# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Abstract provider interfaces for embeddings and vector storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, model_validator


@dataclass(frozen=True, order=True, slots=True)
class Span(BaseModel):
    """Represents a span of text in a document."""

    start: NonNegativeInt
    end: NonNegativeInt

    @model_validator(mode="after")
    def check_span(self) -> Span:
        """Ensure that the start is less than or equal to the end."""
        if self.start > self.end:
            raise ValueError("Start must be less than or equal to end")
        return self


class SearchResult(BaseModel):
    """Result from vector search operations."""

    file_path: Path
    content: str
    score: Annotated[NonNegativeFloat, Field(description="Similarity score")]
    metadata: Annotated[dict[str, Any], Field(description="Additional metadata about the result")]


class CodeChunk(BaseModel):
    """Represents a chunk of code with metadata."""

    file_path: Path
    content: str
    line_range: Annotated[Span, Field(description="Line range in the source file")]
    language: str | None = None
    chunk_type: str = "text_block"  # For Phase 1, simple text blocks
    metadata: Annotated[
        dict[str, Any],
        Field(default_factory=dict, description="Additional metadata about the code chunk"),
    ]


class EmbeddingProvider(BaseModel, ABC):
    """Abstract interface for embedding providers."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query.

        Args:
            query: Query string to embed

        Returns:
            Embedding vector for the query
        """


class VectorStoreProvider(BaseModel, ABC):
    """Abstract interface for vector storage providers."""

    @abstractmethod
    async def search(self, vector: list[float], limit: int = 10) -> list[SearchResult]:
        """Search for similar vectors.

        Args:
            vector: Query vector
            limit: Maximum number of results

        Returns:
            List of search results
        """

    @abstractmethod
    async def upsert_chunks(self, chunks: list[CodeChunk]) -> None:
        """Insert or update code chunks in the vector store.

        Args:
            chunks: List of code chunks to store
        """

    @abstractmethod
    async def delete_by_file(self, file_path: Path) -> None:
        """Delete all chunks for a specific file.

        Args:
            file_path: Path of file to remove from index
        """
