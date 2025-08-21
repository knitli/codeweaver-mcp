# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Abstract provider interfaces for embeddings and vector storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import UUID4, BaseModel, ConfigDict

from codeweaver._common import BaseEnum
from codeweaver._data_structures import CodeChunk, Metadata, SearchResult
from codeweaver._settings import Provider


if TYPE_CHECKING:
    from codeweaver.services._filter import Filter


# SPDX-SnippetBegin
# SPDX-FileCopyrightText: 2022-2025 Qdrant Solutions GmbH
# SPDX-License-Identifier: Apache-2.0

# From Qdrant's python client, [`qdrant-client`](https://github.com/qdrant/qdrant-client/tree/master/qdrant_client/http/models/models.py#L1803-L1820)


class PayloadSchemaType(str, BaseEnum):
    """
    The types of payload fields that can be indexed.
    """

    KEYWORD = "keyword"
    INTEGER = "integer"
    FLOAT = "float"
    GEO = "geo"
    TEXT = "text"
    BOOL = "bool"
    DATETIME = "datetime"
    UUID = "uuid"

    __slots__ = ()


class Entry(BaseModel):
    """
    A single entry in the Qdrant collection.
    """

    content: str
    metadata: Metadata | None = None


# SPDX-SnippetEnd


class VectorStoreProvider[VectorStoreClient, EmbeddingProvider, RerankingProvider](BaseModel, ABC):
    """Abstract interface for vector storage providers."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="allow")

    _client: VectorStoreClient
    _embedder: EmbeddingProvider
    _reranker: RerankingProvider | None = None

    @property
    def client(self) -> VectorStoreClient:
        """Returns the vector store client instance."""
        return self._client

    @property
    def embedder(self) -> EmbeddingProvider:
        """Returns the embedder instance."""
        return self._embedder

    @property
    def reranker(self) -> RerankingProvider | None:
        """Returns the reranker instance if available, otherwise None."""
        return self._reranker

    @property
    @abstractmethod
    def name(self) -> Provider:
        """
        The enum member representing the provider.
        """

    @property
    @abstractmethod
    def base_url(self) -> str | None:
        """
        The base URL for the provider's API, if applicable.
        """
        return None

    @property
    def collection(self) -> str | None:
        """Get the name of the currently configured collection."""
        return None

    @abstractmethod
    def list_collections(self) -> list[str] | None:
        """List all collections in the vector store.

        Returns:
            List of collection names
        """

    @abstractmethod
    async def search(
        self, vector: list[float], query_filter: Filter | None = None
    ) -> list[SearchResult]:
        """Search for similar vectors.

        Args:
            vector: Query vector
            query_filter: Filter to apply to the search

        Returns:
            List of search results
        """

    @abstractmethod
    async def upsert(self, chunks: list[CodeChunk]) -> None:
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

    @abstractmethod
    async def delete_by_id(self, ids: list[UUID4]) -> None:
        """
        Delete a specific code chunk by its unique identifier (the `chunk_id` field).
        """

    @abstractmethod
    async def delete_by_name(self, names: list[str]) -> None:
        """
        Delete specific code chunks by their unique names.
        """
