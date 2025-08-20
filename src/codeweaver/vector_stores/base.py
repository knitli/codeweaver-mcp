# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Abstract provider interfaces for embeddings and vector storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, LiteralString, NotRequired, Required, TypedDict
from uuid import uuid4

from pydantic import UUID4, BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveFloat

from codeweaver._common import BaseEnum
from codeweaver._settings import Provider


if TYPE_CHECKING:
    from ast_grep_py import SgNode

    from codeweaver._data_structures import Span
    from codeweaver.language import SemanticSearchLanguage
    from codeweaver.services._filter import Filter

type NonClient = None


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


class SearchResult(BaseModel):
    """Result from vector search operations."""

    file_path: Path
    content: str
    score: Annotated[NonNegativeFloat, Field(description="Similarity score")]
    metadata: Annotated[
        Metadata | None, Field(description="Additional metadata about the result")
    ] = None


class SemanticMetadata(TypedDict, total=False):
    """Metadata associated with the semantics of a code chunk."""

    language: SemanticSearchLanguage | LiteralString | None
    primary_node: SgNode | None
    nodes: tuple[SgNode, ...] | None


class Metadata(TypedDict, total=False):
    """Metadata associated with a code chunk."""

    chunk_id: Required[Annotated[UUID4, Field(description="Unique identifier for the code chunk")]]
    created_at: Required[
        Annotated[PositiveFloat, Field(description="Timestamp when the chunk was created")]
    ]
    name: NotRequired[
        Annotated[str | None, Field(description="Name of the code chunk, if applicable")]
    ]
    updated_at: NotRequired[
        Annotated[
            PositiveFloat | None,
            Field(description="Timestamp when the chunk was last updated or checked for accuracy."),
        ]
    ]
    tags: NotRequired[
        Annotated[
            tuple[str] | None,
            Field(description="Tags associated with the code chunk, if applicable"),
        ]
    ]
    semantic_meta: NotRequired[
        Annotated[
            SemanticMetadata | None,
            Field(
                description="Semantic metadata associated with the code chunk, if applicable. Should be included if the code chunk was from semantic chunking."
            ),
        ]
    ]


class CodeChunk(BaseModel):
    """Represents a chunk of code with metadata."""

    content: str
    line_range: Annotated[Span, Field(description="Line range in the source file")]
    file_path: Annotated[
        Path | None,
        Field(
            description="Path to the source file. Not all chunks are from files, so this can be None."
        ),
    ] = None
    language: SemanticSearchLanguage | LiteralString | None = None
    chunk_type: str = "text_block"  # For Phase 1, simple text blocks
    timestamp: Annotated[
        PositiveFloat,
        Field(
            default_factory=datetime.now(UTC).timestamp,
            kw_only=True,
            description="Timestamp of the code chunk creation or modification",
        ),
    ] = datetime.now(UTC).timestamp()
    chunk_id: Annotated[
        UUID4,
        Field(
            default_factory=uuid4, kw_only=True, description="Unique identifier for the code chunk"
        ),
    ] = uuid4()
    metadata: Annotated[
        Metadata | None,
        Field(
            default_factory=dict,
            kw_only=True,
            description="Additional metadata about the code chunk",
        ),
    ] = None


class VectorStoreProvider[VectorStoreClient, Embedder, Reranker](BaseModel, ABC):
    """Abstract interface for vector storage providers."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="allow")

    _client: VectorStoreClient
    _embedder: Embedder
    _reranker: Reranker | None = None

    @property
    def client(self) -> VectorStoreClient:
        """Returns the vector store client instance."""
        return self._client

    @property
    def embedder(self) -> Embedder:
        """Returns the embedder instance."""
        return self._embedder

    @property
    def reranker(self) -> Reranker | None:
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
