# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Content models for CodeWeaver using Pydantic.

Contains Pydantic models for code chunks and other content types,
providing validation, serialization, and type safety.
"""

import hashlib

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from codeweaver.types.backends.base import SearchResult
from codeweaver.types.factories.data_structures import ContentItem, ContentType


class CodeChunk(BaseModel):
    """
    Pydantic model representing a semantic chunk of code.
    """

    model_config = ConfigDict(
        extra="allow",  # Allow additional fields for extensibility
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    # Core content fields
    content: Annotated[str, Field(description="The actual code content")]
    file_path: Annotated[str, Field(description="Source file path")]
    start_line: Annotated[int, Field(ge=1, description="Starting line number")]
    end_line: Annotated[int, Field(ge=1, description="Ending line number")]
    chunk_type: Annotated[str, Field(description="Type of chunk (function, class, method, etc.)")]
    language: Annotated[str, Field(description="Programming language")]
    hash: Annotated[str, Field(description="Content hash for deduplication")]

    # Optional metadata fields
    node_kind: Annotated[str | None, Field(default=None, description="AST node kind from ast-grep")]
    size: Annotated[int | None, Field(default=None, ge=0, description="Content size in bytes")]

    # Additional metadata for extensibility
    metadata: Annotated[
        dict[str, Any], Field(default_factory=dict, description="Additional chunk metadata")
    ]

    @computed_field
    @property
    def content_size(self) -> int:
        """Get the size of the content in bytes."""
        return len(self.content.encode("utf-8"))

    @computed_field
    @property
    def line_count(self) -> int:
        """Get the number of lines in this chunk."""
        return self.end_line - self.start_line + 1

    @computed_field
    @property
    def unique_id(self) -> str:
        """Get a unique identifier for this chunk."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}:{self.hash}"

    def to_metadata(self) -> dict[str, Any]:
        """Convert to metadata format for vector database storage.

        Returns:
            Dictionary containing all chunk information for vector storage
        """
        base_metadata = {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_type": self.chunk_type,
            "language": self.language,
            "hash": self.hash,
            "node_kind": self.node_kind or "",
            "content": self.content,  # Store content for reranking
            "content_size": self.content_size,
            "line_count": self.line_count,
        }

        # Add any additional metadata
        base_metadata |= self.metadata

        return base_metadata

    @classmethod
    def create_with_hash(
        cls,
        content: str,
        file_path: str,
        start_line: int,
        end_line: int,
        chunk_type: str,
        language: str,
        node_kind: str | None = None,
        **kwargs: Any,
    ) -> "CodeChunk":
        """Create a CodeChunk with automatically generated hash.

        Args:
            content: The code content
            file_path: Source file path
            start_line: Starting line number
            end_line: Ending line number
            chunk_type: Type of chunk
            language: Programming language
            node_kind: Optional AST node kind
            **kwargs: Additional metadata

        Returns:
            New CodeChunk instance with generated hash
        """
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]  # noqa: S324 # just dedup

        return cls(
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type=chunk_type,
            language=language,
            hash=content_hash,
            node_kind=node_kind,
            **kwargs,
        )

    @classmethod
    def from_content_item(cls, content_item: ContentItem, content: str) -> "CodeChunk":
        """Create a CodeChunk from a ContentItem.

        Args:
            content_item: Source ContentItem
            content: The actual content text

        Returns:
            New CodeChunk instance
        """
        metadata = content_item.metadata or {}

        return cls.create_with_hash(
            content=content,
            file_path=metadata.get("file_path", content_item.path),
            start_line=metadata.get("start_line", 1),
            end_line=metadata.get("end_line", len(content.split("\n"))),
            chunk_type=metadata.get("chunk_type", "unknown"),
            language=content_item.language or "unknown",
            node_kind=metadata.get("node_kind"),
            metadata={
                k: v
                for k, v in metadata.items()
                if k not in {"file_path", "start_line", "end_line", "chunk_type", "node_kind"}
            },
        )

    def to_content_item(self) -> ContentItem:
        """Convert this CodeChunk to a ContentItem.

        Returns:
            ContentItem representation of this chunk
        """
        from codeweaver.types.factories.data_structures import ContentItem

        return ContentItem(
            path=self.unique_id,
            content_type=ContentType.CODE,
            language=self.language,
            size=self.content_size,
            metadata=self.to_metadata(),
        )

    def __str__(self) -> str:
        """String representation of the chunk."""
        return f"CodeChunk({self.file_path}:{self.start_line}-{self.end_line}, {self.chunk_type}, {len(self.content)} chars)"

    def __repr__(self) -> str:
        """Detailed representation of the chunk."""
        return (
            f"CodeChunk(file_path='{self.file_path}', "
            f"start_line={self.start_line}, end_line={self.end_line}, "
            f"chunk_type='{self.chunk_type}', language='{self.language}', "
            f"hash='{self.hash}', content_size={self.content_size})"
        )


class ContentSearchResult(BaseModel):
    """Pydantic model for search results.

    Represents a search result from semantic or structural search
    with relevance scoring and metadata.
    """

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    # Core result fields
    content: Annotated[str, Field(description="Matching content")]
    file_path: Annotated[str, Field(description="Source file path")]
    start_line: Annotated[int, Field(ge=1, description="Starting line number")]
    end_line: Annotated[int, Field(ge=1, description="Ending line number")]

    # Metadata fields
    chunk_type: Annotated[str, Field(description="Type of code chunk")]
    language: Annotated[str, Field(description="Programming language")]
    node_kind: Annotated[str, Field(default="", description="AST node kind")]

    # Scoring fields
    similarity_score: Annotated[float, Field(ge=0.0, le=1.0, description="Similarity score")]
    rerank_score: Annotated[
        float | None, Field(default=None, ge=0.0, le=1.0, description="Rerank score")
    ]

    # Additional metadata
    metadata: Annotated[
        dict[str, Any], Field(default_factory=dict, description="Additional search metadata")
    ]

    @computed_field
    @property
    def relevance_score(self) -> float:
        """Get the best available relevance score."""
        return self.rerank_score if self.rerank_score is not None else self.similarity_score

    @computed_field
    @property
    def line_count(self) -> int:
        """Get the number of lines in this result."""
        return self.end_line - self.start_line + 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        result = {
            "content": self.content,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_type": self.chunk_type,
            "language": self.language,
            "node_kind": self.node_kind,
            "similarity_score": self.similarity_score,
            "relevance_score": self.relevance_score,
            "line_count": self.line_count,
        }

        if self.rerank_score is not None:
            result["rerank_score"] = self.rerank_score

        # Add any additional metadata
        result |= self.metadata

        return result

    @classmethod
    def from_code_chunk(
        cls,
        chunk: CodeChunk,
        similarity_score: float,
        rerank_score: float | None = None,
        **kwargs: Any,
    ) -> SearchResult:
        """Create a `SearchResult` from a `CodeChunk`.

        Args:
            chunk: Source CodeChunk
            similarity_score: Similarity score
            rerank_score: Optional rerank score
            **kwargs: Additional metadata

        Returns:
            New `SearchResult` instance
        """
        return cls(
            content=chunk.content,
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            chunk_type=chunk.chunk_type,
            language=chunk.language,
            node_kind=chunk.node_kind or "",
            similarity_score=similarity_score,
            rerank_score=rerank_score,
            metadata=kwargs,
        )
