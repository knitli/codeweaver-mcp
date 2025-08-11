# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
FastMCP middleware for code chunking services.

Provides intelligent code chunking using ast-grep's tree-sitter parsers
with fallback parsing, integrated as FastMCP middleware for service injection.
"""

import hashlib
import logging

from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, ClassVar, Literal

import mcp.types as mt

from ast_grep_py import Config, SgNode, SgRoot
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.middleware.middleware import CallNext
from pydantic import BaseModel, ConfigDict, Field, computed_field

from codeweaver.language import SemanticSearchLanguage


logger = logging.getLogger(__name__)


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


class ChunkingMiddleware(Middleware):
    """FastMCP middleware providing intelligent code chunking services."""

    # Languages supported by ast-grep
    SUPPORTED_LANGUAGES: ClassVar[MappingProxyType[str, SemanticSearchLanguage]] = (
        SemanticSearchLanguage.ext_map()
    )

    # AST-grep patterns for semantic chunking by language
    CHUNK_PATTERNS: ClassVar[dict[str, tuple[tuple[str, str], ...]]] = {
        # ! IMPORTANT! TODO: We need to add patterns for all languages we support
        "general": (("comment", "comment"),),
        "javascript": (
            ("comment", "comment"),
            ("function_declaration", "function"),
            ("arguments", "argument"),
            ("formal_parameters", "parameter"),
            ("class_declaration", "class"),
            ("arrow_function", "arrow_function"),
            ("method_definition", "method"),
            ("interface_declaration", "interface"),
            ("type_alias", "type_alias"),
            ("type_annotation", "type_annotation"),
            ("type_parameter", "type_parameter"),
            ("return_type", "return_type"),
            ("variable_declaration", "variable"),
            ("import_statement", "import"),
            ("export_statement", "export"),
        ),
        "python": (
            ("import_declaration", "import"),
            ("import_from_declaration", "import_from"),
            ("function_definition", "function"),
            ("class_definition", "class"),
            ("async_function_definition", "async_function"),
            ("type_expression", "type"),
            ("type_alias", "type_alias"),
            ("return_type", "return_type"),
        ),
        "typescript": (
            ("comment", "comment"),
            ("function_declaration", "function"),
            ("arguments", "argument"),
            ("formal_parameters", "parameter"),
            ("class_declaration", "class"),
            ("arrow_function", "arrow_function"),
            ("method_definition", "method"),
            ("interface_declaration", "interface"),
            ("type_alias", "type_alias"),
            ("type_annotation", "type_annotation"),
            ("type_parameter", "type_parameter"),
            ("return_type", "return_type"),
            ("variable_declaration", "variable"),
            ("import_statement", "import"),
            ("export_statement", "export"),
        ),  # TypeScript shares JS patterns
        "rust": (
            ("function_item", "function"),
            ("struct_item", "struct"),
            ("enum_item", "enum"),
            ("impl_item", "impl"),
            ("trait_item", "trait"),
        ),
        "go": (
            ("function_declaration", "function"),
            ("method_declaration", "method"),
            ("type_declaration", "type"),
            ("struct_type", "struct"),
            ("interface_type", "interface"),
        ),
        "java": (
            ("method_declaration", "method"),
            ("class_declaration", "class"),
            ("interface_declaration", "interface"),
            ("constructor_declaration", "constructor"),
        ),
        "c": (
            ("function_definition", "function"),
            ("struct_specifier", "struct"),
            ("union_specifier", "union"),
            ("enum_specifier", "enum"),
        ),
        "cpp": (
            ("function_definition", "function"),
            ("class_specifier", "class"),
            ("struct_specifier", "struct"),
            ("namespace_definition", "namespace"),
        ),
    }

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the chunking middleware.

        Args:
            config: Configuration dictionary with chunking parameters
        """
        self.config = config or {}
        self.max_chunk_size = self.config.get("max_chunk_size", 1500)
        self.min_chunk_size = self.config.get("min_chunk_size", 50)

        logger.info(
            "ChunkingMiddleware initialized: max_size=%d, min_size=%d, ast_grep=%s",
            self.max_chunk_size,
            self.min_chunk_size,
        )

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, mt.CallToolResult],
    ) -> mt.CallToolResult:
        """Handle tool calls that need chunking services."""
        # Check if this tool needs chunking services
        if self._needs_chunking_service(context):
            # Inject chunking service into context
            logger.debug("Injected chunking service for tool: %s", context.message.name)
            # TODO: Implement actual service injection logic
            # wrapper = make_middleware_wrapper(self, call_next)

        return await call_next(context)

    def _needs_chunking_service(self, context: MiddlewareContext[mt.CallToolRequestParams]) -> bool:
        """Check if this tool call needs chunking services."""
        if not hasattr(context.message, "name"):
            return False

        chunking_tools = {"index_codebase", "chunk_file", "analyze_code"}
        return context.message.name in chunking_tools

    async def chunk_file(self, file_path: Path, content: str) -> list[CodeChunk]:
        """Chunk file content using AST-grep or fallback methods.

        Args:
            file_path: Path to the file being chunked
            content: File content to chunk

        Returns:
            List of CodeChunk objects representing chunks
        """
        language = self._detect_language(file_path)

        if language in self.CHUNK_PATTERNS:
            chunks = await self._chunk_with_ast_grep(content, language, file_path)
        else:
            chunks = await self._chunk_with_fallback(content, file_path, language)

        logger.debug(
            "Chunked %s: %d chunks (language: %s, ast_grep: %s)",
            file_path.name,
            len(chunks),
            language,
            language in self.CHUNK_PATTERNS,
        )

        return chunks

    def _detect_language(self, file_path: Path) -> SemanticSearchLanguage | Literal["unknown"]:
        """Detect programming language from file extension."""
        suffix = file_path.suffix.lower()

        return self.SUPPORTED_LANGUAGES.get(suffix, "unknown")

    async def _chunk_with_ast_grep(
        self, content: str, language: str, file_path: Path
    ) -> list[CodeChunk]:
        """Chunk content using AST-grep patterns."""
        try:
            root = SgRoot(content, language)
            patterns = self.CHUNK_PATTERNS[language]
            chunks: list[CodeChunk] = []

            for pattern_name, chunk_type in patterns:
                config = Config(rule={"pattern": pattern_name})
                matched_nodes: list[SgNode] = root.root().find_all(config)
                for node in matched_nodes:
                    chunk_content = node.text()

                    # Filter by size constraints
                    if not (self.min_chunk_size <= len(chunk_content) <= self.max_chunk_size):
                        continue
                    node_range = node.range()
                    # Create CodeChunk for this chunk
                    chunk = CodeChunk.create_with_hash(
                        content=chunk_content,
                        file_path=str(file_path),
                        start_line=node_range.start.line,
                        end_line=node_range.end.line,
                        chunk_type=chunk_type,
                        language=language,
                        node_kind=pattern_name,
                        metadata={"ast_grep_used": True},
                    )
                    chunks.append(chunk)

        except Exception as e:
            logger.warning("AST-grep chunking failed for %s: %s", file_path, e)
            # Fall back to simple chunking
            return await self._chunk_with_fallback(content, file_path, language)

        else:
            return chunks

    async def _chunk_with_fallback(
        self, content: str, file_path: Path, language: str = "unknown"
    ) -> list[CodeChunk]:
        """Fallback chunking using line-based approach."""
        chunks: list[CodeChunk] = []
        lines = content.split("\n")
        current_chunk = []
        current_size = 0
        start_line = 1

        for line_num, line in enumerate(lines, 1):
            line_size = len(line) + 1  # +1 for newline

            # Check if adding this line would exceed max size
            if current_size + line_size > self.max_chunk_size and current_chunk:
                # Create chunk from current lines
                chunk_content = "\n".join(current_chunk)
                if len(chunk_content) >= self.min_chunk_size:
                    chunk = CodeChunk.create_with_hash(
                        content=chunk_content,
                        file_path=str(file_path),
                        start_line=start_line,
                        end_line=line_num - 1,
                        chunk_type="fallback_chunk",
                        language=language,
                        node_kind=None,
                        metadata={"ast_grep_used": False},
                    )
                    chunks.append(chunk)

                # Start new chunk
                current_chunk = [line]
                current_size = line_size
                start_line = line_num
            else:
                current_chunk.append(line)
                current_size += line_size

        # Handle remaining chunk
        if current_chunk:
            chunk_content = "\n".join(current_chunk)
            if len(chunk_content) >= self.min_chunk_size:
                chunk = CodeChunk.create_with_hash(
                    content=chunk_content,
                    file_path=str(file_path),
                    start_line=start_line,
                    end_line=len(lines),
                    chunk_type="fallback_chunk",
                    language=language,
                    node_kind=None,
                    metadata={"ast_grep_used": False},
                )
                chunks.append(chunk)

        return chunks

    def get_supported_languages(self) -> dict[str, Any]:
        """Get information about supported languages and capabilities."""
        return {
            "supported_languages": list(set(self.SUPPORTED_LANGUAGES.values())),
            "language_extensions": self.SUPPORTED_LANGUAGES,
            "chunk_patterns": {
                lang: [pattern for pattern, _ in patterns]
                for lang, patterns in self.CHUNK_PATTERNS.items()
            },
            "config": {
                "max_chunk_size": self.max_chunk_size,
                "min_chunk_size": self.min_chunk_size,
            },
        }
