# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
FastMCP middleware for code chunking services.

Provides intelligent code chunking using ast-grep's tree-sitter parsers
with fallback parsing, integrated as FastMCP middleware for service injection.
"""

import logging

from pathlib import Path
from typing import Any, ClassVar

from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.middleware.middleware import CallNext

from codeweaver._types import CodeChunk


# ast-grep for proper tree-sitter parsing
try:
    from ast_grep_py import SgRoot

    AST_GREP_AVAILABLE = True
except ImportError:
    AST_GREP_AVAILABLE = False


logger = logging.getLogger(__name__)


class ChunkingMiddleware(Middleware):
    """FastMCP middleware providing intelligent code chunking services."""

    # Languages supported by ast-grep
    SUPPORTED_LANGUAGES: ClassVar[dict[str, str]] = {
        # Web technologies
        ".html": "html",
        ".htm": "html",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".css": "css",
        ".scss": "scss",
        ".sass": "sass",
        ".less": "less",
        ".vue": "vue",
        ".svelte": "svelte",
        # Systems programming
        ".rs": "rust",
        ".go": "go",
        ".c": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".zig": "zig",
        # High-level languages
        ".py": "python",
        ".java": "java",
        ".kt": "kotlin",
        ".scala": "scala",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        # Functional languages
        ".ex": "elixir",
        ".exs": "elixir",
        # Data/config
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        # Scripts
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
    }

    # AST-grep patterns for semantic chunking by language
    CHUNK_PATTERNS: ClassVar[dict[str, list[tuple[str, str]]]] = {
        "python": [
            ("function_definition", "function"),
            ("class_definition", "class"),
            ("async_function_definition", "async_function"),
        ],
        "javascript": [
            ("function_declaration", "function"),
            ("class_declaration", "class"),
            ("method_definition", "method"),
            ("arrow_function", "arrow_function"),
        ],
        "typescript": [
            ("function_declaration", "function"),
            ("class_declaration", "class"),
            ("method_definition", "method"),
            ("interface_declaration", "interface"),
            ("type_alias_declaration", "type"),
        ],
        "rust": [
            ("function_item", "function"),
            ("struct_item", "struct"),
            ("enum_item", "enum"),
            ("impl_item", "impl"),
            ("trait_item", "trait"),
        ],
        "go": [
            ("function_declaration", "function"),
            ("method_declaration", "method"),
            ("type_declaration", "type"),
            ("struct_type", "struct"),
            ("interface_type", "interface"),
        ],
        "java": [
            ("method_declaration", "method"),
            ("class_declaration", "class"),
            ("interface_declaration", "interface"),
            ("constructor_declaration", "constructor"),
        ],
        "c": [
            ("function_definition", "function"),
            ("struct_specifier", "struct"),
            ("union_specifier", "union"),
            ("enum_specifier", "enum"),
        ],
        "cpp": [
            ("function_definition", "function"),
            ("class_specifier", "class"),
            ("struct_specifier", "struct"),
            ("namespace_definition", "namespace"),
        ],
    }

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the chunking middleware.

        Args:
            config: Configuration dictionary with chunking parameters
        """
        self.config = config or {}
        self.max_chunk_size = self.config.get("max_chunk_size", 1500)
        self.min_chunk_size = self.config.get("min_chunk_size", 50)
        self.ast_grep_enabled = self.config.get("ast_grep_enabled", True) and AST_GREP_AVAILABLE

        logger.info(
            "ChunkingMiddleware initialized: max_size=%d, min_size=%d, ast_grep=%s",
            self.max_chunk_size,
            self.min_chunk_size,
            self.ast_grep_enabled,
        )

    async def on_call_tool(self, context: MiddlewareContext, call_next: CallNext) -> Any:
        """Handle tool calls that need chunking services."""
        # Check if this tool needs chunking services
        if self._needs_chunking_service(context):
            # Inject chunking service into context
            context.fastmcp_context.set_state_value("chunking_service", self)
            logger.debug("Injected chunking service for tool: %s", context.message.name)

        return await call_next(context)

    def _needs_chunking_service(self, context: MiddlewareContext) -> bool:
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

        if self.ast_grep_enabled and language in self.CHUNK_PATTERNS:
            chunks = await self._chunk_with_ast_grep(content, language, file_path)
        else:
            chunks = await self._chunk_with_fallback(content, file_path, language)

        logger.debug(
            "Chunked %s: %d chunks (language: %s, ast_grep: %s)",
            file_path.name,
            len(chunks),
            language,
            self.ast_grep_enabled and language in self.CHUNK_PATTERNS,
        )

        return chunks

    def _detect_language(self, file_path: Path) -> str:
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
            chunks = []

            for pattern, chunk_type in patterns:
                matches = root.root().find_all(pattern)
                for match in matches:
                    chunk_content = match.text()

                    # Filter by size constraints
                    if not (self.min_chunk_size <= len(chunk_content) <= self.max_chunk_size):
                        continue

                    # Create CodeChunk for this chunk
                    chunk = CodeChunk.create_with_hash(
                        content=chunk_content,
                        file_path=str(file_path),
                        start_line=match.start_pos().line,
                        end_line=match.end_pos().line,
                        chunk_type=chunk_type,
                        language=language,
                        node_kind=pattern,
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
        chunks = []
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
            "ast_grep_available": AST_GREP_AVAILABLE,
            "ast_grep_enabled": self.ast_grep_enabled,
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
