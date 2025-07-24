# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Code chunking functionality using ast-grep for semantic parsing.

Handles intelligent code chunking using ast-grep's tree-sitter parsers
for 20+ programming languages with fallback parsing.
"""

import contextlib
import logging
import re

from pathlib import Path
from typing import ClassVar

from codeweaver.models import CodeChunk


logger = logging.getLogger(__name__)

# ast-grep for proper tree-sitter parsing
try:
    from ast_grep_py import SgRoot

    AST_GREP_AVAILABLE = True
except ImportError:
    AST_GREP_AVAILABLE = False
    logger.warning("ast-grep-py not available, falling back to simple chunking")


class AstGrepChunker:
    """Handles semantic chunking using ast-grep's tree-sitter parsers for 20+ languages."""

    # Languages supported by ast-grep
    SUPPORTED_LANGUAGES: ClassVar[dict[str, str]] = {
        # Web technologies
        ".html": "html",
        ".htm": "html",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".css": "css",
        # Systems programming
        ".rs": "rust",
        ".go": "go",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".hpp": "cpp",
        ".hxx": "cpp",
        ".cs": "csharp",
        # High-level languages
        ".py": "python",
        ".java": "java",
        ".kt": "kotlin",
        ".scala": "scala",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".lua": "lua",
        ".hs": "haskell",
        ".ex": "elixir",
        ".exs": "elixir",
        # Data and config
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".xml": "xml",
        # Documentation
        ".md": "markdown",
        ".markdown": "markdown",
        ".rst": "rst",
        ".txt": "text",
        # Shell and scripts
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".nix": "nix",
        # Build systems
        ".gradle": "kotlin",  # Gradle uses Kotlin DSL
        ".bazel": "python",  # Bazel uses Python-like syntax
        ".bzl": "python",
    }

    # Chunk types we look for in different languages
    CHUNK_PATTERNS: ClassVar[dict[list[tuple[str, str]]]] = {
        "python": [
            ("function_definition", "function"),
            ("class_definition", "class"),
            ("async_function_definition", "async_function"),
        ],
        "javascript": [
            ("function_declaration", "function"),
            ("function_expression", "function"),
            ("arrow_function", "function"),
            ("class_declaration", "class"),
            ("method_definition", "method"),
        ],
        "typescript": [
            ("function_declaration", "function"),
            ("function_expression", "function"),
            ("arrow_function", "function"),
            ("class_declaration", "class"),
            ("method_definition", "method"),
            ("interface_declaration", "interface"),
            ("type_alias_declaration", "type"),
            ("enum_declaration", "enum"),
        ],
        "rust": [
            ("function_item", "function"),
            ("impl_item", "impl"),
            ("struct_item", "struct"),
            ("enum_item", "enum"),
            ("trait_item", "trait"),
            ("mod_item", "module"),
            ("macro_definition", "macro"),
        ],
        "go": [
            ("function_declaration", "function"),
            ("method_declaration", "method"),
            ("type_declaration", "type"),
            ("interface_type", "interface"),
            ("struct_type", "struct"),
        ],
        "java": [
            ("method_declaration", "method"),
            ("class_declaration", "class"),
            ("interface_declaration", "interface"),
            ("enum_declaration", "enum"),
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
            ("template_declaration", "template"),
        ],
        "markdown": [
            # Ast-grep doesn't support markdown, will use fallback chunking
        ],
        "rst": [
            # Ast-grep doesn't support markdown, will use fallback chunking
        ],
        "text": [
            # Ast-grep doesn't support plaintext, will use fallback chunking
        ],
        "toml": [
            # Ast-grep doesn't support toml, will use fallback chunking
        ],
        "xml": [
            # Ast-grep doesn't support xml, will use fallback chunking
        ],
    }

    def __init__(self, max_chunk_size: int = 1500, min_chunk_size: int = 50):
        """Initialize the AstGrepChunker with size constraints.

        Args:
            max_chunk_size: Maximum characters per chunk (default: 1500)
            min_chunk_size: Minimum characters per chunk (default: 50)
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

        if not AST_GREP_AVAILABLE:
            logger.warning("ast-grep not available, install with: [uv] pip install ast-grep-py")

    def chunk_file(self, file_path: Path, content: str) -> list[CodeChunk]:
        """Chunk a file into semantic pieces using ast-grep."""
        if not content.strip():
            return []

        language = self._detect_language(file_path)

        if not AST_GREP_AVAILABLE or language not in self.CHUNK_PATTERNS:
            return self._fallback_chunk(file_path, content, language)

        try:
            return self._ast_grep_chunk(file_path, content, language)
        except Exception as e:
            logger.warning("ast-grep chunking failed for %s: %s, falling back", file_path, e)
            return self._fallback_chunk(file_path, content, language)

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        suffix = file_path.suffix.lower()

        # Handle special cases
        if file_path.name.lower() in ("dockerfile", "containerfile"):
            return "docker"
        if file_path.name.lower() in ("makefile", "gnumakefile"):
            return "make"
        if file_path.name.lower().endswith(".cmake"):
            return "cmake"

        return self.SUPPORTED_LANGUAGES.get(suffix, "text")

    def _ast_grep_chunk(self, file_path: Path, content: str, language: str) -> list[CodeChunk]:
        """Use ast-grep to semantically chunk code."""
        try:
            # Parse with ast-grep
            root = SgRoot(content, language)
            tree = root.root()

            chunks = []
            patterns = self.CHUNK_PATTERNS.get(language, [])

            # Find semantic chunks using ast-grep patterns
            for node_kind, chunk_type in patterns:
                matches = self._find_nodes_by_kind(tree, node_kind)

                for match in matches:
                    chunk = self._create_chunk_from_node(
                        file_path, match, content, chunk_type, language, node_kind
                    )
                    if chunk and len(chunk.content.strip()) >= self.min_chunk_size:
                        chunks.append(chunk)

            # If no semantic chunks found, fall back to block-level chunking
            if not chunks:
                chunks = self._block_chunk_with_ast_grep(file_path, content, language, tree)

            return self._merge_overlapping_chunks(chunks)

        except Exception:
            logger.exception("ast-grep parsing failed for %s", language)
            raise

    def _find_nodes_by_kind(self, root_node, kind: str) -> list:
        """Find all nodes of a specific kind in the AST."""
        matches = []

        # Use ast-grep's find method with pattern matching
        pattern = f"$_:{kind}"  # Generic pattern to match node kind

        # For specific known patterns, use more precise matching
        if kind == "function_definition":
            pattern = "def $_($$$_): $$$_"
        elif kind == "class_definition":
            pattern = "class $_($$$_): $$$_"
        elif kind == "function_declaration":
            pattern = "function $_($$$_) { $$$_ }"
        elif kind == "class_declaration":
            pattern = "class $_ { $$$_ }"

        try:
            if found := root_node.find(pattern=pattern):
                matches.append(found)
        except Exception:
            # If pattern matching fails, traverse manually
            matches = self._traverse_for_kind(root_node, kind)

        return matches

    def _traverse_for_kind(self, node, target_kind: str) -> list:
        """Manually traverse AST to find nodes of target kind."""
        matches = []

        # This is a simplified traversal - ast-grep has better methods
        # but this provides a fallback
        with contextlib.suppress(Exception):
            if hasattr(node, "kind") and node.kind() == target_kind:
                matches.append(node)

            # Recursively check children
            if hasattr(node, "children"):
                for child in node.children():
                    matches.extend(self._traverse_for_kind(child, target_kind))
        return matches

    def _create_chunk_from_node(
        self, file_path: Path, node, content: str, chunk_type: str, language: str, node_kind: str
    ) -> CodeChunk | None:
        """Create a CodeChunk from an ast-grep node."""
        try:
            # Get node range
            range_info = node.range()
            start_pos = range_info.start
            end_pos = range_info.end

            # Extract content
            lines = content.split("\n")
            chunk_lines = lines[start_pos.line : end_pos.line + 1]

            if not chunk_lines:
                return None

            # Handle partial first and last lines
            if len(chunk_lines) == 1:
                chunk_content = chunk_lines[0][start_pos.column : end_pos.column]
            else:
                chunk_lines[0] = chunk_lines[0][start_pos.column :]
                chunk_lines[-1] = chunk_lines[-1][: end_pos.column]
                chunk_content = "\n".join(chunk_lines)

            if not chunk_content.strip():
                return None

            return CodeChunk.create_with_hash(
                content=chunk_content,
                file_path=str(file_path),
                start_line=start_pos.line + 1,  # 1-indexed
                end_line=end_pos.line + 1,  # 1-indexed
                chunk_type=chunk_type,
                language=language,
                node_kind=node_kind,
            )

        except Exception as e:
            logger.warning("Failed to create chunk from node: %s", e)
            return None

    def _block_chunk_with_ast_grep(
        self, file_path: Path, content: str, language: str, tree
    ) -> list[CodeChunk]:
        """Create larger block chunks when no semantic chunks found."""
        lines = content.split("\n")
        chunks = []

        # Create reasonably sized chunks
        chunk_size = min(self.max_chunk_size // 20, 50)  # ~50 lines per chunk

        for i in range(0, len(lines), chunk_size):
            end_idx = min(i + chunk_size, len(lines))
            chunk_lines = lines[i:end_idx]

            if chunk_lines and any(line.strip() for line in chunk_lines):
                chunk_content = "\n".join(chunk_lines)

                chunks.append(
                    CodeChunk.create_with_hash(
                        content=chunk_content,
                        file_path=str(file_path),
                        start_line=i + 1,
                        end_line=end_idx,
                        chunk_type="block",
                        language=language,
                        node_kind="block",
                    )
                )

        return chunks

    def _fallback_chunk(self, file_path: Path, content: str, language: str) -> list[CodeChunk]:
        """Simple fallback chunking when ast-grep is not available."""
        lines = content.split("\n")
        chunks = []

        # Look for common patterns with regex
        patterns = {
            "python": [
                (r"^(class\s+\w+.*?:)", "class"),
                (r"^(def\s+\w+.*?:)", "function"),
                (r"^(async\s+def\s+\w+.*?:)", "async_function"),
            ],
            "javascript": [
                (r"^(function\s+\w+.*?\{)", "function"),
                (r"^(class\s+\w+.*?\{)", "class"),
                (r"^(\w+\s*=\s*function.*?\{)", "function"),
                (r"^(\w+\s*=>\s*\{)", "function"),
            ],
            "rust": [
                (r"^(fn\s+\w+.*?\{)", "function"),
                (r"^(struct\s+\w+.*?\{)", "struct"),
                (r"^(impl\s+.*?\{)", "impl"),
                (r"^(enum\s+\w+.*?\{)", "enum"),
            ],
            "markdown": [
                (r"^(#{1,6}\s+.+)", "heading"),
                (r"^(```\w*)", "code_block"),
                (r"^(\s*[-*+]\s+.+)", "list_item"),
                (r"^(\d+\.\s+.+)", "numbered_list"),
            ],
            "rst": [(r"^(.+\n[=-]+)", "heading"), (r"^(\.\.\s+code-block::)", "code_block")],
            "toml": [(r"^(\[.+\])", "section")],
            "xml": [(r"^(<\w+[^>]*>)", "element")],
        }

        lang_patterns = patterns.get(language, [])

        if lang_patterns:
            current_chunk = []
            current_start = 0
            current_type = "block"

            for i, line in enumerate(lines):
                # Check if line starts a new semantic block
                for pattern, chunk_type in lang_patterns:
                    if re.match(pattern, line.strip()):
                        # Save previous chunk
                        if current_chunk:
                            chunks.append(
                                self._create_fallback_chunk(
                                    file_path,
                                    current_chunk,
                                    current_start,
                                    i - 1,
                                    current_type,
                                    language,
                                )
                            )

                        # Start new chunk
                        current_chunk = [line]
                        current_start = i
                        current_type = chunk_type
                        break
                else:
                    current_chunk.append(line)

            # Add final chunk
            if current_chunk:
                chunks.append(
                    self._create_fallback_chunk(
                        file_path,
                        current_chunk,
                        current_start,
                        len(lines) - 1,
                        current_type,
                        language,
                    )
                )

        else:
            # Pure line-based chunking
            chunk_size = 30  # Lines per chunk
            for i in range(0, len(lines), chunk_size):
                end_idx = min(i + chunk_size, len(lines))
                chunk_lines = lines[i:end_idx]

                if chunk_lines and any(line.strip() for line in chunk_lines):
                    chunks.append(
                        self._create_fallback_chunk(
                            file_path, chunk_lines, i, end_idx - 1, "block", language
                        )
                    )

        return chunks

    def _create_fallback_chunk(
        self,
        file_path: Path,
        lines: list[str],
        start_line: int,
        end_line: int,
        chunk_type: str,
        language: str,
    ) -> CodeChunk:
        """Create a CodeChunk from lines (fallback method)."""
        content = "\n".join(lines)

        return CodeChunk.create_with_hash(
            content=content,
            file_path=str(file_path),
            start_line=start_line + 1,  # 1-indexed
            end_line=end_line + 1,  # 1-indexed
            chunk_type=chunk_type,
            language=language,
            node_kind=None,
        )

    def _merge_overlapping_chunks(self, chunks: list[CodeChunk]) -> list[CodeChunk]:
        """Merge chunks that are too small or overlapping."""
        if not chunks:
            return chunks

        # Sort by file path and start line
        chunks.sort(key=lambda c: (c.file_path, c.start_line))

        merged = []
        current = chunks[0]

        for next_chunk in chunks[1:]:
            # Merge if chunks are from same file and very small or overlapping
            if current.file_path == next_chunk.file_path and (
                len(current.content) < self.min_chunk_size
                or current.end_line >= next_chunk.start_line - 1
            ):
                # Merge chunks
                merged_content = current.content + "\n" + next_chunk.content
                current = CodeChunk.create_with_hash(
                    content=merged_content,
                    file_path=current.file_path,
                    start_line=current.start_line,
                    end_line=next_chunk.end_line,
                    chunk_type=current.chunk_type,
                    language=current.language,
                    node_kind=current.node_kind,
                )
            else:
                merged.append(current)
                current = next_chunk

        merged.append(current)
        return merged
