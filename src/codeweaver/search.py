# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Structural search functionality using ast-grep.

Provides direct ast-grep pattern matching capabilities for precise
code structure queries across multiple programming languages.
"""

import logging
import operator

from pathlib import Path
from typing import Any

import rignore


logger = logging.getLogger(__name__)


EXT_MAP = {
    ("bash", "bats", "cgi", "command", "env", "fcgi", "ksh", "sh", "tmux", "tool", "zsh"):
        "bash",
    ("c", "h"): "c",
    ("cc", "hpp", "cpp", "c++", "hh", "cxx", "cu", "ino"): "cpp",
    ("cs", "csx"): "csharp",
    ("css", "scss"): "css",
    ("ex", "exs"): "elixir",
    ("go",): "go",
    ("html", "htm", "xhtml"): "html",
    ("java",): "java",
    ("js", "mjs", "cjs"): "javascript",
    ("json",): "json",
    ("kt", "ktm", "kts"): "kotlin",
    ("lua",): "lua",
    ("nix",): "nix",
    ("php",): "php",
    ("py", "py3", "pyi", "bzl"): "python",
    ("rb", "gemspec", "rbw"): "ruby",
    ("rs",): "rust",
    ("scala", "sbt", "sc"): "scala",
    ("sol",): "solidity",
    ("swift",): "swift",
    ("ts", "cts", "mts"): "typescript",
    ("tsx",): "tsx",
    ("yaml", "yml"): "yaml",
}
"""Map of file extensions to their corresponding programming languages."""

LANG_MAP = {v: k for k, v in EXT_MAP.items()}
"""Map of programming languages to their file extensions."""

ALL_EXTS = tuple(sorted(((ex, lang) for ex_group, lang in EXT_MAP.items() for ex in ex_group), key=operator.itemgetter(0)))
"""Tuples of all file extensions and their corresponding languages with extension as first item in tuple (0 position) and language as second (1 position)."""

# ast-grep for structural search
try:
    from ast_grep_py import SgRoot

    AST_GREP_AVAILABLE = True
except ImportError:
    AST_GREP_AVAILABLE = False
    logger.warning("ast-grep-py not available for structural search")


def get_language_from_extension(extension: str) -> str:
    """Get the programming language from a file extension."""
    ext = extension.strip().lower().lstrip('.')
    return next((lang for exts, lang in EXT_MAP.items() if ext in exts), None)


class AstGrepStructuralSearch:
    """Provides direct ast-grep structural search capabilities."""

    def __init__(self):
        """Initialize the structural search engine.

        Checks for ast-grep availability and sets up the search capability.
        """
        self.available = AST_GREP_AVAILABLE
        if not self.available:
            logger.warning("ast-grep not available for structural search")

    async def structural_search(
        self, pattern: str, language: str | None = None, root_path: Path | str | None = None
    ) -> list[dict[str, Any]]:
        """Perform structural search using ast-grep patterns."""
        if not self.available:
            raise ValueError("ast-grep not available, install with: pip install ast-grep-py")
        if not root_path:
            root_path = Path.cwd()
        elif isinstance(root_path, str):
            root_path = Path(root_path)

        root = Path(root_path)

        # Find files for the language
        extensions = self._get_extensions_for_language(language)
        mapped_extensions = extensions if isinstance(extensions, tuple) and isinstance(extensions[0], tuple) else tuple((ext, language) for ext in extensions if ext)
        ext_all = tuple(ext for ext, _ in mapped_extensions)
        results = []

        for file in rignore.walk(root, read_ignore_files=True, read_git_ignore=True, read_git_exclude=True):
            if not file.is_file() or file.suffix[1:] not in ext_all:
                continue

            file_path = Path(file)
            try:
                content = Path(file_path).read_text(encoding="utf-8", errors="ignore")

                # Parse and search with ast-grep
                lang = language or LANG_MAP
                if not lang:
                    continue
                sg_root = SgRoot(content, language)
                tree = sg_root.root()

                if matches := tree.find(pattern=pattern):
                    range_info = matches.range()
                    results.append({
                        "file_path": str(file_path),
                        "match_content": matches.text(),
                        "start_line": range_info.start.line + 1,
                        "end_line": range_info.end.line + 1,
                        "start_column": range_info.start.column + 1,
                        "end_column": range_info.end.column + 1,
                    })

            except Exception as e:
                logger.warning("Error searching %s: %s", file_path, e)
                continue

        return results

    def _get_extensions_for_language(self, language: str | None = None) -> tuple[str, ...]:
        """Get file extensions for a language."""
        if not language:
            return ALL_EXTS
        lang = language.strip().lower()
        return LANG_MAP.get(lang, [f".{lang}"])
