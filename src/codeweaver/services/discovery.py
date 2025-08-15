# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""File discovery service with rignore integration."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal, LiteralString, NamedTuple

import rignore  # type: ignore

from codeweaver._constants import get_ext_lang_pairs
from codeweaver._utils import normalize_ext
from codeweaver.exceptions import IndexingError
from codeweaver.language import SemanticSearchLanguage


if TYPE_CHECKING:
    from codeweaver.settings import CodeWeaverSettings

TEST_FILE_PATTERNS = ["*.test.*", "*.spec.*", "test/**/*", "spec/**/*"]

# TODO: Add process_filename implementation and probably remove detect_language. It probably makes the most sense to unify returning paths, ExtKind/language detection into a single data structure. A TypedDict would do it, or just add to ExtKind.


class ExtKind(NamedTuple):
    """Represents a file extension and its associated kind."""

    language: str
    kind: Literal["code", "config", "docs", "other"]

    def __str__(self) -> str:
        """Return a string representation of the extension kind."""
        return f"{self.kind}: {self.language}"


def process_filename(filename: str) -> ExtKind | None:
    """Process a filename to extract its base name and extension."""
    # The order we do this in is important:
    if semantic_config_file := next(
        (
            config
            for config in iter(SemanticSearchLanguage.filename_pairs())
            if config.filename == filename
        ),
        None,
    ):
        return ExtKind(language=semantic_config_file.language.value, kind="config")
    filename_parts = tuple(part for part in filename.split(".") if part)
    extension = normalize_ext(filename_parts[-1]) if filename_parts else filename_parts[0].lower()
    if (semantic_config_language := _has_semantic_extension(extension)) and _is_semantic_config_ext(
        extension
    ):
        return ExtKind(language=semantic_config_language.value, kind="config")
    if semantic_language := _has_semantic_extension(extension):
        return ExtKind(language=semantic_language.value, kind="code")
    return next(
        (
            ExtKind(language=extpair.language, kind=extpair.category)
            for extpair in get_ext_lang_pairs()
            if extpair.is_same(filename)
        ),
        None,
    )


@cache
def _is_semantic_config_ext(ext: str) -> bool:
    """Check if the given extension is a semantic config file."""
    ext = normalize_ext(ext)
    return any(ext == config_ext for config_ext in SemanticSearchLanguage.config_language_exts())


@cache
def _has_semantic_extension(ext: str) -> SemanticSearchLanguage | None:
    """Check if the given extension is a semantic search language."""
    if found_lang := next(
        (lang for lang_ext, lang in SemanticSearchLanguage.ext_pairs() if lang_ext == ext), None
    ):
        return found_lang
    return None


class FileDiscoveryService:
    """Service for discovering and filtering files in a codebase.

    Integrates with rignore for gitignore support and provides
    language-aware file filtering.
    """

    def __init__(self, settings: CodeWeaverSettings) -> None:
        """Initialize file discovery service.

        Args:
            settings: CodeWeaver configuration settings
        """
        self.settings = settings
        self._language_extensions = SemanticSearchLanguage.extension_map()

    async def discover_files(
        self,
        *,
        include_tests: bool | None = None,
        max_file_size: int | None = None,
        read_git_ignore: bool | None = None,
        read_ignore_files: bool | None = None,
        ignore_hidden: bool | None = None,
        additional_ignore_paths: list[Path] | None = None,
    ) -> list[Path]:
        """Discover files using rignore integration with filtering.

        Args:
            patterns: Optional file patterns to filter by
            include_tests: Whether to include test files

        Returns:
            List of discovered file paths

        Raises:
            IndexingError: If file discovery fails
        """
        try:
            discovered: list[Path] = []
            additional_ignore_paths = additional_ignore_paths or []
            extra_ignores = [str(path) for path in additional_ignore_paths]
            if not include_tests:
                extra_ignores.extend(TEST_FILE_PATTERNS)

            # Use rignore for gitignore support
            walker = rignore.walk(
                self.settings.project_path,
                max_filesize=max_file_size or self.settings.max_file_size,
                case_insensitive=True,
                read_git_ignore=read_git_ignore or self.settings.filter_settings.use_gitignore,
                read_ignore_files=read_ignore_files
                or self.settings.filter_settings.use_other_ignore_files,
                ignore_hidden=ignore_hidden or self.settings.filter_settings.ignore_hidden,
                additional_ignore_paths=extra_ignores,
            )

            for file_path in walker:
                # rignore returns Path objects directly
                if file_path.is_file():
                    # Convert to relative path from project root
                    try:
                        relative_path = file_path.relative_to(self.settings.project_path)
                        discovered.append(relative_path)
                    except ValueError:
                        # File is outside project root, skip
                        continue

            return sorted(discovered)

        except Exception as e:
            raise IndexingError(
                f"Failed to discover files in {self.settings.project_path}",
                details={"error": str(e)},
                suggestions=[
                    "Check that the project path exists and is readable",
                    "Verify that rignore can access the directory",
                ],
            ) from e

    def _is_supported_language(self, file_path: Path) -> bool:
        """Check if a file is in a supported programming language.

        Args:
            file_path: Path to check

        Returns:
            True if file is in a supported language
        """
        if not file_path.suffix:
            return False

        extension = file_path.suffix.lstrip(".")
        return extension in self._language_extensions

    def detect_language(self, file_path: Path) -> SemanticSearchLanguage | LiteralString | None:
        """Detect the programming language of a file.

        Args:
            file_path: Path to analyze

        Returns:
            Language name or None if not supported
        """
        from codeweaver.language import SemanticSearchLanguage

        if not file_path.suffix:
            return None

        if semantic := SemanticSearchLanguage.lang_from_ext(file_path.suffix):
            return semantic
        from codeweaver._constants import get_ext_lang_pairs

        return next(
            (pair.language for pair in get_ext_lang_pairs() if pair.is_same(file_path.name)), None
        )

    async def get_project_languages(self) -> list[SemanticSearchLanguage | LiteralString]:
        """Get all programming languages present in the project.

        Returns:
            List of detected language names
        """
        files = await self.discover_files()
        languages: set[SemanticSearchLanguage | LiteralString] = set()

        for file_path in files:
            if not file_path.is_file():
                continue
            if language := self.detect_language(file_path):
                languages.add(language)

        return sorted(
            languages,
            key=lambda lang: lang.value if isinstance(lang, SemanticSearchLanguage) else lang,
        )
