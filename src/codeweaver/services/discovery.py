# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""File discovery service with rignore integration."""

from __future__ import annotations

import fnmatch

from pathlib import Path
from typing import TYPE_CHECKING, LiteralString

import rignore  # type: ignore

from codeweaver.exceptions import IndexingError


if TYPE_CHECKING:
    from codeweaver.language import SemanticSearchLanguage
    from codeweaver.settings import CodeWeaverSettings


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
        self, patterns: list[str] | None = None, *, include_tests: bool = True
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

            # Use rignore for gitignore support
            walker = rignore.walk(
                self.settings.project_path,
                max_filesize=self.settings.max_file_size,
                read_git_ignore=self.settings.filter_settings.use_gitignore,
                read_ignore_files=self.settings.filter_settings.use_other_ignore_files,
                ignore_hidden=self.settings.filter_settings.ignore_hidden,
                additional_ignore_paths=list(self.settings.filter_settings.excluded_dirs),
            )

            for file_path in walker:
                # rignore returns Path objects directly
                if file_path.is_file() and self._should_include_file(
                    file_path, patterns, include_tests=include_tests
                ):
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

    def _should_include_file(
        self, file_path: Path, patterns: list[str] | None, *, include_tests: bool
    ) -> bool:
        """Determine if a file should be included based on filtering rules.

        Args:
            file_path: Path to the file to check
            patterns: Optional patterns to match against
            include_tests: Whether to include test files

        Returns:
            True if file should be included
        """
        try:
            # Check file size limits
            if file_path.stat().st_size > self.settings.max_file_size:
                return False
        except OSError:
            # File doesn't exist or can't be read
            return False

        # Check excluded extensions
        if file_path.suffix.lower() in self.settings.filter_settings.excluded_extensions:
            return False

        # Check if it's a test file and tests are excluded
        if not include_tests and self._is_test_file(file_path):
            return False

        # Check if file has a supported language extension
        if not self._is_supported_language(file_path):
            return False

        # Apply pattern filtering if specified
        return not (
            patterns and not any(fnmatch.fnmatch(str(file_path), pattern) for pattern in patterns)
        )

    def _is_test_file(self, file_path: Path) -> bool:
        """Check if a file appears to be a test file.

        Args:
            file_path: Path to check

        Returns:
            True if file appears to be a test file
        """
        filename = str(file_path).lower()
        test_indicators = [
            "test",
            "tests",
            "spec",
            "specs",
            "__tests__",
            ".test.",
            ".spec.",
            "_test.",
            "_spec.",
        ]

        return any(indicator in filename for indicator in test_indicators)

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
