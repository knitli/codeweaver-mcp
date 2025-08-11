# sourcery skip: avoid-single-character-names-variables
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
FastMCP middleware for file filtering services.

Provides intelligent file discovery and filtering using rignore.walk()
with gitignore support, integrated as FastMCP middleware for service injection.
"""

import fnmatch
import logging

from pathlib import Path
from typing import Any

import mcp.types as mt
import rignore

from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.middleware.middleware import CallNext


logger = logging.getLogger(__name__)


class FileFilteringMiddleware(Middleware):
    """FastMCP middleware providing file filtering and discovery services."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the file filtering middleware.

        Args:
            config: Configuration dictionary with filtering parameters
        """
        self.config = config or {}
        self.use_gitignore = self.config.get("use_gitignore", True)
        self.max_file_size = self._parse_size(self.config.get("max_file_size", "1MB"))
        self.excluded_dirs = set(
            self.config.get(
                "excluded_dirs",
                [
                    "node_modules",
                    ".git",
                    "__pycache__",
                    ".pytest_cache",
                    "build",
                    "dist",
                    ".next",
                    ".nuxt",
                    "target",
                    "bin",
                    "obj",
                ],
            )
        )
        extensions = self.config.get("included_extensions")
        self.included_extensions = set(extensions) if extensions else None
        self.additional_ignore_patterns = self.config.get("additional_ignore_patterns", [])
        logger.info(
            "FileFilteringMiddleware initialized: gitignore=%s, max_size=%s, excluded_dirs=%d",
            self.use_gitignore,
            self._format_size(self.max_file_size),
            len(self.excluded_dirs),
        )

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, mt.CallToolResult],
    ) -> mt.CallToolResult:
        """Handle tool calls that need file filtering services."""
        if self._needs_filtering_service(context):
            fast_ctx = getattr(context, "fastmcp_context", None)
            if fast_ctx is not None and hasattr(fast_ctx, "set_state_value"):
                fast_ctx.set_state_value("filtering_service", self)
            logger.debug("Injected filtering service for tool: %s", context.message.name)
        return await call_next(context)

    def _needs_filtering_service(
        self, context: MiddlewareContext[mt.CallToolRequestParams]
    ) -> bool:
        """Check if this tool call needs filtering services."""
        if not hasattr(context.message, "name"):
            return False
        filtering_tools = {"index_codebase", "search_code", "find_files", "ast_grep_search"}
        return context.message.name in filtering_tools

    async def find_files(
        self, base_path: Path, patterns: list[str] | None = None, *, recursive: bool = True
    ) -> list[Path]:
        """Find files using rignore.walk() with filtering criteria.

        Args:
            base_path: Base directory to search in
            patterns: Optional glob patterns to match (defaults to all files)
            recursive: Whether to search recursively

        Returns:
            List of filtered file paths
        """
        patterns = patterns or ["*"]
        found_files: list[Path] = []
        if not base_path.exists():
            logger.warning("Base path does not exist: %s", base_path)
            return found_files
        try:
            if self.use_gitignore:
                walker = rignore.walk(base_path)
                for entry in walker:
                    if entry.is_file():
                        file_path = Path(entry)
                        if await self._should_include_file(
                            file_path, base_path
                        ) and self._matches_patterns(file_path, patterns):
                            found_files.append(file_path)
            else:
                logger.warning("rignore not available, using fallback file discovery")
                found_files = await self._fallback_file_discovery(
                    base_path, patterns, recursive=recursive
                )
        except Exception:
            logger.exception("File discovery error")
            found_files = await self._fallback_file_discovery(
                base_path, patterns, recursive=recursive
            )
        logger.debug("Found %d files in %s (patterns: %s)", len(found_files), base_path, patterns)
        return found_files

    # python
    async def _fallback_file_discovery(
        self, base_path: Path, patterns: list[str], *, recursive: bool
    ) -> list[Path]:
        """Fallback file discovery using Path.rglob()."""
        found_files: list[Path] = []
        try:
            for pattern in patterns:
                matches = base_path.rglob(pattern) if recursive else base_path.glob(pattern)
                for file_path in matches:
                    if file_path.is_file() and await self._should_include_file(
                        file_path, base_path
                    ):
                        found_files.append(file_path)  # noqa: PERF401  # with the async context, `extend` won't work
        except Exception:
            logger.exception("Fallback file discovery error")
        return found_files

    async def _should_include_file(self, file_path: Path, base_path: Path) -> bool:
        """Check if file should be included based on filtering criteria."""
        if not self._is_valid_file_size(file_path):
            return False
        if not self._is_under_base_path(file_path, base_path):
            return False
        if not self._has_allowed_extension(file_path):
            return False
        return not self._matches_ignore_patterns(file_path)

    def _is_valid_file_size(self, file_path: Path) -> bool:
        """Check if file has valid size (not empty and not too large)."""
        try:
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                logger.debug("File too large (%s): %s", self._format_size(file_size), file_path)
                return False
        except OSError as e:
            logger.debug("Cannot stat file %s: %s", file_path, e)
            return False
        else:
            return file_size != 0

    def _is_under_base_path(self, file_path: Path, base_path: Path) -> bool:
        """Check if file is under base path and not in excluded directories."""
        try:
            relative_path = file_path.relative_to(base_path)
        except ValueError:
            logger.debug("File not under base path: %s", file_path)
            return False
        else:
            return all(part not in self.excluded_dirs for part in relative_path.parts[:-1])

    def _has_allowed_extension(self, file_path: Path) -> bool:
        """Check if file has an allowed extension."""
        if not self.included_extensions:
            return True
        return file_path.suffix.lower() in self.included_extensions

    def _matches_ignore_patterns(self, file_path: Path) -> bool:
        """Check if file matches any additional ignore patterns."""
        if not self.additional_ignore_patterns:
            return False
        return any(
            fnmatch.fnmatch(str(file_path), pattern) for pattern in self.additional_ignore_patterns
        )

    def _matches_patterns(self, file_path: Path, patterns: list[str]) -> bool:
        """Check if file matches any of the given patterns."""
        for pattern in patterns:
            if fnmatch.fnmatch(file_path.name, pattern):
                return True
            if fnmatch.fnmatch(str(file_path), pattern):
                return True
        return False

    def _get_size_unit(self, parsed_size: str) -> str:
        """Extract size unit from a string like '1MB'."""
        if (
            all(char.isdigit() for char in parsed_size)
            or (parsed_size.endswith("B") and len(parsed_size) > 1 and parsed_size[-2].isdigit())
            or len(parsed_size) == 1
        ):
            return "B"
        match parsed_size.upper()[-2:]:
            case "GB":
                return "GB"
            case "MB":
                return "MB"
            case "KB":
                return "KB"
            case _:
                return "B"

    def _parse_size(self, value: str | int) -> int:
        # sourcery skip: extract-method
        """Parse size like '1MB', '512KB', '2048', or 2048 to bytes."""
        if isinstance(value, int):
            return value

        s = (
            str(value)
            .upper()
            .replace(" ", "")
            .replace("-", "")
            .replace("_", "")
            .replace(",", ".")
            .strip()
        )

        if not s:
            logger.warning("Empty size string provided, defaulting to 0B")
            return 0

        # Strip a trailing 'B' when it's part of the unit (e.g., '10MB' or '1024B')
        if len(s) > 1 and s.endswith("B") and s[-2].isdigit():
            s = s[:-1]

        try:
            if s.endswith("TB"):
                num = float(s[:-2])
                return int(num * 1024**4)
            if s.endswith("GB"):
                num = float(s[:-2])
                return int(num * 1024**3)
            if s.endswith("MB"):
                num = float(s[:-2])
                return int(num * 1024**2)
            if s.endswith("KB"):
                num = float(s[:-2])
                return int(num * 1024)

            # If it ends with a non-digit unit we don't explicitly handle (like '1M'), drop the suffix.
            if s and not s[-1].isdigit():
                num = float(s[:-1]) if len(s) > 1 else 0.0
                return int(num)

            # Plain number (bytes)
            return int(float(s))
        except Exception:
            logger.warning("Invalid size string %s, defaulting to 0B", value)
            return 0

    def _format_size(self, size_bytes: int) -> str:
        """Format size in bytes to human-readable string."""
        size = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024.0:
                return f"{size:.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}TB"

    def get_filtering_stats(self) -> dict[str, Any]:
        """Get statistics about filtering configuration."""
        return {
            "use_gitignore": self.use_gitignore,
            "rignore_available": True,
            "max_file_size": self.max_file_size,
            "max_file_size_formatted": self._format_size(self.max_file_size),
            "excluded_dirs": list(self.excluded_dirs),
            "included_extensions": list(self.included_extensions)
            if self.included_extensions
            else None,
            "additional_ignore_patterns": self.additional_ignore_patterns,
        }
