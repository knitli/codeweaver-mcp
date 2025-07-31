"""
FastMCP middleware for file filtering services.

Provides intelligent file discovery and filtering using rignore.walk()
with gitignore support, integrated as FastMCP middleware for service injection.
"""

import fnmatch
import logging

from pathlib import Path
from typing import Any

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

    async def on_call_tool(self, context: MiddlewareContext, call_next: CallNext) -> Any:
        """Handle tool calls that need file filtering services."""
        if self._needs_filtering_service(context):
            context.fastmcp_context.set_state_value("filtering_service", self)
            logger.debug("Injected filtering service for tool: %s", context.message.name)
        return await call_next(context)

    def _needs_filtering_service(self, context: MiddlewareContext) -> bool:
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
        found_files = []
        if not base_path.exists():
            logger.warning("Base path does not exist: %s", base_path)
            return found_files
        try:
            if self.use_gitignore:
                walker = rignore.walk(str(base_path))
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

    async def _fallback_file_discovery(
        self, base_path: Path, patterns: list[str], *, recursive: bool
    ) -> list[Path]:
        """Fallback file discovery using Path.rglob()."""
        found_files = []
        try:
            for pattern in patterns:
                matches = base_path.rglob(pattern) if recursive else base_path.glob(pattern)
                found_files.extend(
                    (
                        file_path
                        for file_path in matches
                        if file_path.is_file()
                        and await self._should_include_file(file_path, base_path)
                    )
                )
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
            return all((part not in self.excluded_dirs for part in relative_path.parts[:-1]))

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
            (
                fnmatch.fnmatch(str(file_path), pattern)
                for pattern in self.additional_ignore_patterns
            )
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
            all((char.isdigit() for char in parsed_size))
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

    def _parse_size(self, parsed_size: str | int) -> int:
        """Parse size string like '1MB' to bytes."""
        if isinstance(parsed_size, int):
            return parsed_size
        parsed_size = (
            str(parsed_size)
            .upper()
            .replace(" ", "")
            .replace("-", "")
            .replace("_", "")
            .replace(",", ".")
            .strip()
        )
        if not parsed_size:
            logger.warning("Empty size string provided, defaulting to 0B")
            return 0
        if any((char.isalpha() for char in parsed_size)):
            if len(parsed_size) == 1:
                return 0
            if parsed_size[-1] == "B" and parsed_size[-2].isdigit():
                parsed_size = int(parsed_size[:-1])
            match parsed_size[-2:]:
                case "TB":
                    parsed_size = int(float(parsed_size[:-2]) * 1024 * 1024 * 1024 * 1024)
                case "GB":
                    parsed_size = int(float(parsed_size[:-2]) * 1024 * 1024 * 1024)
                case "MB":
                    parsed_size = int(float(parsed_size[:-2]) * 1024 * 1024)
                case "KB":
                    parsed_size = int(float(parsed_size[:-2]) * 1024)
                case _:
                    parsed_size = int(parsed_size[:-1])
        else:
            parsed_size = int(float(parsed_size))
        return parsed_size

    def _format_size(self, size_bytes: int) -> str:
        """Format size in bytes to human-readable string."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}TB"

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
