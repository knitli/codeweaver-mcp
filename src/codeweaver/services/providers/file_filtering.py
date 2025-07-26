# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""FastMCP filtering service provider."""

import time

from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path
from typing import Any

from codeweaver._types.config import ServiceType
from codeweaver._types.service_config import FilteringServiceConfig
from codeweaver._types.service_data import (
    DirectoryStats,
    FileMetadata,
    FilteringStats,
    ServiceCapabilities,
)
from codeweaver._types.service_exceptions import DirectoryNotFoundError, FilteringError
from codeweaver._types.services import FilteringService
from codeweaver.middleware.filtering import FileFilteringMiddleware
from codeweaver.services.providers.base_provider import BaseServiceProvider


class FilteringService(BaseServiceProvider, FilteringService):
    """FastMCP-based filtering service provider."""

    VERSION = "1.0.0"

    def __init__(self, service_type: ServiceType, config: FilteringServiceConfig) -> None:
        """Initialize the filtering service provider."""
        super().__init__(service_type, config)
        self._config = config  # Type-specific config
        self._middleware: FileFilteringMiddleware | None = None
        self._stats = FilteringStats()

        # Active patterns
        self._include_patterns = set(self._config.include_patterns)
        self._exclude_patterns = set(self._config.exclude_patterns)

    @property
    def capabilities(self) -> ServiceCapabilities:
        """Provider capabilities."""
        return ServiceCapabilities(
            supports_streaming=True,
            supports_batch=True,
            supports_async=True,
            max_concurrency=self._config.max_concurrent_scans,
            memory_usage="low",
            performance_profile="standard",
        )

    async def _initialize_provider(self) -> None:
        """Initialize FastMCP filtering middleware."""
        middleware_config = {
            "use_gitignore": self._config.use_gitignore,
            "max_file_size": self._config.max_file_size,
            "excluded_dirs": self._config.ignore_directories,
            "included_extensions": self._config.allowed_extensions or None,
            "additional_ignore_patterns": list(self._exclude_patterns),
        }

        self._middleware = FileFilteringMiddleware(middleware_config)
        self._logger.info("FastMCP filtering provider initialized")

    async def _shutdown_provider(self) -> None:
        """Shutdown filtering provider."""
        self._middleware = None
        self._logger.info("FastMCP filtering provider shut down")

    async def _check_health(self) -> bool:
        """Check if filtering service is healthy."""
        return self._middleware is not None

    async def discover_files(
        self,
        base_path: Path,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        max_depth: int | None = None,
        *,
        follow_symlinks: bool = False,
    ) -> list[Path]:
        """Discover files matching criteria."""
        if not self._middleware:
            raise FilteringError(base_path, "Filtering service not initialized")

        if not base_path.exists():
            raise DirectoryNotFoundError(base_path)

        start_time = time.time()

        try:
            # Combine patterns
            effective_include = include_patterns or list(self._include_patterns) or ["*"]
            effective_exclude = exclude_patterns or list(self._exclude_patterns)

            # Use middleware to discover files
            files = await self._middleware.find_files(
                base_path,
                patterns=effective_include,
                recursive=True,  # TODO: Handle max_depth and follow_symlinks
            )

            # Apply additional exclude patterns
            if effective_exclude:
                files = [
                    f for f in files if not self._matches_exclude_patterns(f, effective_exclude)
                ]

            # Update statistics
            scan_time = time.time() - start_time
            self._update_discovery_stats(len(files), 0, scan_time, success=True)

            self.record_operation(True)

        except Exception as e:
            scan_time = time.time() - start_time
            self._update_discovery_stats(0, 0, scan_time, success=False)

            error_msg = f"File discovery failed: {e}"
            self.record_operation(False, error_msg)
            self._logger.exception("File discovery failed for %s", base_path)

            raise FilteringError(base_path, str(e)) from e
        else:
            return files

    async def discover_files_stream(
        self,
        base_path: Path,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        max_depth: int | None = None,
        *,
        follow_symlinks: bool = False,
    ) -> AsyncGenerator[Path]:
        """Stream file discovery."""
        files = await self.discover_files(
            base_path, include_patterns, exclude_patterns, max_depth, follow_symlinks
        )
        for file_path in files:
            yield file_path

    def should_include_file(
        self,
        file_path: Path,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> bool:
        """Check if file should be included."""
        # Check include patterns
        effective_include = include_patterns or list(self._include_patterns)
        if effective_include and not self._matches_include_patterns(file_path, effective_include):
            return False

        # Check exclude patterns
        effective_exclude = exclude_patterns or list(self._exclude_patterns)
        if effective_exclude and self._matches_exclude_patterns(file_path, effective_exclude):
            return False

        # Check allowed extensions
        if (
            self._config.allowed_extensions
            and file_path.suffix.lower() not in self._config.allowed_extensions
        ):
            return False

        # Check blocked extensions
        if (
            self._config.blocked_extensions
            and file_path.suffix.lower() in self._config.blocked_extensions
        ):
            return False

        # Check file size
        try:
            if file_path.stat().st_size > self._config.max_file_size:
                return False
        except OSError:
            return False

        return True

    def should_include_directory(
        self, dir_path: Path, exclude_patterns: list[str] | None = None
    ) -> bool:
        """Determine if a directory should be traversed."""
        # Check if directory is in ignore list
        if dir_path.name in self._config.ignore_directories:
            return False

        # Check exclude patterns
        effective_exclude = exclude_patterns or list(self._exclude_patterns)
        if effective_exclude and self._matches_exclude_patterns(dir_path, effective_exclude):
            return False

        # Check if hidden and ignore_hidden is enabled
        return not (self._config.ignore_hidden and dir_path.name.startswith("."))

    async def get_file_metadata(self, file_path: Path) -> FileMetadata:
        """Get file metadata."""
        try:
            stat = file_path.stat()

            return FileMetadata(
                path=file_path,
                size=stat.st_size,
                modified_time=datetime.fromtimestamp(stat.st_mtime),
                created_time=datetime.fromtimestamp(stat.st_ctime),
                file_type=file_path.suffix.lower(),
                permissions=oct(stat.st_mode)[-3:],
                is_binary=self._is_binary_file(file_path),
            )

        except OSError as e:
            raise FilteringError(file_path, f"Cannot get metadata: {e}") from e

    async def get_directory_stats(self, dir_path: Path) -> DirectoryStats:
        """Get statistics for a directory tree."""
        if not dir_path.exists() or not dir_path.is_dir():
            raise DirectoryNotFoundError(dir_path)

        start_time = time.time()
        total_files = 0
        total_directories = 0
        total_size = 0
        file_types = {}
        largest_file = None
        largest_size = 0

        try:
            for item in dir_path.rglob("*"):
                if item.is_file():
                    total_files += 1
                    try:
                        size = item.stat().st_size
                        total_size += size

                        if size > largest_size:
                            largest_size = size
                            largest_file = item

                        file_type = item.suffix.lower() or "no_extension"
                        file_types[file_type] = file_types.get(file_type, 0) + 1

                    except OSError:
                        continue

                elif item.is_dir():
                    total_directories += 1

            scan_time = time.time() - start_time

            return DirectoryStats(
                total_files=total_files,
                total_directories=total_directories,
                total_size=total_size,
                file_types=file_types,
                largest_file=largest_file,
                scan_time=scan_time,
            )

        except Exception as e:
            raise FilteringError(dir_path, f"Cannot get directory stats: {e}") from e

    def add_include_pattern(self, pattern: str) -> None:
        """Add an include pattern to the service."""
        self._include_patterns.add(pattern)
        self._logger.debug("Added include pattern: %s", pattern)

    def add_exclude_pattern(self, pattern: str) -> None:
        """Add an exclude pattern to the service."""
        self._exclude_patterns.add(pattern)
        self._logger.debug("Added exclude pattern: %s", pattern)

    def remove_pattern(self, pattern: str, pattern_type: str) -> None:
        """Remove a pattern from the service."""
        if pattern_type == "include":
            self._include_patterns.discard(pattern)
        elif pattern_type == "exclude":
            self._exclude_patterns.discard(pattern)
        else:
            raise ValueError(f"Invalid pattern type: {pattern_type}")

        self._logger.debug("Removed %s pattern: %s", pattern_type, pattern)

    def get_active_patterns(self) -> dict[str, Any]:
        """Get currently active patterns."""
        return {
            "include_patterns": list(self._include_patterns),
            "exclude_patterns": list(self._exclude_patterns),
            "allowed_extensions": self._config.allowed_extensions,
            "blocked_extensions": self._config.blocked_extensions,
            "ignore_directories": self._config.ignore_directories,
        }

    async def get_filtering_stats(self) -> FilteringStats:
        """Get filtering performance statistics."""
        return self._stats

    async def reset_stats(self) -> None:
        """Reset filtering statistics."""
        self._stats = FilteringStats()
        self._logger.info("Filtering statistics reset")

    def _matches_include_patterns(self, file_path: Path, patterns: list[str]) -> bool:
        """Check if file matches any include patterns."""
        import fnmatch

        for pattern in patterns:
            if fnmatch.fnmatch(file_path.name, pattern):
                return True
            if fnmatch.fnmatch(str(file_path), pattern):
                return True

        return False

    def _matches_exclude_patterns(self, file_path: Path, patterns: list[str]) -> bool:
        """Check if file matches any exclude patterns."""
        import fnmatch

        for pattern in patterns:
            if fnmatch.fnmatch(file_path.name, pattern):
                return True
            if fnmatch.fnmatch(str(file_path), pattern):
                return True

        return False

    def _is_binary_file(self, file_path: Path) -> bool:
        """Check if file is binary."""
        try:
            with file_path.open("rb") as f:
                chunk = f.read(8192)
                return b"\0" in chunk
        except OSError:
            return False

    def _update_discovery_stats(
        self, files_found: int, files_excluded: int, scan_time: float, *, success: bool
    ) -> None:
        """Update filtering statistics."""
        if success:
            self._stats.total_files_scanned += files_found + files_excluded
            self._stats.total_files_included += files_found
            self._stats.total_files_excluded += files_excluded
            self._stats.total_directories_scanned += 1  # Approximate
            self._stats.total_scan_time += scan_time
        else:
            self._stats.error_count += 1
