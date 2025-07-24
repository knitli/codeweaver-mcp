# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
File system data source implementation for CodeWeaver.

Refactors the existing file system indexing logic into the new data source
abstraction while maintaining full backward compatibility.
"""

import asyncio
import contextlib
import logging

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from codeweaver.sources.base import AbstractDataSource, ContentItem, SourceCapability, SourceWatcher


logger = logging.getLogger(__name__)


class FileSystemSourceConfig(TypedDict, total=False):
    """Configuration specific to file system data sources."""

    # Inherited from BaseSourceConfig
    enabled: bool
    priority: int
    source_id: str
    include_patterns: list[str]
    exclude_patterns: list[str]
    max_file_size_mb: int
    batch_size: int
    max_concurrent_requests: int
    request_timeout_seconds: int
    enable_change_watching: bool
    change_check_interval_seconds: int
    enable_content_deduplication: bool
    enable_metadata_extraction: bool
    supported_languages: list[str]

    # File system specific settings
    root_path: str
    follow_symlinks: bool
    use_gitignore: bool
    additional_ignore_patterns: list[str]
    recursive_discovery: bool
    file_extensions: list[str]

    # Legacy compatibility
    patterns: list[str]  # For backward compatibility with existing code


class FileSystemSourceWatcher(SourceWatcher):
    """File system specific watcher implementation."""

    def __init__(
        self,
        source_id: str,
        callback: Callable[[list[ContentItem]], None],
        root_path: Path,
        config: FileSystemSourceConfig,
    ):
        """Initialize file system watcher.

        Args:
            source_id: Source identifier
            callback: Change notification callback
            root_path: Root directory to watch
            config: File system source configuration
        """
        super().__init__(source_id, callback)
        self.root_path = root_path
        self.config = config
        self._watch_task: asyncio.Task | None = None
        self._last_scan_time = datetime.now(datetime.UTC)

    async def start(self) -> bool:
        """Start watching for file system changes."""
        if self.is_active:
            return True

        try:
            self.is_active = True
            self._watch_task = asyncio.create_task(self._watch_loop())
            logger.info("Started file system watching: %s", self.root_path)
        except Exception:
            logger.exception("Failed to start file system watching")
            self.is_active = False
            return False
        else:
            return True

    async def stop(self) -> bool:
        """Stop watching for file system changes."""
        if not self.is_active:
            return True

        try:
            self.is_active = False
            if self._watch_task:
                self._watch_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._watch_task
                self._watch_task = None

            logger.info("Stopped file system watching: %s", self.root_path)
        except Exception:
            logger.exception("Error stopping file system watching")
            return False
        else:
            return True

    async def _watch_loop(self) -> None:
        """Main watching loop for detecting file changes."""
        interval = self.config.get("change_check_interval_seconds", 60)

        while self.is_active:
            try:
                await asyncio.sleep(interval)

                if not self.is_active:
                    break

                # Simple change detection based on modification times
                changed_items = await self._detect_changes()
                if changed_items:
                    await self.notify_changes(changed_items)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in file system watch loop")
                await asyncio.sleep(interval)

    async def _detect_changes(self) -> list[ContentItem]:
        """Detect files that have changed since last scan."""
        changed_items = []
        current_time = datetime.now()

        try:
            # Find files modified since last scan
            for file_path in self.root_path.rglob("*"):
                if not file_path.is_file():
                    continue

                try:
                    stat = file_path.stat()
                    modified_time = datetime.fromtimestamp(stat.st_mtime)

                    if modified_time > self._last_scan_time:
                        # File was modified since last scan
                        item = self._path_to_content_item(file_path, stat)
                        if item:
                            changed_items.append(item)

                except (OSError, PermissionError):
                    # Skip files that can't be accessed
                    continue

            self._last_scan_time = current_time

            if changed_items:
                logger.info("Detected %d changed files in %s", len(changed_items), self.root_path)

        except Exception:
            logger.exception("Error detecting file changes")

        return changed_items

    def _path_to_content_item(self, file_path: Path, stat: Any = None) -> ContentItem | None:
        """Convert a file path to a ContentItem."""
        try:
            if stat is None:
                stat = file_path.stat()

            # Detect language from file extension
            language = self._detect_language(file_path)

            return ContentItem(
                path=str(file_path),
                content_type="file",
                metadata={
                    "file_extension": file_path.suffix,
                    "relative_path": str(file_path.relative_to(self.root_path)),
                    "parent_directory": str(file_path.parent),
                },
                last_modified=datetime.fromtimestamp(stat.st_mtime),
                size=stat.st_size,
                language=language,
                source_id=self.source_id,
                checksum=None,  # Could compute if needed
            )
        except Exception:
            logger.debug("Failed to create ContentItem for %s", file_path)
            return None

    def _detect_language(self, file_path: Path) -> str | None:
        """Detect programming language from file extension."""
        # Import here to avoid circular dependencies
        from codeweaver.chunker import AstGrepChunker

        chunker = AstGrepChunker()
        suffix = file_path.suffix.lower()
        return chunker.SUPPORTED_LANGUAGES.get(suffix)


class FileSystemSource(AbstractDataSource):
    """File system data source implementation.

    Provides content discovery from local file systems with intelligent
    file filtering, gitignore support, and change watching capabilities.
    """

    def __init__(self, source_id: str | None = None):
        """Initialize file system data source.

        Args:
            source_id: Unique identifier for this source instance
        """
        super().__init__("filesystem", source_id)

    def get_capabilities(self) -> set[SourceCapability]:
        """Get capabilities supported by file system source."""
        return {
            SourceCapability.CONTENT_DISCOVERY,
            SourceCapability.CONTENT_READING,
            SourceCapability.CHANGE_WATCHING,
            SourceCapability.METADATA_EXTRACTION,
            SourceCapability.BATCH_PROCESSING,
            SourceCapability.CONTENT_DEDUPLICATION,
        }

    async def discover_content(self, config: FileSystemSourceConfig) -> list[ContentItem]:
        """Discover files from the file system.

        Args:
            config: File system source configuration

        Returns:
            List of discovered file content items
        """
        if not config.get("enabled", True):
            return []

        root_path_str = config.get("root_path", ".")
        root_path = Path(root_path_str).resolve()

        if not root_path.exists():
            raise ValueError(f"Root path does not exist: {root_path}")

        if not root_path.is_dir():
            raise ValueError(f"Root path is not a directory: {root_path}")

        logger.info("Discovering content in: %s", root_path)

        # Use existing FileFilter for intelligent filtering
        files = await self._discover_files(root_path, config)

        # Convert to ContentItems
        content_items = []
        for file_path in files:
            item = await self._file_to_content_item(file_path, root_path)
            if item:
                content_items.append(item)

        # Apply content filters
        filtered_items = self._apply_content_filters(content_items, config)

        logger.info(
            "File system discovery complete: %d files found, %d after filtering",
            len(content_items),
            len(filtered_items),
        )

        return filtered_items

    async def read_content(self, item: ContentItem) -> str:
        """Read content from a file.

        Args:
            item: Content item representing a file

        Returns:
            Text content of the file
        """
        if item.content_type != "file":
            raise ValueError(
                f"Unsupported content type for file system source: {item.content_type}"
            )

        file_path = Path(item.path)

        if not file_path.exists():
            raise FileNotFoundError(f"File no longer exists: {file_path}")

        try:
            # Read with UTF-8 encoding and error handling
            return file_path.read_text(encoding="utf-8", errors="ignore")
        except PermissionError as e:
            raise PermissionError(f"Permission denied reading file: {file_path}") from e
        except Exception as e:
            raise ValueError(f"Error reading file {file_path}: {e}") from e

    async def watch_changes(
        self, config: FileSystemSourceConfig, callback: Callable[[list[ContentItem]], None]
    ) -> SourceWatcher:
        """Set up file system change watching.

        Args:
            config: File system source configuration
            callback: Function to call when changes are detected

        Returns:
            File system watcher instance
        """
        if not config.get("enable_change_watching", False):
            raise NotImplementedError("Change watching is disabled in configuration")

        root_path_str = config.get("root_path", ".")
        root_path = Path(root_path_str).resolve()

        watcher = FileSystemSourceWatcher(
            source_id=self.source_id, callback=callback, root_path=root_path, config=config
        )

        self._watchers.append(watcher)
        return watcher

    async def validate_source(self, config: FileSystemSourceConfig) -> bool:
        """Validate file system source configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Call parent validation first
            if not await super().validate_source(config):
                return False

            # Check root path
            root_path_str = config.get("root_path")
            if not root_path_str:
                logger.warning("Missing root_path in file system source configuration")
                return False

            root_path = Path(root_path_str)
            if not root_path.exists():
                logger.warning("Root path does not exist: %s", root_path)
                return False

            if not root_path.is_dir():
                logger.warning("Root path is not a directory: %s", root_path)
                return False

            # Check if readable
            try:
                list(root_path.iterdir())
            except PermissionError:
                logger.warning("Permission denied accessing root path: %s", root_path)
                return False

        except Exception:
            logger.exception("Error validating file system source configuration")
            return False

        else:
            logger.info("File system source configuration is valid: %s", root_path)
            return True

    async def get_content_metadata(self, item: ContentItem) -> dict[str, Any]:
        """Get detailed metadata for a file.

        Args:
            item: Content item representing a file

        Returns:
            Dictionary with detailed file metadata
        """
        metadata = await super().get_content_metadata(item)

        try:
            file_path = Path(item.path)
            if file_path.exists():
                stat = file_path.stat()

                metadata.update({
                    "file_size_bytes": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "accessed_at": datetime.fromtimestamp(stat.st_atime).isoformat(),
                    "permissions": oct(stat.st_mode)[-3:],
                    "is_symlink": file_path.is_symlink(),
                    "absolute_path": str(file_path.resolve()),
                })

                # Add language detection
                if not item.language:
                    language = self._detect_language(file_path)
                    if language:
                        metadata["detected_language"] = language

        except Exception:
            logger.debug("Failed to get extended metadata for %s", item.path)

        return metadata

    async def _discover_files(self, root_path: Path, config: FileSystemSourceConfig) -> list[Path]:
        """Discover files using existing FileFilter logic.

        Args:
            root_path: Root directory to search
            config: Source configuration

        Returns:
            List of discovered file paths
        """
        # Import here to avoid circular dependencies
        from codeweaver.config import get_config
        from codeweaver.file_filter import FileFilter

        # Convert source config to FileFilter-compatible config
        app_config = get_config()

        # Update config with source-specific settings
        if config.get("use_gitignore") is not None:
            app_config.indexing.use_gitignore = config["use_gitignore"]

        if config.get("additional_ignore_patterns"):
            app_config.indexing.additional_ignore_patterns.extend(
                config["additional_ignore_patterns"]
            )

        if config.get("max_file_size_mb"):
            app_config.chunking.max_file_size_mb = config["max_file_size_mb"]

        # Create file filter and discover files
        file_filter = FileFilter(app_config, root_path)

        # Use patterns from config, with fallback to default
        patterns = config.get("patterns") or config.get("file_extensions")

        files = file_filter.find_files(patterns)

        logger.info("Discovered %d files with file filter", len(files))
        return files

    async def _file_to_content_item(self, file_path: Path, root_path: Path) -> ContentItem | None:
        """Convert a file path to a ContentItem.

        Args:
            file_path: Path to the file
            root_path: Root directory for relative path calculation

        Returns:
            ContentItem or None if file cannot be processed
        """
        try:
            stat = file_path.stat()

            # Calculate relative path
            try:
                relative_path = file_path.relative_to(root_path)
            except ValueError:
                relative_path = file_path

            # Detect language
            language = self._detect_language(file_path)

            return ContentItem(
                path=str(file_path),
                content_type="file",
                metadata={
                    "file_extension": file_path.suffix,
                    "relative_path": str(relative_path),
                    "parent_directory": str(file_path.parent),
                    "root_path": str(root_path),
                },
                last_modified=datetime.fromtimestamp(stat.st_mtime),
                size=stat.st_size,
                language=language,
                source_id=self.source_id,
            )

        except Exception:
            logger.debug("Failed to create ContentItem for %s", file_path)
            return None

    def _detect_language(self, file_path: Path) -> str | None:
        """Detect programming language from file extension."""
        # Import here to avoid circular dependencies
        from codeweaver.chunker import AstGrepChunker

        chunker = AstGrepChunker()
        suffix = file_path.suffix.lower()
        return chunker.SUPPORTED_LANGUAGES.get(suffix)
