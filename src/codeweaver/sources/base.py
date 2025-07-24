# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Base protocols and data structures for CodeWeaver data source abstraction.

Defines the core interfaces that all data sources must implement to provide
universal content discovery, reading, and change watching capabilities.
"""

import hashlib
import logging

from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, TypedDict, runtime_checkable


logger = logging.getLogger(__name__)


class SourceCapability(Enum):
    """Capabilities supported by different data sources."""

    # Core capabilities
    CONTENT_DISCOVERY = "content_discovery"  # Can discover content items
    CONTENT_READING = "content_reading"  # Can read content from items
    CHANGE_WATCHING = "change_watching"  # Can watch for content changes

    # Advanced capabilities
    INCREMENTAL_SYNC = "incremental_sync"  # Supports incremental updates
    VERSION_HISTORY = "version_history"  # Provides version/commit history
    METADATA_EXTRACTION = "metadata_extraction"  # Rich metadata extraction
    REAL_TIME_UPDATES = "real_time_updates"  # Real-time change notifications
    BATCH_PROCESSING = "batch_processing"  # Efficient batch operations
    CONTENT_DEDUPLICATION = "content_deduplication"  # Built-in deduplication
    RATE_LIMITING = "rate_limiting"  # Built-in rate limiting
    AUTHENTICATION = "authentication"  # Supports authentication
    PAGINATION = "pagination"  # Supports paginated discovery


class ContentItem:
    """Universal content item representing discoverable content from any source."""

    def __init__(
        self,
        path: str,
        content_type: str,
        metadata: dict[str, Any] | None = None,
        last_modified: datetime | None = None,
        size: int | None = None,
        language: str | None = None,
        source_id: str | None = None,
        version: str | None = None,
        checksum: str | None = None,
    ):
        """Initialize a content item.

        Args:
            path: Universal identifier for the content (URL, file path, etc.)
            content_type: Type of content ('file', 'url', 'database', 'api', 'git')
            metadata: Source-specific metadata
            last_modified: Last modification timestamp
            size: Content size in bytes
            language: Detected programming language
            source_id: Identifier of the source that discovered this content
            version: Version identifier (commit hash, revision, etc.)
            checksum: Content checksum for change detection
        """
        self.path = path
        self.content_type = content_type
        self.metadata = metadata or {}
        self.last_modified = last_modified
        self.size = size
        self.language = language
        self.source_id = source_id
        self.version = version
        self.checksum = checksum

        # Generate unique identifier
        self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate a unique identifier for this content item."""
        # Combine path, content_type, and source_id for uniqueness
        identifier_parts = [self.path, self.content_type]
        if self.source_id:
            identifier_parts.append(self.source_id)

        identifier_string = "|".join(identifier_parts)
        return hashlib.sha256(identifier_string.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert content item to dictionary representation."""
        return {
            "id": self.id,
            "path": self.path,
            "content_type": self.content_type,
            "metadata": self.metadata,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "size": self.size,
            "language": self.language,
            "source_id": self.source_id,
            "version": self.version,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContentItem":
        """Create content item from dictionary representation."""
        last_modified = None
        if data.get("last_modified"):
            last_modified = datetime.fromisoformat(data["last_modified"])

        return cls(
            path=data["path"],
            content_type=data["content_type"],
            metadata=data.get("metadata"),
            last_modified=last_modified,
            size=data.get("size"),
            language=data.get("language"),
            source_id=data.get("source_id"),
            version=data.get("version"),
            checksum=data.get("checksum"),
        )

    def __str__(self) -> str:
        """String representation of the content item."""
        return f"ContentItem(id={self.id}, type={self.content_type}, path={self.path})"

    def __repr__(self) -> str:
        """Detailed representation of the content item."""
        return (
            f"ContentItem(path={self.path!r}, content_type={self.content_type!r}, "
            f"size={self.size}, language={self.language!r})"
        )


class SourceWatcher:
    """Handle for watching changes in a data source."""

    def __init__(self, source_id: str, callback: Callable[[list[ContentItem]], None]):
        """Initialize a source watcher.

        Args:
            source_id: Identifier of the source being watched
            callback: Function to call when changes are detected
        """
        self.source_id = source_id
        self.callback = callback
        self.is_active = False

    async def start(self) -> bool:
        """Start watching for changes.

        Returns:
            True if watching started successfully, False otherwise
        """
        self.is_active = True
        logger.info("Started watching source: %s", self.source_id)
        return True

    async def stop(self) -> bool:
        """Stop watching for changes.

        Returns:
            True if watching stopped successfully, False otherwise
        """
        self.is_active = False
        logger.info("Stopped watching source: %s", self.source_id)
        return True

    async def notify_changes(self, changed_items: list[ContentItem]) -> None:
        """Notify about detected changes."""
        if self.is_active and self.callback:
            try:
                await self.callback(changed_items)
            except Exception:
                logger.exception(
                    "Error in change notification callback for source %s", self.source_id
                )


class BaseSourceConfig(TypedDict, total=False):
    """Base configuration for all data sources."""

    # Core settings
    enabled: bool
    priority: int
    source_id: str

    # Content filtering
    include_patterns: list[str]
    exclude_patterns: list[str]
    max_file_size_mb: int

    # Performance settings
    batch_size: int
    max_concurrent_requests: int
    request_timeout_seconds: int

    # Change detection
    enable_change_watching: bool
    change_check_interval_seconds: int

    # Content processing
    enable_content_deduplication: bool
    enable_metadata_extraction: bool
    supported_languages: list[str]


# Type alias for source-specific configurations
SourceConfig = BaseSourceConfig


@runtime_checkable
class DataSource(Protocol):
    """Universal data source protocol for content discovery and processing.

    All data source implementations must support this interface to provide
    universal content discovery, reading, and change watching capabilities.
    """

    def get_capabilities(self) -> set[SourceCapability]:
        """Get the capabilities supported by this data source.

        Returns:
            Set of capabilities supported by this source
        """
        ...

    async def discover_content(self, config: SourceConfig) -> list[ContentItem]:
        """Discover all available content from this source.

        Args:
            config: Source-specific configuration

        Returns:
            List of discovered content items

        Raises:
            ValueError: If configuration is invalid
            BackendConnectionError: If source is unreachable
        """
        ...

    async def read_content(self, item: ContentItem) -> str:
        """Read content from a specific content item.

        Args:
            item: Content item to read

        Returns:
            Text content of the item

        Raises:
            FileNotFoundError: If content item no longer exists
            PermissionError: If access is denied
            ValueError: If content cannot be decoded as text
        """
        ...

    async def watch_changes(
        self, config: SourceConfig, callback: Callable[[list[ContentItem]], None]
    ) -> SourceWatcher:
        """Set up watching for content changes.

        Args:
            config: Source configuration
            callback: Function to call when changes are detected

        Returns:
            Watcher handle for managing the watch operation

        Raises:
            NotImplementedError: If change watching is not supported
        """
        ...

    async def validate_source(self, config: SourceConfig) -> bool:
        """Validate that the source is accessible and properly configured.

        Args:
            config: Source configuration to validate

        Returns:
            True if source is valid and accessible, False otherwise
        """
        ...

    async def get_content_metadata(self, item: ContentItem) -> dict[str, Any]:
        """Get detailed metadata for a content item.

        Args:
            item: Content item to get metadata for

        Returns:
            Dictionary with detailed metadata
        """
        ...


class AbstractDataSource(ABC):
    """Abstract base class for data source implementations.

    Provides common functionality and ensures consistent implementation
    of the DataSource protocol across different source types.
    """

    def __init__(self, source_type: str, source_id: str | None = None):
        """Initialize the abstract data source.

        Args:
            source_type: Type identifier for this source (e.g., 'filesystem', 'git')
            source_id: Unique identifier for this source instance
        """
        self.source_type = source_type
        self.source_id = source_id or f"{source_type}_{id(self)}"
        self._watchers: list[SourceWatcher] = []

    @abstractmethod
    def get_capabilities(self) -> set[SourceCapability]:
        """Get capabilities supported by this source implementation."""

    @abstractmethod
    async def discover_content(self, config: SourceConfig) -> list[ContentItem]:
        """Discover content from this source."""

    @abstractmethod
    async def read_content(self, item: ContentItem) -> str:
        """Read content from an item."""

    async def watch_changes(
        self, config: SourceConfig, callback: Callable[[list[ContentItem]], None]
    ) -> SourceWatcher:
        """Default implementation for change watching."""
        if SourceCapability.CHANGE_WATCHING not in self.get_capabilities():
            raise NotImplementedError(f"Change watching not supported by {self.source_type}")

        watcher = SourceWatcher(self.source_id, callback)
        self._watchers.append(watcher)
        return watcher

    async def validate_source(self, config: SourceConfig) -> bool:
        """Default validation implementation."""
        # Basic validation - check if config is properly formatted
        try:
            required_fields = ["enabled"]
            for field in required_fields:
                if field not in config:
                    logger.warning("Missing required field '%s' in source config", field)
                    return False
        except Exception:
            logger.exception("Error validating source configuration")
            return False
        return True

    async def get_content_metadata(self, item: ContentItem) -> dict[str, Any]:
        """Default metadata extraction implementation."""
        metadata = {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "discovered_at": datetime.now(datetime.UTC).isoformat(),
        }

        # Add any existing metadata from the item
        metadata |= item.metadata
        return metadata

    def _apply_content_filters(
        self, items: list[ContentItem], config: SourceConfig
    ) -> list[ContentItem]:
        """Apply include/exclude patterns and size filters to content items."""
        if not items:
            return items

        filtered_items = []
        include_patterns = config.get("include_patterns", [])
        exclude_patterns = config.get("exclude_patterns", [])
        max_size_mb = config.get("max_file_size_mb", 100)
        max_size_bytes = max_size_mb * 1024 * 1024

        for item in items:
            # Check size limits
            if item.size and item.size > max_size_bytes:
                logger.debug("Skipping large file: %s (%.1fMB)", item.path, item.size / 1024 / 1024)
                continue

            # Check exclude patterns
            if exclude_patterns and any(pattern in item.path for pattern in exclude_patterns):
                logger.debug("Excluded by pattern: %s", item.path)
                continue

            # Check include patterns (if specified)
            if include_patterns and all(pattern not in item.path for pattern in include_patterns):
                logger.debug("Not included by pattern: %s", item.path)
                continue

            filtered_items.append(item)

        logger.info(
            "Content filtering: %d items -> %d items (source: %s)",
            len(items),
            len(filtered_items),
            self.source_id,
        )
        return filtered_items

    async def cleanup(self) -> None:
        """Clean up resources and stop all watchers."""
        for watcher in self._watchers:
            await watcher.stop()
        self._watchers.clear()


class SourceRegistry:
    """Registry for managing available data source implementations."""

    def __init__(self):
        """Initialize the source registry."""
        self._sources: dict[str, type[DataSource]] = {}

    def register(self, source_type: str, source_class: type[DataSource]) -> None:
        """Register a data source implementation.

        Args:
            source_type: Type identifier for the source (e.g., 'filesystem', 'git')
            source_class: Data source implementation class
        """
        if not issubclass(source_class, DataSource):
            raise TypeError(f"Source class must implement DataSource protocol: {source_class}")

        self._sources[source_type] = source_class
        logger.info("Registered data source: %s -> %s", source_type, source_class.__name__)

    def get_source_class(self, source_type: str) -> type[DataSource] | None:
        """Get a registered source class by type.

        Args:
            source_type: Type identifier for the source

        Returns:
            Source class if registered, None otherwise
        """
        return self._sources.get(source_type)

    def list_available_sources(self) -> list[str]:
        """Get list of all registered source types.

        Returns:
            List of registered source type identifiers
        """
        return list(self._sources.keys())

    def get_source_capabilities(self, source_type: str) -> set[SourceCapability] | None:
        """Get capabilities for a registered source type.

        Args:
            source_type: Type identifier for the source

        Returns:
            Set of capabilities if source is registered, None otherwise
        """
        if source_class := self._sources.get(source_type):
            # Create a temporary instance to get capabilities
            try:
                instance = source_class()
                return instance.get_capabilities()
            except Exception:
                logger.warning("Failed to get capabilities for source type: %s", source_type)
        return None


# Global source registry instance
_source_registry = SourceRegistry()


def get_source_registry() -> SourceRegistry:
    """Get the global source registry instance."""
    return _source_registry
