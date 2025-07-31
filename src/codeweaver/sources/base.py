"""
Base protocols and data structures for CodeWeaver data source abstraction.

Defines the core interfaces that all data sources must implement to provide
universal content discovery, reading, and change watching capabilities.
"""

import logging

from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from codeweaver.types import ContentItem, SourceCapabilities, SourceCapability, SourceProvider
from codeweaver.utils.decorators import require_implementation


logger = logging.getLogger(__name__)


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
                logger.exception("Error in change notification callback for source %s")


class SourceConfig(BaseModel):
    """Base configuration for all sources."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)
    enabled: Annotated[bool, Field(True, description="Whether source is enabled")]
    priority: Annotated[int, Field(1, ge=1, le=100, description="Source priority")]
    source_id: Annotated[str | None, Field(None, description="Unique source identifier")]
    include_patterns: Annotated[list[str], Field(default_factory=list)]
    exclude_patterns: Annotated[list[str], Field(default_factory=list)]
    max_file_size_mb: Annotated[int, Field(1, ge=1, le=1000)]
    batch_size: Annotated[int, Field(8, ge=1, le=1000)]
    max_concurrent_requests: Annotated[int, Field(10, ge=1, le=100)]
    request_timeout_seconds: Annotated[int, Field(30, ge=1, le=300)]
    enable_change_watching: bool = Field(False, description="Enable change watching")
    change_check_interval_seconds: Annotated[int, Field(60, ge=1, le=3600)]
    enable_content_deduplication: bool = Field(True, description="Enable content deduplication")
    enable_metadata_extraction: bool = Field(False, description="Enable metadata extraction")
    supported_languages: list[str] = Field(
        default_factory=list, description="Supported programming languages"
    )


@runtime_checkable
class DataSource(Protocol):
    """Universal data source protocol for content discovery and processing.

    All data source implementations must support this interface to provide
    universal content discovery, reading, and change watching capabilities.
    """

    def get_capabilities(self) -> SourceCapabilities:
        """Get the capabilities supported by this data source.

        Returns:
            Capabilities model with detailed source capabilities
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

    @require_implementation("discover_content", "read_content")
    def __init__(self, source_type: SourceProvider, source_id: str | None = None):
        """Initialize the abstract data source.

        Args:
            source_type: Type identifier for this source (e.g., SourceProvider.FILESYSTEM)
            source_id: Unique identifier for this source instance
        """
        self.source_type = source_type
        self.source_id = source_id or f"{source_type.value}_{id(self)}"
        self._watchers: list[SourceWatcher] = []

    @abstractmethod
    def get_capabilities(self) -> SourceCapabilities:
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
            "discovered_at": datetime.now(UTC).isoformat(),
        }
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
            if item.size and item.size > max_size_bytes:
                logger.debug("Skipping large file: %s (%.1fMB)", item.path, item.size / 1024 / 1024)
                continue
            if exclude_patterns and any((pattern in item.path for pattern in exclude_patterns)):
                logger.debug("Excluded by pattern: %s", item.path)
                continue
            if include_patterns and all((pattern not in item.path for pattern in include_patterns)):
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

    def get_source_capabilities(self, source_type: SourceProvider) -> SourceCapabilities | None:
        """Get capabilities for a registered source type.

        Args:
            source_type: Type identifier for the source

        Returns:
            SourceCapabilities object if source is registered, None otherwise
        """
        if source_class := self._sources.get(source_type):
            try:
                instance = source_class()
            except Exception:
                logger.warning("Failed to get capabilities for source type: %s", source_type.value)
            else:
                return instance.get_capabilities()
        return None


_source_registry = SourceRegistry()


def get_source_registry() -> SourceRegistry:
    """Get the global source registry instance."""
    return _source_registry
