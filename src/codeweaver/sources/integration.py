# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Integration module for data source abstraction with existing CodeWeaver server.

Provides backward-compatible integration of the new data source system
with the existing server implementation, enabling gradual migration.
"""

import logging

from collections.abc import Callable
from pathlib import Path
from typing import Any

from codeweaver.sources.base import ContentItem, DataSource
from codeweaver.sources.config import DataSourcesConfig, extend_config_with_data_sources
from codeweaver.sources.factory import get_source_factory


logger = logging.getLogger(__name__)


class DataSourceManager:
    """Manages multiple data sources and provides unified content discovery.

    This class orchestrates content discovery across multiple data sources
    while maintaining backward compatibility with the existing file system
    based indexing approach.
    """

    def __init__(self, config: DataSourcesConfig):
        """Initialize the data source manager.

        Args:
            config: Data sources configuration
        """
        self.config = config
        self.source_factory = get_source_factory()
        self._active_sources: list[DataSource] = []
        self._watchers: list[Any] = []

    async def initialize_sources(self) -> None:
        """Initialize all configured data sources."""
        if not self.config.enabled:
            logger.info("Data sources are disabled in configuration")
            return

        enabled_configs = self.config.get_enabled_source_configs()

        if not enabled_configs:
            logger.warning("No enabled data sources configured")
            return

        for source_config in enabled_configs:
            try:
                source_type = source_config["type"]
                config = source_config["config"]

                # Create and validate the source
                source = self.source_factory.create_source(source_type, config)

                if await source.validate_source(config):
                    self._active_sources.append(source)
                    logger.info("Initialized data source: %s (%s)", source.source_id, source_type)
                else:
                    logger.warning("Validation failed for data source: %s", source_type)

            except Exception:
                logger.exception("Failed to initialize data source %s.", source_config)

        logger.info("Initialized %d data sources", len(self._active_sources))

    async def discover_all_content(self) -> list[ContentItem]:
        """Discover content from all active data sources.

        Returns:
            List of all discovered content items from all sources
        """
        all_content = []

        for source in self._active_sources:
            try:
                # Get the source configuration
                source_config = self._get_source_config(source.source_id)
                if not source_config:
                    logger.warning("No configuration found for source: %s", source.source_id)
                    continue

                content = await source.discover_content(source_config["config"])
                all_content.extend(content)

                logger.info(
                    "Discovered %d content items from source: %s", len(content), source.source_id
                )

            except Exception as e:
                logger.exception(
                    "Error discovering content from source %s: %s", source.source_id, e
                )

        # Apply deduplication if enabled
        if self.config.enable_content_deduplication:
            all_content = self._deduplicate_content(all_content)

        logger.info("Total discovered content items: %d", len(all_content))
        return all_content

    async def read_content_item(self, item: ContentItem) -> str:
        """Read content from a specific content item.

        Args:
            item: Content item to read

        Returns:
            Text content of the item

        Raises:
            ValueError: If no source can handle the content item
        """
        # Find the source that can handle this content item
        source = self._find_source_for_item(item)
        if not source:
            raise ValueError(f"No source available for content item: {item}")

        return await source.read_content(item)

    async def setup_change_watching(self, callback: Callable[[list[ContentItem]], None]) -> None:
        """Set up change watching for all sources that support it.

        Args:
            callback: Function to call when changes are detected
        """
        for source in self._active_sources:
            try:
                source_config = self._get_source_config(source.source_id)
                if not source_config:
                    continue

                config = source_config["config"]
                if not config.get("enable_change_watching", False):
                    continue

                watcher = await source.watch_changes(config, callback)
                await watcher.start()
                self._watchers.append(watcher)

                logger.info("Started change watching for source: %s", source.source_id)

            except NotImplementedError:
                logger.debug("Change watching not supported by source: %s", source.source_id)
            except Exception as e:
                logger.exception(
                    "Failed to setup change watching for source %s: %s", source.source_id, e
                )

    async def cleanup(self) -> None:
        """Clean up all sources and watchers."""
        # Stop all watchers
        for watcher in self._watchers:
            try:
                await watcher.stop()
            except Exception:
                logger.exception("Error stopping watcher: %s", watcher)

        # Clean up sources
        for source in self._active_sources:
            try:
                if hasattr(source, "cleanup"):
                    await source.cleanup()
            except Exception:
                logger.exception("Error cleaning up source %s", source.source_id)

        self._watchers.clear()
        self._active_sources.clear()

    def _get_source_config(self, source_id: str) -> dict[str, Any] | None:
        """Get configuration for a specific source ID."""
        for source_config in self.config.sources:
            if source_config.get("source_id") == source_id:
                return source_config
        return None

    def _find_source_for_item(self, item: ContentItem) -> DataSource | None:
        """Find the source that can handle a specific content item."""
        for source in self._active_sources:
            if source.source_id == item.source_id:
                return source
        return None

    def _deduplicate_content(self, content_items: list[ContentItem]) -> list[ContentItem]:
        """Remove duplicate content items based on path and checksum."""
        seen_items = {}
        deduplicated = []

        for item in content_items:
            # Create a unique key based on path and content type
            key = f"{item.path}|{item.content_type}"

            if key not in seen_items:
                seen_items[key] = item
                deduplicated.append(item)
            else:
                # If we have a checksum, prefer the newer item
                existing_item = seen_items[key]
                if (
                    item.checksum
                    and existing_item.checksum
                    and item.last_modified
                    and existing_item.last_modified
                ) and item.last_modified > existing_item.last_modified:
                    # Replace with newer item
                    deduplicated.remove(existing_item)
                    deduplicated.append(item)
                    seen_items[key] = item

        removed_count = len(content_items) - len(deduplicated)
        if removed_count > 0:
            logger.info("Removed %d duplicate content items", removed_count)

        return deduplicated


class BackwardCompatibilityAdapter:
    """Adapter for backward compatibility with existing server implementation.

    This adapter allows the existing server code to use the new data source
    system while maintaining the same interface and behavior.
    """

    def __init__(self, data_source_manager: DataSourceManager, chunker):
        """Initialize the compatibility adapter.

        Args:
            data_source_manager: Data source manager instance
            chunker: Existing chunker instance for processing content
        """
        self.data_source_manager = data_source_manager
        self.chunker = chunker

    async def index_codebase_with_sources(
        self, root_path: str | None = None, patterns: list[str] | None = None
    ) -> dict[str, Any]:
        """Index codebase using the new data source system.

        This method provides the same interface as the original index_codebase
        method but uses the new data source system underneath.

        Args:
            root_path: Root path (for backward compatibility, may be ignored)
            patterns: File patterns (for backward compatibility, may be ignored)

        Returns:
            Dictionary with indexing results in the same format as original
        """
        # Discover content from all sources
        content_items = await self.data_source_manager.discover_all_content()

        # Convert content items to code chunks
        all_chunks = []
        processed_languages = set()

        for item in content_items:
            try:
                # Read content from the item
                content = await self.data_source_manager.read_content_item(item)

                # Convert to Path for chunker compatibility
                if item.content_type == "file":
                    file_path = Path(item.path)
                else:
                    # For non-file content, create a virtual path
                    file_path = Path(f"{item.content_type}_{item.id}.txt")

                # Chunk the content
                chunks = self.chunker.chunk_file(file_path, content)

                # Enhance chunks with source metadata
                for chunk in chunks:
                    chunk.metadata = chunk.metadata or {}
                    chunk.metadata.update({
                        "source_id": item.source_id,
                        "content_type": item.content_type,
                        "original_path": item.path,
                    })

                all_chunks.extend(chunks)

                if chunks and item.language:
                    processed_languages.add(item.language)

            except Exception as e:
                logger.warning("Error processing content item %s: %s", item.path, e)
                continue

        # Return results in the same format as original method
        return {
            "status": "success",
            "files_processed": len(content_items),
            "total_chunks": len(all_chunks),
            "languages_found": list(processed_languages),
            "data_sources_used": [
                source.source_id for source in self.data_source_manager._active_sources
            ],
            "source_statistics": self._get_source_statistics(content_items),
        }

    def _get_source_statistics(self, content_items: list[ContentItem]) -> dict[str, Any]:
        """Get statistics about content discovery by source."""
        stats = {}

        for item in content_items:
            source_id = item.source_id or "unknown"
            content_type = item.content_type

            if source_id not in stats:
                stats[source_id] = {"total_items": 0, "content_types": {}}

            stats[source_id]["total_items"] += 1

            if content_type not in stats[source_id]["content_types"]:
                stats[source_id]["content_types"][content_type] = 0

            stats[source_id]["content_types"][content_type] += 1

        return stats


def integrate_data_sources_with_config(config_class: type) -> type:
    """Integrate data source system with existing configuration.

    This function extends the existing configuration system to support
    data sources while maintaining backward compatibility.

    Args:
        config_class: Existing configuration class to extend

    Returns:
        Extended configuration class with data source support
    """
    # Extend the config class with data sources support
    extended_class = extend_config_with_data_sources(config_class)

    # Add initialization method for data source manager
    def create_data_source_manager(self) -> DataSourceManager:
        """Create a data source manager from this configuration."""
        self.ensure_data_sources_initialized()
        return DataSourceManager(self.get_data_sources_config())

    extended_class.create_data_source_manager = create_data_source_manager

    return extended_class


def create_backward_compatible_server_integration(server_class: type) -> type:
    """Create backward compatible integration for existing server class.

    This function extends the existing server class to optionally use
    the new data source system while maintaining full backward compatibility.

    Args:
        server_class: Existing server class to extend

    Returns:
        Extended server class with optional data source support
    """
    # Store original methods
    original_init = server_class.__init__
    original_index_codebase = getattr(server_class, "index_codebase", None)

    def enhanced_init(self, config=None):
        """Enhanced initialization with optional data source support."""
        # Call original initialization
        original_init(self, config)

        # Add data source manager (optional)
        self._data_source_manager = None
        self._backward_compatibility_adapter = None
        self._use_data_sources = getattr(self.config, "data_sources", None) is not None and getattr(
            self.config.data_sources, "enabled", False
        )

    async def enhanced_index_codebase(self, root_path: str, patterns: list[str] | None = None):
        """Enhanced index_codebase with optional data source support."""
        # Check if we should use the new data source system
        if self._use_data_sources and hasattr(self.config, "create_data_source_manager"):
            # Use new data source system
            if not self._data_source_manager:
                self._data_source_manager = self.config.create_data_source_manager()
                await self._data_source_manager.initialize_sources()

                self._backward_compatibility_adapter = BackwardCompatibilityAdapter(
                    self._data_source_manager, self.chunker
                )

            return await self._backward_compatibility_adapter.index_codebase_with_sources(
                root_path, patterns
            )
        # Use original implementation
        if original_index_codebase:
            return await original_index_codebase(self, root_path, patterns)
        raise NotImplementedError("Original index_codebase method not found")

    # Replace methods
    server_class.__init__ = enhanced_init
    if original_index_codebase:
        server_class.index_codebase = enhanced_index_codebase

    return server_class
