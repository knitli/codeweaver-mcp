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
from typing import Any

from codeweaver._types import ContentItem
from codeweaver.factories.source_registry import SourceRegistry
from codeweaver.sources.base import DataSource, SourceConfig


logger = logging.getLogger(__name__)


class DataSourceManager:
    """Manages multiple data sources and provides unified content discovery.

    This class orchestrates content discovery across multiple data sources
    while maintaining backward compatibility with the existing file system
    based indexing approach.
    """

    def __init__(self, sources: list[SourceConfig]):
        """Initialize the data source manager.

        Args:
            sources: List of data source configurations
        """
        self.sources = sources
        self.source_registry = SourceRegistry()
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
                config_obj = source_config["config"]

                # Create source configuration object
                source_config_obj = SourceConfig(config_obj)

                # Validate source type is available
                if not self.source_registry.has_component(source_type):
                    logger.warning("Unknown source type: %s", source_type)
                    continue

                # Create and validate the source
                source = self.source_registry.create_source(source_type, source_config_obj)

                # Validate source configuration if the source supports it
                if (
                    (hasattr(source, 'validate_source')
                    and await source.validate_source(config_obj))
                    or not hasattr(source, 'validate_source')
                ):
                    self._active_sources.append(source)
                    logger.info(
                        "Initialized data source: %s (%s)", source.source_id, source_type
                    )
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

                config_obj = source_config["config"]
                content = await source.discover_content(config_obj)
                all_content.extend(content)

                logger.info(
                    "Discovered %d content items from source: %s", len(content), source.source_id
                )

            except Exception:
                logger.exception("Error discovering content from source %s", source.source_id)

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
        if source := self._find_source_for_item(item):
            return await source.read_content(item)
        raise ValueError(f"No source available for content item: {item}")

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

                config_obj = source_config["config"]
                if not config_obj.get("enable_change_watching", False):
                    continue

                watcher = await source.watch_changes(config_obj, callback)
                await watcher.start()
                self._watchers.append(watcher)

                logger.info("Started change watching for source: %s", source.source_id)

            except NotImplementedError:
                logger.debug("Change watching not supported by source: %s", source.source_id)
            except Exception:
                logger.exception("Failed to setup change watching for source %s", source.source_id)

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
        return next(
            (
                source_config
                for source_config in self.config.sources
                if source_config.get("source_id") == source_id
            ),
            None,
        )

    def _find_source_for_item(self, item: ContentItem) -> DataSource | None:
        """Find the source that can handle a specific content item."""
        return next(
            (source for source in self._active_sources if source.source_id == item.source_id), None
        )

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
