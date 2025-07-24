# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Factory system for creating and managing data source instances.

Provides centralized creation and configuration of different data source
types with validation and dependency management.
"""

import logging

from typing import Any

from codeweaver.sources.base import DataSource, SourceConfig, get_source_registry


logger = logging.getLogger(__name__)


class SourceFactory:
    """Factory for creating and managing data source instances.

    Handles the creation of different data source types with proper
    configuration validation and dependency injection.
    """

    def __init__(self):
        """Initialize the source factory."""
        self._registry = get_source_registry()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure the factory is initialized with all available sources."""
        if self._initialized:
            return

        # Register all available source implementations
        self._register_builtin_sources()
        self._initialized = True

    def _register_builtin_sources(self) -> None:
        """Register all built-in source implementations."""
        try:
            # File system source (always available)
            from codeweaver.sources.filesystem import FileSystemSource

            self._registry.register("filesystem", FileSystemSource)

            # Git repository source (placeholder)
            from codeweaver.sources.git import GitRepositorySource

            self._registry.register("git", GitRepositorySource)

            # Database source (placeholder)
            from codeweaver.sources.database import DatabaseSource

            self._registry.register("database", DatabaseSource)

            # API source (placeholder)
            from codeweaver.sources.api import APISource

            self._registry.register("api", APISource)

            # Web crawler source (placeholder)
            from codeweaver.sources.web import WebCrawlerSource

            self._registry.register("web", WebCrawlerSource)

            logger.info(
                "Registered %d built-in data sources", len(self._registry.list_available_sources())
            )

        except ImportError as e:
            logger.warning("Failed to import some data sources: %s", e)

    def create_source(self, source_type: str, config: SourceConfig) -> DataSource:
        """Create a data source instance of the specified type.

        Args:
            source_type: Type of data source to create
            config: Configuration for the data source

        Returns:
            Configured data source instance

        Raises:
            ValueError: If source type is not supported or config is invalid
            ImportError: If required dependencies are not available
        """
        self._ensure_initialized()

        # Get source class from registry
        source_class = self._registry.get_source_class(source_type)
        if not source_class:
            available_types = self._registry.list_available_sources()
            raise ValueError(
                f"Unsupported source type: {source_type}. "
                f"Available types: {', '.join(available_types)}"
            )

        # Extract source_id from config if provided
        source_id = config.get("source_id")

        try:
            # Create source instance
            source = source_class(source_id=source_id) if source_id else source_class()

            logger.info("Created %s data source: %s", source_type, source.source_id)
            return source

        except Exception:
            logger.exception("Failed to create %s data source", source_type)
            raise

    def create_multiple_sources(self, source_configs: list[dict[str, Any]]) -> list[DataSource]:
        """Create multiple data source instances from configurations.

        Args:
            source_configs: List of source configuration dictionaries

        Returns:
            List of configured data source instances
        """
        sources = []

        for config in source_configs:
            try:
                source_type = config.get("type")
                if not source_type:
                    logger.warning("Skipping source config without type: %s", config)
                    continue

                # Extract the actual config (usually nested under 'config' key)
                source_config = config.get("config", {})

                # Add top-level fields to config
                for field in ["enabled", "priority", "source_id"]:
                    if field in config:
                        source_config[field] = config[field]

                if not source_config.get("enabled", True):
                    logger.info("Skipping disabled source: %s", source_type)
                    continue

                source = self.create_source(source_type, source_config)
                sources.append(source)

            except Exception:
                logger.exception("Failed to create source from config %s.", config)
                continue

        logger.info("Created %d data sources", len(sources))
        return sources

    async def validate_source_config(self, source_type: str, config: SourceConfig) -> bool:
        """Validate a source configuration without creating the source.

        Args:
            source_type: Type of data source
            config: Configuration to validate

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            source = self.create_source(source_type, config)
            is_valid = await source.validate_source(config)

            # Clean up the source
            if hasattr(source, "cleanup"):
                await source.cleanup()

            return is_valid

        except Exception:
            logger.exception("Error validating %s source config", source_type)
            return False

    def list_available_sources(self) -> dict[str, dict[str, Any]]:
        """Get information about all available data source types.

        Returns:
            Dictionary mapping source types to their information
        """
        self._ensure_initialized()

        source_info = {}

        for source_type in self._registry.list_available_sources():
            capabilities = self._registry.get_source_capabilities(source_type)

            source_info[source_type] = {
                "capabilities": [cap.value for cap in capabilities] if capabilities else [],
                "implemented": self._is_source_implemented(source_type),
                "description": self._get_source_description(source_type),
            }

        return source_info

    def _is_source_implemented(self, source_type: str) -> bool:
        """Check if a source type is fully implemented or a placeholder.

        Args:
            source_type: Type of data source

        Returns:
            True if fully implemented, False if placeholder
        """
        # File system is the only fully implemented source currently
        return source_type == "filesystem"

    def _get_source_description(self, source_type: str) -> str:
        """Get a description for a source type.

        Args:
            source_type: Type of data source

        Returns:
            Human-readable description of the source
        """
        descriptions = {
            "filesystem": "Local file system with gitignore support and change watching",
            "git": "Git repositories with branch/commit support (placeholder)",
            "database": "SQL and NoSQL databases for schema and data indexing (placeholder)",
            "api": "REST and GraphQL APIs for documentation and schema indexing (placeholder)",
            "web": "Web crawler for documentation sites and technical content (placeholder)",
        }

        return descriptions.get(source_type, f"Data source of type {source_type}")

    def get_source_registry(self):
        """Get the underlying source registry.

        Returns:
            Source registry instance for advanced operations
        """
        self._ensure_initialized()
        return self._registry


# Global factory instance
_source_factory = SourceFactory()


def get_source_factory() -> SourceFactory:
    """Get the global source factory instance.

    Returns:
        Global source factory for creating data sources
    """
    return _source_factory
