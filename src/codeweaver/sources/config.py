# sourcery skip: name-type-suffix
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Configuration extensions for data source abstraction.

Extends the existing CodeWeaver configuration system to support
multiple data sources with proper validation and migration.
"""

import logging

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from codeweaver.types import SourceProvider


logger = logging.getLogger(__name__)


class DataSourcesConfig(BaseModel):
    """Configuration for data source abstraction system."""

    model_config = ConfigDict(extra="allow", validate_assignment=True, use_enum_values=True)

    # Global data source settings
    enabled: bool = Field(True, description="Enable sources system")
    default_source_type: SourceProvider = Field(
        SourceProvider.FILESYSTEM, description="Default source provider"
    )
    max_concurrent_sources: Annotated[
        int, Field(5, ge=1, le=20, description="Maximum concurrent sources")
    ]

    # Content processing settings
    enable_content_deduplication: bool = Field(True, description="Enable content deduplication")
    content_cache_ttl_hours: Annotated[
        int, Field(24, ge=1, le=168, description="Content cache TTL in hours")
    ]
    enable_metadata_extraction: bool = Field(True, description="Enable metadata extraction")

    # Source-specific configurations
    sources: list[dict[str, Any]] = Field(default_factory=list, description="Source configurations")

    def add_source_config(
        self,
        source_type: SourceProvider,
        config: dict[str, Any],
        *,
        enabled: bool = True,
        priority: int = 1,
        source_id: str | None = None,
    ) -> None:
        """Add a data source configuration.

        Args:
            source_type: Type of data source (SourceProvider enum)
            config: Source-specific configuration
            enabled: Whether the source is enabled
            priority: Priority for source ordering (lower = higher priority)
            source_id: Optional unique identifier for the source
        """
        source_config = {
            "type": source_type.value,
            "enabled": enabled,
            "priority": priority,
            "config": config,
        }

        if source_id:
            source_config["source_id"] = source_id

        self.sources.append(source_config)

    def get_source_configs_by_type(self, source_type: SourceProvider) -> list[dict[str, Any]]:
        """Get all source configurations of a specific type.

        Args:
            source_type: Type of data source to filter by

        Returns:
            List of matching source configurations
        """
        return [source for source in self.sources if source.get("type") == source_type.value]

    def get_enabled_source_configs(self) -> list[dict[str, Any]]:
        """Get all enabled source configurations, sorted by priority.

        Returns:
            List of enabled source configurations, sorted by priority
        """
        enabled_sources = [source for source in self.sources if source.get("enabled", True)]

        # Sort by priority (lower priority number = higher precedence)
        return sorted(enabled_sources, key=lambda x: x.get("priority", 999))

    # Legacy migration method removed as per improvement plan

    def validate_configurations(self) -> list[str]:
        """Validate all source configurations.

        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []

        if not self.sources:
            errors.append("No data sources configured")

        source_ids = set()
        for i, source in enumerate(self.sources):
            # Check required fields
            if "type" not in source:
                errors.append(f"Source {i}: missing 'type' field")
                continue

            if "config" not in source:
                errors.append(f"Source {i}: missing 'config' field")
                continue

            if source_id := source.get("source_id"):
                if source_id in source_ids:
                    errors.append(f"Source {i}: duplicate source_id '{source_id}'")
                source_ids.add(source_id)

            # Validate priority
            priority = source.get("priority", 1)
            if not isinstance(priority, int) or priority < 1:
                errors.append(f"Source {i}: priority must be a positive integer")

        return errors
