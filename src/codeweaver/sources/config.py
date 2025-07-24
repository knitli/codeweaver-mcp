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

from dataclasses import dataclass, field
from typing import Any


logger = logging.getLogger(__name__)


@dataclass
class DataSourcesConfig:
    """Configuration for data source abstraction system."""

    # Global data source settings
    enabled: bool = True
    default_source_type: str = "filesystem"  # Backward compatibility
    max_concurrent_sources: int = 5

    # Content processing settings
    enable_content_deduplication: bool = True
    content_cache_ttl_hours: int = 24
    enable_metadata_extraction: bool = True

    # Source-specific configurations
    sources: list[dict[str, Any]] = field(default_factory=list)

    def add_source_config(
        self,
        source_type: str,
        config: dict[str, Any],
        enabled: bool = True,
        priority: int = 1,
        source_id: str | None = None,
    ) -> None:
        """Add a data source configuration.

        Args:
            source_type: Type of data source (filesystem, git, etc.)
            config: Source-specific configuration
            enabled: Whether the source is enabled
            priority: Priority for source ordering (lower = higher priority)
            source_id: Optional unique identifier for the source
        """
        source_config = {
            "type": source_type,
            "enabled": enabled,
            "priority": priority,
            "config": config,
        }

        if source_id:
            source_config["source_id"] = source_id

        self.sources.append(source_config)

    def get_source_configs_by_type(self, source_type: str) -> list[dict[str, Any]]:
        """Get all source configurations of a specific type.

        Args:
            source_type: Type of data source to filter by

        Returns:
            List of matching source configurations
        """
        return [source for source in self.sources if source.get("type") == source_type]

    def get_enabled_source_configs(self) -> list[dict[str, Any]]:
        """Get all enabled source configurations, sorted by priority.

        Returns:
            List of enabled source configurations, sorted by priority
        """
        enabled_sources = [source for source in self.sources if source.get("enabled", True)]

        # Sort by priority (lower priority number = higher precedence)
        return sorted(enabled_sources, key=lambda x: x.get("priority", 999))

    def migrate_from_legacy_config(self, legacy_config: Any) -> None:
        """Migrate from legacy file system only configuration.

        Args:
            legacy_config: Legacy CodeWeaverConfig object
        """
        # Check if we already have data sources configured
        if self.sources:
            logger.info("Data sources already configured, skipping legacy migration")
            return

        # Create file system source from legacy configuration
        filesystem_config = {
            "root_path": ".",  # Default to current directory
            "use_gitignore": getattr(legacy_config.indexing, "use_gitignore", True),
            "additional_ignore_patterns": getattr(
                legacy_config.indexing, "additional_ignore_patterns", []
            ),
            "max_file_size_mb": getattr(legacy_config.chunking, "max_file_size_mb", 1),
            "batch_size": getattr(legacy_config.indexing, "batch_size", 8),
            "enable_change_watching": getattr(legacy_config.indexing, "enable_auto_reindex", False),
            "change_check_interval_seconds": getattr(
                legacy_config.indexing, "watch_debounce_seconds", 2.0
            )
            * 10,  # Convert to seconds and add buffer
        }

        self.add_source_config(
            source_type="filesystem",
            config=filesystem_config,
            enabled=True,
            priority=1,
            source_id="default_filesystem",
        )

        logger.info("Migrated legacy configuration to file system data source")

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

            # Check for duplicate source IDs
            source_id = source.get("source_id")
            if source_id:
                if source_id in source_ids:
                    errors.append(f"Source {i}: duplicate source_id '{source_id}'")
                source_ids.add(source_id)

            # Validate priority
            priority = source.get("priority", 1)
            if not isinstance(priority, int) or priority < 1:
                errors.append(f"Source {i}: priority must be a positive integer")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary representation."""
        return {
            "enabled": self.enabled,
            "default_source_type": self.default_source_type,
            "max_concurrent_sources": self.max_concurrent_sources,
            "enable_content_deduplication": self.enable_content_deduplication,
            "content_cache_ttl_hours": self.content_cache_ttl_hours,
            "enable_metadata_extraction": self.enable_metadata_extraction,
            "sources": self.sources,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataSourcesConfig":
        """Create configuration from dictionary representation."""
        config = cls()

        # Update fields from dictionary
        for field_name, value in data.items():
            if hasattr(config, field_name):
                setattr(config, field_name, value)

        return config


def extend_config_with_data_sources(config_class: type) -> type:
    """Extend a configuration class with data sources support.

    This function extends the existing CodeWeaverConfig class to include
    data source configuration while maintaining backward compatibility.

    Args:
        config_class: The configuration class to extend

    Returns:
        Extended configuration class with data sources support
    """
    # Add data_sources field to the config class
    if not hasattr(config_class, "data_sources"):
        # Dynamically add the field (this is a bit hacky but maintains compatibility)
        original_init = config_class.__init__

        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if not hasattr(self, "data_sources"):
                self.data_sources = DataSourcesConfig()

        config_class.__init__ = new_init
        config_class.data_sources = field(default_factory=DataSourcesConfig)

    # Extend the merge_from_dict method
    original_merge = getattr(config_class, "merge_from_dict", None)

    def enhanced_merge_from_dict(self, config_dict: dict[str, Any]) -> None:
        # Handle data_sources section specially
        if "data_sources" in config_dict:
            data_sources_dict = config_dict["data_sources"]
            if isinstance(data_sources_dict, dict):
                self.data_sources = DataSourcesConfig.from_dict(data_sources_dict)
            config_dict = {k: v for k, v in config_dict.items() if k != "data_sources"}

        # Call original merge method for other sections
        if original_merge:
            original_merge(self, config_dict)
        else:
            # Fallback if no original method exists
            for section_name, section_data in config_dict.items():
                if hasattr(self, section_name) and isinstance(section_data, dict):
                    section = getattr(self, section_name)
                    for key, value in section_data.items():
                        if hasattr(section, key):
                            setattr(section, key, value)

    config_class.merge_from_dict = enhanced_merge_from_dict

    # Add convenience methods
    def get_data_sources_config(self) -> DataSourcesConfig:
        """Get the data sources configuration."""
        if not hasattr(self, "data_sources"):
            self.data_sources = DataSourcesConfig()
        return self.data_sources

    def ensure_data_sources_initialized(self) -> None:
        """Ensure data sources are initialized with legacy migration."""
        if not hasattr(self, "data_sources"):
            self.data_sources = DataSourcesConfig()

        # Migrate from legacy configuration if no sources configured
        if not self.data_sources.sources:
            self.data_sources.migrate_from_legacy_config(self)

    config_class.get_data_sources_config = get_data_sources_config
    config_class.ensure_data_sources_initialized = ensure_data_sources_initialized

    return config_class


def get_example_data_sources_config() -> str:
    """Get an example TOML configuration for data sources."""
    return """
# Data Sources Configuration
[data_sources]
enabled = true
default_source_type = "filesystem"
max_concurrent_sources = 5
enable_content_deduplication = true
content_cache_ttl_hours = 24
enable_metadata_extraction = true

# File System Source (default, maintains backward compatibility)
[[data_sources.sources]]
type = "filesystem"
enabled = true
priority = 1
source_id = "main_codebase"

[data_sources.sources.config]
root_path = "."
use_gitignore = true
additional_ignore_patterns = ["node_modules", ".git", ".venv", "__pycache__"]
max_file_size_mb = 1
batch_size = 8
enable_change_watching = false
change_check_interval_seconds = 60
patterns = ["**/*.py", "**/*.js", "**/*.ts", "**/*.go", "**/*.rs"]

# Git Repository Source (placeholder implementation)
[[data_sources.sources]]
type = "git"
enabled = false
priority = 2
source_id = "external_repo"

[data_sources.sources.config]
repository_url = "https://github.com/user/repo.git"
branch = "main"
local_clone_path = "/tmp/codeweaver_repos/external_repo"
auto_pull = true
pull_interval_minutes = 60
track_file_history = false

# Database Source (placeholder implementation)
[[data_sources.sources]]
type = "database"
enabled = false
priority = 3
source_id = "main_database"

[data_sources.sources.config]
database_type = "postgresql"
connection_string = "postgresql://user:password@localhost:5432/dbname"
include_tables = ["users", "products", "orders"]
include_views = ["user_summary"]
include_procedures = ["calculate_metrics"]
max_record_length = 10000
sample_size = 100

# API Source (placeholder implementation)
[[data_sources.sources]]
type = "api"
enabled = false
priority = 4
source_id = "rest_api"

[data_sources.sources.config]
api_type = "rest"
base_url = "https://api.example.com"
endpoints = ["/docs", "/schema", "/openapi.json"]
auth_type = "bearer"
bearer_token = "your-api-token"
schema_discovery = true
include_documentation = true

# Web Crawler Source (placeholder implementation)
[[data_sources.sources]]
type = "web"
enabled = false
priority = 5
source_id = "documentation_site"

[data_sources.sources.config]
start_urls = ["https://docs.example.com"]
allowed_domains = ["docs.example.com", "api.example.com"]
max_depth = 3
max_pages = 1000
delay_between_requests = 1.0
respect_robots_txt = true
extract_text_only = true
include_code_blocks = true
"""
