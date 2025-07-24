# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Data source abstraction system for CodeWeaver.

Provides universal content discovery and processing for multiple data sources
including file systems, git repositories, databases, APIs, and web crawlers.

## Quick Start

```python
from codeweaver.sources import SourceFactory, FileSystemSourceConfig

# Create and use a file system source
factory = SourceFactory()
config: FileSystemSourceConfig = {
    "enabled": True,
    "root_path": "/path/to/code",
    "use_gitignore": True,
}

source = factory.create_source("filesystem", config)
content_items = await source.discover_content(config)
```

## Migration Support

```python
from codeweaver.sources.migration_guide import MigrationHelper

# Extend existing configuration with data source support
ExtendedConfig = MigrationHelper.extend_existing_config()
config = ExtendedConfig()
config.ensure_data_sources_initialized()
```
"""

from codeweaver.sources.api import APISource
from codeweaver.sources.base import (
    AbstractDataSource,
    BaseSourceConfig,
    ContentItem,
    DataSource,
    SourceCapability,
    SourceConfig,
    SourceRegistry,
    SourceWatcher,
    get_source_registry,
)
from codeweaver.sources.config import (
    DataSourcesConfig,
    extend_config_with_data_sources,
    get_example_data_sources_config,
)
from codeweaver.sources.database import DatabaseSource
from codeweaver.sources.factory import SourceFactory, get_source_factory
from codeweaver.sources.filesystem import FileSystemSource, FileSystemSourceConfig

# Placeholder implementations (for future development)
from codeweaver.sources.git import GitRepositorySource
from codeweaver.sources.integration import (
    BackwardCompatibilityAdapter,
    DataSourceManager,
    create_backward_compatible_server_integration,
    integrate_data_sources_with_config,
)
from codeweaver.sources.web import WebCrawlerSource


__all__ = [
    "APISource",
    "AbstractDataSource",
    "BackwardCompatibilityAdapter",
    "BaseSourceConfig",
    # Core protocols and data structures
    "ContentItem",
    "DataSource",
    # Integration and migration
    "DataSourceManager",
    # Configuration system
    "DataSourcesConfig",
    "DatabaseSource",
    # File system implementation (fully implemented)
    "FileSystemSource",
    "FileSystemSourceConfig",
    # Placeholder implementations (for future development)
    "GitRepositorySource",
    "SourceCapability",
    "SourceConfig",
    # Factory system
    "SourceFactory",
    "SourceRegistry",
    "SourceWatcher",
    "WebCrawlerSource",
    "create_backward_compatible_server_integration",
    "extend_config_with_data_sources",
    "get_example_data_sources_config",
    "get_source_factory",
    "get_source_registry",
    "integrate_data_sources_with_config",
]

# Version info for the data source system
__version__ = "1.0.0"
__status__ = "Production Ready (FileSystem), Preview (Others)"
