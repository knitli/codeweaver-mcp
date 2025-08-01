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
"""

# Import new types
from codeweaver.sources.base import (
    AbstractDataSource,
    DataSource,
    SourceConfig,
    SourceRegistry,
    SourceWatcher,
    get_source_registry,
)
from codeweaver.sources.config import DataSourcesConfig
from codeweaver.sources.factory import SourceFactory, get_source_factory
from codeweaver.sources.filesystem import FileSystemSource, FileSystemSourceConfig
from codeweaver.sources.integration import DataSourceManager
from codeweaver.types import (
    SOURCE_PROVIDERS,
    APIType,
    AuthType,
    ContentType,
    DatabaseType,
    SourceCapabilities,
    SourceCapability,
    SourceProvider,
    SourceProviderInfo,
)


__all__ = [
    "SOURCE_PROVIDERS",
    "APIType",
    # Core protocols and data structures
    "AbstractDataSource",
    "AuthType",
    "ContentType",
    "DataSource",
    "DataSourceManager",
    "DataSourcesConfig",
    "DatabaseType",
    "FileSystemSource",
    "FileSystemSourceConfig",
    "SourceCapabilities",
    "SourceCapability",
    "SourceConfig",
    "SourceFactory",
    # New type system
    "SourceProvider",
    "SourceProviderInfo",
    "SourceRegistry",
    "SourceWatcher",
    "get_source_factory",
    "get_source_registry",
]
