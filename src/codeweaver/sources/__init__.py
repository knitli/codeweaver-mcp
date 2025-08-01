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
from codeweaver.cw_types import (
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
from codeweaver.sources.integration import DataSourceManager
from codeweaver.sources.providers import (
    DatabaseSourceConfig,
    DatabaseSourceProvider,
    FileSystemSource,
    FileSystemSourceConfig,
    FileSystemSourceWatcher,
    GitRepositorySourceConfig,
    GitRepositorySourceProvider,
    WebCrawlerSourceConfig,
    WebCrawlerSourceProvider,
)


__all__ = (
    "SOURCE_PROVIDERS",
    "APIType",
    "AbstractDataSource",
    "AuthType",
    "ContentType",
    "DataSource",
    "DataSourceManager",
    "DataSourcesConfig",
    "DatabaseSourceConfig",
    "DatabaseSourceProvider",
    "DatabaseType",
    "FileSystemSource",
    "FileSystemSourceConfig",
    "FileSystemSourceWatcher",
    "GitRepositorySourceConfig",
    "GitRepositorySourceProvider",
    "SourceCapabilities",
    "SourceCapability",
    "SourceConfig",
    "SourceFactory",
    "SourceProvider",
    "SourceProviderInfo",
    "SourceRegistry",
    "SourceWatcher",
    "WebCrawlerSourceConfig",
    "WebCrawlerSourceProvider",
    "get_source_factory",
    "get_source_registry",
)
