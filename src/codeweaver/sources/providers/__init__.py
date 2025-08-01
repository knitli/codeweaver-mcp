# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Source Providers for CodeWeaver."""

from codeweaver.sources.providers.database import DatabaseSourceConfig, DatabaseSourceProvider
from codeweaver.sources.providers.filesystem import (
    FileSystemSource,
    FileSystemSourceConfig,
    FileSystemSourceWatcher,
)
from codeweaver.sources.providers.git import GitRepositorySourceConfig, GitRepositorySourceProvider
from codeweaver.sources.providers.web import WebCrawlerSourceConfig, WebCrawlerSourceProvider


__all__ = (
    "DatabaseSourceConfig",
    "DatabaseSourceProvider",
    "FileSystemSource",
    "FileSystemSourceConfig",
    "FileSystemSourceWatcher",
    "GitRepositorySourceConfig",
    "GitRepositorySourceProvider",
    "WebCrawlerSourceConfig",
    "WebCrawlerSourceProvider",
)
