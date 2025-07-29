# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Centralized enum types for the sources module.

Replaces string literals with proper enum types following the
backend improvement pattern for type safety and attribute consolidation.
"""

from codeweaver.types.base_enum import BaseEnum


class SourceProvider(BaseEnum):
    """All supported source providers."""

    FILESYSTEM = "filesystem"
    GIT = "git"
    DATABASE = "database"
    API = "api"
    WEB = "web"


class ContentType(BaseEnum):
    """Content types for discovered items."""

    # Source types
    FILE = "file"
    URL = "url"
    DATABASE = "database"
    API = "api"
    GIT = "git"

    # Content categories
    CODE = "code"
    DOCUMENTATION = "documentation"
    CONFIG = "config"
    TEXT = "text"


class DatabaseType(BaseEnum):
    """Supported database types."""

    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"
    ELASTICSEARCH = "elasticsearch"


class APIType(BaseEnum):
    """Supported API types."""

    REST = "rest"
    GRAPHQL = "graphql"
    OPENAPI = "openapi"
    SWAGGER = "swagger"


class AuthType(BaseEnum):
    """Authentication types."""

    BEARER = "bearer"
    BASIC = "basic"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"


class SourceCapability(BaseEnum):
    """Capabilities supported by different data sources."""

    # Core capabilities
    CONTENT_DISCOVERY = "content_discovery"
    CONTENT_READING = "content_reading"
    CHANGE_WATCHING = "change_watching"

    # Advanced capabilities
    INCREMENTAL_SYNC = "incremental_sync"
    VERSION_HISTORY = "version_history"
    METADATA_EXTRACTION = "metadata_extraction"
    REAL_TIME_UPDATES = "real_time_updates"
    BATCH_PROCESSING = "batch_processing"
    CONTENT_DEDUPLICATION = "content_deduplication"
    RATE_LIMITING = "rate_limiting"
    AUTHENTICATION = "authentication"
    PAGINATION = "pagination"
