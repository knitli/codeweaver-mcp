# SPDX-FileCopyrightText: 2025 Knitli Inc.
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Data source types and capabilities."""

from codeweaver.cw_types.sources.capabilities import SourceCapabilities
from codeweaver.cw_types.sources.enums import (
    APIType,
    AuthType,
    ContentType,
    DatabaseType,
    SourceCapability,
    SourceProvider,
)
from codeweaver.cw_types.sources.providers import SOURCE_PROVIDERS, SourceProviderInfo


__all__ = [
    "SOURCE_PROVIDERS",
    "APIType",
    "AuthType",
    "ContentType",
    "DatabaseType",
    "SourceCapabilities",
    "SourceCapability",
    "SourceProvider",
    "SourceProviderInfo",
]
