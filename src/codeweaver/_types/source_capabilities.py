# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Centralized source capability definitions using Pydantic v2.

Replaces hardcoded capability sets scattered across implementations
with a single source of truth for all source capabilities.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class SourceCapabilities(BaseModel):
    """Centralized source capability definitions."""

    # Core capabilities
    supports_content_discovery: bool = Field(True, description="Can discover content items")
    supports_content_reading: bool = Field(True, description="Can read content from items")
    supports_change_watching: bool = Field(False, description="Can watch for content changes")
    supports_incremental_sync: bool = Field(False, description="Supports incremental updates")
    supports_version_history: bool = Field(False, description="Provides version/commit history")
    supports_metadata_extraction: bool = Field(False, description="Rich metadata extraction")
    supports_real_time_updates: bool = Field(False, description="Real-time change notifications")
    supports_batch_processing: bool = Field(False, description="Efficient batch operations")
    supports_content_deduplication: bool = Field(False, description="Built-in deduplication")
    supports_rate_limiting: bool = Field(False, description="Built-in rate limiting")
    supports_authentication: bool = Field(False, description="Supports authentication")
    supports_pagination: bool = Field(False, description="Supports paginated discovery")

    # Performance characteristics
    max_content_size_mb: Annotated[
        int | None, Field(None, ge=1, description="Maximum supported content size")
    ]
    max_concurrent_requests: Annotated[
        int, Field(10, ge=1, le=100, description="Maximum concurrent requests")
    ]
    default_batch_size: Annotated[
        int, Field(8, ge=1, le=1000, description="Default batch processing size")
    ]

    # Dependencies
    required_dependencies: list[str] = Field(
        default_factory=list, description="Required Python packages"
    )
    optional_dependencies: list[str] = Field(
        default_factory=list, description="Optional Python packages"
    )
