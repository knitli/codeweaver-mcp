# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Provider registry enhancement for sources module.

Following the backend improvement pattern with full capability information
and validation support for custom source providers.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeweaver._types.source_capabilities import SourceCapabilities
from codeweaver._types.source_enums import SourceProvider


if TYPE_CHECKING:
    from codeweaver.sources.base import DataSource


@dataclass
class SourceProviderInfo:
    """Information about a source provider."""

    source_class: type["DataSource"]
    capabilities: SourceCapabilities
    provider_type: SourceProvider
    display_name: str
    description: str
    implemented: bool = True


# Registry with full capability information
# This will be populated when sources are registered
SOURCE_PROVIDERS: dict[SourceProvider, SourceProviderInfo] = {}
