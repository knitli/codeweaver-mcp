# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Integration utilities for migrating to the unified factory system.

Provides backward compatibility helpers and migration utilities for
seamlessly transitioning from direct instantiation to factory-based creation.
"""

import logging

from contextlib import asynccontextmanager

from codeweaver.config import CodeWeaverConfig
from codeweaver.factories.extensibility_manager import ExtensibilityConfig, ExtensibilityManager


logger = logging.getLogger(__name__)


@asynccontextmanager
async def create_extensibility_context(
    config: CodeWeaverConfig, extensibility_config: ExtensibilityConfig | None = None
) -> ExtensibilityManager:
    """Create an extensibility context for easy migration.

    Args:
        config: Main CodeWeaver configuration
        extensibility_config: Optional extensibility configuration

    Yields:
        Initialized ExtensibilityManager instance
    """
    manager = ExtensibilityManager(config, extensibility_config)

    try:
        await manager.initialize()
        yield manager
    finally:
        await manager.shutdown()
