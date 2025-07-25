# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Unified factory system for CodeWeaver extensible architecture.

Provides centralized creation and management of backends, providers, and data sources
with plugin discovery, dependency injection, and lifecycle management.
"""

from codeweaver._types import ExtensibilityConfig
from codeweaver.factories.backend_registry import BackendRegistry
from codeweaver.factories.codeweaver_factory import CodeWeaverFactory
from codeweaver.factories.error_handling import FactoryError
from codeweaver.factories.extensibility_manager import ExtensibilityManager
from codeweaver.factories.initialization import FactoryInitializer
from codeweaver.factories.plugin_protocols import BackendPlugin, ProviderPlugin, SourcePlugin
from codeweaver.factories.source_registry import SourceRegistry


__all__ = [
    # Plugin protocols
    "BackendPlugin",
    "BackendRegistry",
    # Core factories
    "CodeWeaverFactory",
    # Extensibility management
    "ExtensibilityConfig",
    "ExtensibilityManager",
    # Error handling
    "FactoryError",
    # Initialization
    "FactoryInitializer",
    "ProviderPlugin",
    "SourcePlugin",
    "SourceRegistry",
]
