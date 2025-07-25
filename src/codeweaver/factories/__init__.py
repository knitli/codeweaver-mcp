# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Unified factory system for CodeWeaver extensible architecture.

Provides centralized creation and management of backends, providers, and data sources
with plugin discovery, dependency injection, and lifecycle management.
"""

from codeweaver.factories.dependency_injection import DependencyContainer, Lifecycle, ServiceLocator
from codeweaver.factories.extensibility_manager import ExtensibilityConfig, ExtensibilityManager
from codeweaver.factories.integration import (
    ServerMigrationHelper,
    create_extensibility_context,
    create_migration_config,
    validate_migration_readiness,
)
from codeweaver.factories.plugin_discovery import (
    BackendPlugin,
    PluginDiscovery,
    ProviderPlugin,
    SourcePlugin,
)
from codeweaver.factories.unified_factory import BackendFactory, UnifiedFactory
from codeweaver.factories.unified_factory import ProviderFactory as UnifiedProviderFactory
from codeweaver.factories.unified_factory import SourceFactory as UnifiedSourceFactory
from codeweaver.factories.validation import CompatibilityLevel, FactoryValidator, ValidationLevel


__all__ = [
    # Core factories
    "BackendFactory",
    "BackendPlugin",
    "CompatibilityLevel",
    # Dependency injection
    "DependencyContainer",
    "ExtensibilityConfig",
    # Extensibility management
    "ExtensibilityManager",
    # Validation
    "FactoryValidator",
    # Integration utilities
    "Lifecycle",
    # Plugin discovery
    "PluginDiscovery",
    "ProviderPlugin",
    "ServerMigrationHelper",
    "ServiceLocator",
    "SourcePlugin",
    "UnifiedFactory",
    "UnifiedProviderFactory",
    "UnifiedSourceFactory",
    "ValidationLevel",
    "create_extensibility_context",
    "create_migration_config",
    "validate_migration_readiness",
]
