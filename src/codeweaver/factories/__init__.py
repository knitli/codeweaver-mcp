# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Unified factory system for CodeWeaver extensible architecture.

Provides centralized creation and management of backends, providers, and data sources
with consolidated registry, base patterns, and simplified factory implementation.
"""

# New consolidated structure
# Legacy imports for backward compatibility (deprecated)
from codeweaver.factories.backend_registry import BackendRegistry
from codeweaver.factories.base import (
    BaseComponentFactory,
    BaseRegistry,
    ComponentFactory,
    ComponentRegistration,
    FactoryContext,
    RegistryProtocol,
    create_factory_context,
    validate_component_factory,
    validate_registry,
)
from codeweaver.factories.codeweaver_factory import CodeWeaverFactory as LegacyCodeWeaverFactory
from codeweaver.factories.error_handling import FactoryError
from codeweaver.factories.factory import CodeWeaverFactory, get_global_factory, reset_global_factory
from codeweaver.factories.plugin_protocols import BackendPlugin, ProviderPlugin, SourcePlugin
from codeweaver.factories.registry import (
    BackendInfo,
    ComponentRegistry,
    SourceInfo,
    get_global_registry,
    reset_global_registry,
)
from codeweaver.factories.source_registry import SourceRegistry


__all__ = [
    "BackendInfo",
    # Plugin protocols (still used)
    "BackendPlugin",
    # Legacy exports (deprecated - use new structure)
    "BackendRegistry",
    "BaseComponentFactory",
    "BaseRegistry",
    # New consolidated structure (primary exports)
    "CodeWeaverFactory",
    "ComponentFactory",
    "ComponentRegistration",
    "ComponentRegistry",
    "FactoryContext",
    "FactoryError",
    "LegacyCodeWeaverFactory",
    "ProviderPlugin",
    "RegistryProtocol",
    "SourceInfo",
    "SourcePlugin",
    "SourceRegistry",
    # Factory utilities
    "create_factory_context",
    "get_global_factory",
    "get_global_registry",
    "reset_global_factory",
    "reset_global_registry",
    "validate_component_factory",
    "validate_registry",
]
