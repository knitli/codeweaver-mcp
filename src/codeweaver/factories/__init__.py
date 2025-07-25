# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Unified factory system for CodeWeaver extensible architecture.

Provides centralized creation and management of backends, providers, and data sources
with plugin discovery, dependency injection, and lifecycle management.
"""

from codeweaver.factories.backend_registry import BackendRegistry
from codeweaver.factories.codeweaver_factory import CodeWeaverFactory
from codeweaver.factories.error_handling import ComponentError, FactoryError
from codeweaver.factories.error_handling import ValidationError as FactoryValidationError
from codeweaver.factories.extensibility_manager import ExtensibilityConfig, ExtensibilityManager
from codeweaver.factories.initialization import FactoryInitializer
from codeweaver.factories.integration import create_extensibility_context
from codeweaver.factories.plugin_protocols import (
    BackendPlugin,
    ComponentRegistry,
    ProviderPlugin,
    SourcePlugin,
)
from codeweaver.factories.source_registry import SourceRegistry
from codeweaver.factories.validation import CompatibilityLevel, FactoryValidator, ValidationLevel


__all__ = [
    # Plugin protocols
    "BackendPlugin",
    "BackendRegistry",
    # Core factories
    "CodeWeaverFactory",
    # Validation
    "CompatibilityLevel",
    "ComponentError",
    "ComponentRegistry",
    # Extensibility management
    "ExtensibilityConfig",
    "ExtensibilityManager",
    # Error handling
    "FactoryError",
    # Initialization
    "FactoryInitializer",
    "FactoryValidationError",
    "FactoryValidator",
    "ProviderPlugin",
    "SourcePlugin",
    "SourceRegistry",
    "ValidationLevel",
    # Integration utilities
    "create_extensibility_context",
]
