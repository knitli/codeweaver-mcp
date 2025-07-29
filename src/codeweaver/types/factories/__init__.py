# SPDX-FileCopyrightText: 2025 Knitli Inc.
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Types for CodeWeaver's factories."""

from codeweaver.types.factories.core import (
    BaseCapabilities,
    BaseComponentInfo,
    CapabilityQueryMixin,
    ComponentRegistration,
    ComponentRegistry,
    ComponentType,
    CreationResult,
    RegistrationResult,
    ValidationResult,
)
from codeweaver.types.factories.data_structures import (
    ComponentInstances,
    ComponentLifecycle,
    ContentItem,
    ExtensibilityConfig,
    InitializationContext,
    InitializationResult,
    PluginInfo,
    ProviderRegistration,
)


__all__ = (
    "BaseCapabilities",
    "BaseComponentInfo",
    "CapabilityQueryMixin",
    "ComponentInstances",
    "ComponentLifecycle",
    "ComponentRegistration",
    "ComponentRegistry",
    "ComponentType",
    "ContentItem",
    "CreationResult",
    "ExtensibilityConfig",
    "InitializationContext",
    "InitializationResult",
    "PluginInfo",
    "ProviderRegistration",
    "RegistrationResult",
    "ValidationResult",
)
