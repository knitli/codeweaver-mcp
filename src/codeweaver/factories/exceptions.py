# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Exception classes for the CodeWeaver factory system.

Provides a comprehensive hierarchy of exceptions for different types of
factory-related errors with clear categorization and recovery guidance.
"""


class CodeWeaverFactoryError(Exception):
    """Base exception for factory-related errors."""


class ConfigurationError(CodeWeaverFactoryError):
    """Configuration validation or processing error."""


class ComponentNotFoundError(CodeWeaverFactoryError):
    """Requested component not found in registry."""


class ComponentUnavailableError(CodeWeaverFactoryError):
    """Component found but not available for use."""


class ComponentCreationError(CodeWeaverFactoryError):
    """Error during component creation."""


class PluginError(CodeWeaverFactoryError):
    """Plugin-related error."""


class RegistrationError(CodeWeaverFactoryError):
    """Component registration error."""


class ValidationError(CodeWeaverFactoryError):
    """Validation failure error."""
