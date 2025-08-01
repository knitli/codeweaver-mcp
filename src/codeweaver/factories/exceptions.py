# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Factory exceptions moved to codeweaver.types to avoid circular imports.

This module is kept for backwards compatibility but should not be used.
Import factory exceptions from codeweaver.types instead.
"""

import warnings

def __getattr__(name: str):
    """Provide backwards compatibility for moved exceptions."""
    if name in [
        "CodeWeaverFactoryError",
        "ComponentCreationError", 
        "ComponentNotFoundError",
        "ComponentUnavailableError",
        "PluginError",
        "RegistrationError", 
        "ValidationError",
    ]:
        warnings.warn(
            f"Importing {name} from codeweaver.factories.exceptions is deprecated. "
            f"Import from codeweaver.types instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Import at runtime to avoid circular dependency
        from codeweaver.types import __dict__ as types_dict
        return types_dict[name]
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# For backwards compatibility, we'll keep the old names but warn about deprecation
__all__ = [
    "CodeWeaverFactoryError",
    "ComponentCreationError", 
    "ComponentNotFoundError",
    "ComponentUnavailableError",
    "PluginError",
    "RegistrationError",
    "ValidationError",
]