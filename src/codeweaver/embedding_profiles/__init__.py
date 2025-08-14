# Copyright (c) 2024 to present Pydantic Services Inc
# SPDX-License-Identifier: MIT
# Applies to original code in this directory (`src/codeweaver/embedding_providers/`) from `pydantic_ai`.
#
# SPDX-FileCopyrightText: 2025 (c) 2025 Knitli Inc.
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
# applies to new/modified code in this directory (`src/codeweaver/embedding_providers/`)

"""
Embedding model profiles for different providers.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Self

from pydantic_ai.profiles import InlineDefsJsonSchemaTransformer, JsonSchemaTransformer


__all__ = [
    "DEFAULT_PROFILE",
    "EmbeddingModelProfile",
    "EmbeddingModelProfileSpec",
    "InlineDefsJsonSchemaTransformer",
    "JsonSchemaTransformer",
]

# This is a placeholder implementation for the embedding model profile.
# We need to create the profiles for the providers and finish the factory function here.
# To see the overall idea, we're copying the pattern in `pydantic_ai.profiles`


@dataclass
class EmbeddingModelProfile:
    """Describes how requests to a specific model or family of models need to be constructed to get the best results, independent of the model and provider classes used."""

    @classmethod
    def from_profile(cls, profile: EmbeddingModelProfile | None) -> Self:
        """Build a ModelProfile subclass instance from a ModelProfile instance."""
        return profile if isinstance(profile, cls) else cls().update(profile)

    def update(self, profile: EmbeddingModelProfile | None) -> Self:
        """Update this EmbeddingModelProfile (subclass) instance with the non-default values from another ModelProfile instance."""
        if not profile:
            return self
        field_names = {f.name for f in fields(self)}
        non_default_attrs = {
            f.name: getattr(profile, f.name)
            for f in fields(profile)
            if f.name in field_names and getattr(profile, f.name) != f.default
        }
        return replace(self, **non_default_attrs)


EmbeddingModelProfileSpec = EmbeddingModelProfile | Callable[[str], EmbeddingModelProfile | None]

DEFAULT_PROFILE = EmbeddingModelProfile()
