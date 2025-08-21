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
from typing import Annotated, Self

from pydantic import Field

from codeweaver.embedding.models import EmbeddingModelCapabilities


__all__ = ["DEFAULT_PROFILE", "EmbeddingModelProfile", "EmbeddingModelProfileSpec"]

DefaultCapabilities = EmbeddingModelCapabilities()


@dataclass
class EmbeddingModelProfile[ModelCapabilities]:
    """Describes how requests to a specific model or family of models need to be constructed to get the best results, independent of the model and provider classes used."""

    _capabilities: Annotated[
        EmbeddingModelCapabilities, Field(default_factory=EmbeddingModelCapabilities)
    ]

    @classmethod
    def from_profile(cls, profile: EmbeddingModelProfile[ModelCapabilities] | None) -> Self:
        """Build a ModelProfile subclass instance from a ModelProfile instance."""
        return profile if isinstance(profile, cls) else cls(cls._capabilities).update(profile)

    def update(self, profile: EmbeddingModelProfile[ModelCapabilities] | None) -> Self:
        """Update this EmbeddingModelProfile (subclass) instance with the non-default values from another EmbeddingModelProfile instance."""
        if not profile:
            return self
        field_names = {f.name for f in fields(self)}
        non_default_attrs = {
            f.name: getattr(profile, f.name)
            for f in fields(profile)
            if f.name in field_names and getattr(profile, f.name) != f.default
        }
        return replace(self, **non_default_attrs)


type EmbeddingModelProfileSpec = (
    EmbeddingModelProfile[EmbeddingModelCapabilities]
    | Callable[[str], EmbeddingModelProfile[EmbeddingModelCapabilities] | None]
)

DEFAULT_PROFILE: EmbeddingModelProfile[EmbeddingModelCapabilities] = EmbeddingModelProfile(
    DefaultCapabilities
)
