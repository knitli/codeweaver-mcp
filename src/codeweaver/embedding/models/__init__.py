# Copyright (c) 2024 to present Pydantic Services Inc
# SPDX-License-Identifier: MIT
# Applies to original code in this directory (`src/codeweaver/embedding/`) from `pydantic_ai`.
# in files that are marked like this one.
#
# SPDX-FileCopyrightText: (c) 2025 Knitli Inc.
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
# applies to new/modified code in this directory (`src/codeweaver/embedding/`)

"""Entrypoint for CodeWeaver's embedding model system, heavily inspired by pydantic-ai."""

from codeweaver.embedding.models.base import EmbeddingModelCapabilities


def get_model(model: str) -> EmbeddingModelCapabilities:
    """Get the embedding model class by name."""
    if model.startswith("voyage"):
        from codeweaver.embedding.models.voyage import get_voyage_capabilities

        return next(model for model in get_voyage_capabilities() if model.name == model)
    raise ValueError(f"Unknown embedding model: {model}")
