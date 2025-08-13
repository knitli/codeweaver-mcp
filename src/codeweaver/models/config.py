# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Configuration models for provider settings."""

# For Phase 1, provider configurations are defined in settings.py
# This file is a placeholder for future provider-specific configurations

from codeweaver.settings import EmbeddingConfig, QdrantConfig, VectorStoreConfig, VoyageConfig


__all__ = [
    "EmbeddingConfig",
    "QdrantConfig",
    "VectorStoreConfig",
    "VoyageConfig",
]
