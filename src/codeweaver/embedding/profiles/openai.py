# Copyright (c) 2024 to present Pydantic Services Inc
# SPDX-License-Identifier: MIT
# Applies to original code in this directory (`src/codeweaver/embedding_providers/`) from `pydantic_ai`.
#
# SPDX-FileCopyrightText: 2025 (c) 2025 Knitli Inc.
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
# applies to new/modified code in this directory (`src/codeweaver/embedding_providers/`)
"""OpenAI embedding profiles."""

from codeweaver.embedding.profiles import EmbeddingModelProfile


class OpenAIEmbeddingModelProfile(EmbeddingModelProfile): ...


def openai_model_profile(model_name: str) -> EmbeddingModelProfile | None:
    """Get OpenAI model profile."""
