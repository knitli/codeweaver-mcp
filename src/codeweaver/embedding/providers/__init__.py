# Copyright (c) 2024 to present Pydantic Services Inc
# SPDX-License-Identifier: MIT
# Applies to original code in this directory (`src/codeweaver/embedding_providers/`) from `pydantic_ai`.
#
# SPDX-FileCopyrightText: 2025 (c) 2025 Knitli Inc.
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
# applies to new/modified code in this directory (`src/codeweaver/embedding_providers/`)

"""Entry point for embedding providers. Defines the abstract base class and includes a utility for retrieving specific provider implementations."""

from typing import Any

from codeweaver._settings import Provider
from codeweaver.embedding.providers.base import EmbeddingProvider


def _infer_embedding_provider_class(provider: Provider) -> type[EmbeddingProvider[Any]]:  # noqa: C901  # long? yes. Complex. No.
    # sourcery skip: no-long-functions
    """
    Infer the embedding provider class from the provider name.

    Args:
        provider: The Provider enum representing the embedding provider.

    Returns:
        The class of the embedding provider.
    """
    if provider == Provider.VOYAGE:
        from codeweaver.embedding.providers.voyage import VoyageEmbeddingProvider

        return VoyageEmbeddingProvider  # type: ignore[return-value]

    if provider == Provider.OPENAI:
        from codeweaver.embedding.providers.openai import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider  # type: ignore[return-value]

    if provider == Provider.MISTRAL:
        from codeweaver.embedding.providers.mistral import MistralEmbeddingProvider

        return MistralEmbeddingProvider  # type: ignore[return-value]

    if provider == Provider.COHERE:
        from codeweaver.embedding.providers.cohere import CohereEmbeddingProvider

        return CohereEmbeddingProvider  # type: ignore[return-value]

    if provider == Provider.VERCEL:
        from codeweaver.embedding.providers.vercel import VercelEmbeddingProvider

        return VercelEmbeddingProvider  # type: ignore[return-value]

    if provider == Provider.BEDROCK:
        from codeweaver.embedding.providers.bedrock import BedrockEmbeddingProvider

        return BedrockEmbeddingProvider  # type: ignore[return-value]
    if provider == Provider.GOOGLE:
        from codeweaver.embedding.providers.google import GoogleEmbeddingProvider

        return GoogleEmbeddingProvider  # type: ignore[return-value]
    if provider == Provider.HUGGINGFACE:
        from codeweaver.embedding.providers.huggingface import HuggingFaceEmbeddingProvider

        return HuggingFaceEmbeddingProvider  # type: ignore[return-value]
    if provider == Provider.FIREWORKS:
        from codeweaver.embedding.providers.fireworks import FireworksEmbeddingProvider

        return FireworksEmbeddingProvider  # type: ignore[return-value]

    if provider == Provider.OLLAMA:
        from codeweaver.embedding.providers.ollama import OllamaEmbeddingProvider

        return OllamaEmbeddingProvider  # type: ignore[return-value]
    if provider == Provider.TOGETHER:
        from codeweaver.embedding.providers.together import TogetherEmbeddingProvider

        return TogetherEmbeddingProvider  # type: ignore[return-value]
    if provider == Provider.AZURE:
        from codeweaver.embedding.providers.azure import AzureEmbeddingProvider

        return AzureEmbeddingProvider  # type: ignore[return-value]

    if provider == Provider.HEROKU:
        from codeweaver.embedding.providers.heroku import HerokuEmbeddingProvider

        return HerokuEmbeddingProvider  # type: ignore[return-value]
    if provider == Provider.GITHUB:
        from codeweaver.embedding.providers.github import GitHubEmbeddingProvider

        return GitHubEmbeddingProvider  # type: ignore[return-value]

    raise ValueError(f"Unknown embedding provider: {provider}")


def infer_embedding_provider(provider: Provider) -> EmbeddingProvider[Any]:
    """
    Infer the embedding provider from the provider name.

    Args:
        provider: The name of the embedding provider.

    Returns:
        An instance of the embedding provider.
    """
    provider_class = _infer_embedding_provider_class(provider)
    return provider_class()


__all__ = ("EmbeddingProvider", "infer_embedding_provider")
