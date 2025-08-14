# Copyright (c) 2024 to present Pydantic Services Inc
# SPDX-License-Identifier: MIT
# Applies to original code in this directory (`src/codeweaver/embedding_providers/`) from `pydantic_ai`.
#
# SPDX-FileCopyrightText: 2025 (c) 2025 Knitli Inc.
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
# applies to new/modified code in this directory (`src/codeweaver/embedding_providers/`)

"""Entry point for embedding providers. Defines the abstract base class and includes a utility for retrieving specific provider implementations."""

from abc import ABC, abstractmethod
from typing import Any, Generic, LiteralString, TypeVar

from codeweaver.embedding_profiles import EmbeddingModelProfile


InterfaceClient = TypeVar("InterfaceClient")


class EmbeddingProvider(ABC, Generic[InterfaceClient]):
    """
    Abstract class for an embedding provider.

    This class mirrors `pydantic_ai.providers.Provider` class to make it simple to use
    existing implementations of `pydantic_ai.providers.Provider` as embedding providers.

    We chose to separate this from the `pydantic_ai.providers.Provider` class for clarity.
    We didn't want folks accidentally conflating agent operations with embedding operations. That's kind of a 'dogs and cats living together' 🐕🐈 situation.

    Each provider only supports a specific interface, but an interface can be used by multiple providers.

    The primary example of this one-to-many relationship is the OpenAI provider, which supports any OpenAI-compatible provider.
    """

    _client: InterfaceClient

    @property
    @abstractmethod
    def name(self) -> LiteralString:
        """
        The name of the embedding provider.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def base_url(self) -> str:
        """
        The base URL of the embedding provider.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def client(self) -> InterfaceClient:
        """
        The client used to interact with the embedding provider.
        """
        raise NotImplementedError

    def model_profile(self) -> EmbeddingModelProfile | None:
        """
        Get the model profile for the embedding provider.
        """
        return None


def infer_embedding_provider_class(provider: LiteralString) -> type[EmbeddingProvider[Any]]:  # noqa: C901  # long? yes. Complex. No.
    # sourcery skip: no-long-functions
    """
    Infer the embedding provider class from the provider name.

    Args:
        provider: The name of the embedding provider.

    Returns:
        The class of the embedding provider.
    """
    if provider == "voyage":
        from codeweaver.models.embedding_providers.voyage import VoyageEmbeddingProvider

        return VoyageEmbeddingProvider

    if provider == "openai":
        from codeweaver.models.embedding_providers.openai import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider

    if provider == "mistral":
        from codeweaver.models.embedding_providers.mistral import MistralEmbeddingProvider

        return MistralEmbeddingProvider

    if provider == "cohere":
        from codeweaver.models.embedding_providers.cohere import CohereEmbeddingProvider

        return CohereEmbeddingProvider

    if provider == "vercel":
        from codeweaver.models.embedding_providers.vercel import VercelEmbeddingProvider

        return VercelEmbeddingProvider

    if provider == "bedrock":
        from codeweaver.models.embedding_providers.bedrock import BedrockEmbeddingProvider

        return BedrockEmbeddingProvider
    if provider == "google":
        from codeweaver.models.embedding_providers.google import GoogleGlaEmbeddingProvider

        return GoogleEmbeddingProvider
    if provider == "huggingface":
        from codeweaver.models.embedding_providers.huggingface import HuggingFaceEmbeddingProvider

        return HuggingFaceEmbeddingProvider
    if provider == "fireworks":
        from codeweaver.models.embedding_providers.fireworks import FireworksEmbeddingProvider

        return FireworksEmbeddingProvider

    if provider == "ollama":
        from codeweaver.models.embedding_providers.ollama import OllamaEmbeddingProvider

        return OllamaEmbeddingProvider
    if provider == "together":
        from codeweaver.models.embedding_providers.together import TogetherEmbeddingProvider

        return TogetherEmbeddingProvider
    if provider == "azure-openai":
        from codeweaver.models.embedding_providers.azure_openai import AzureOpenAIEmbeddingProvider

        return AzureOpenAIEmbeddingProvider

    if provider == "heroku":
        from codeweaver.models.embedding_providers.heroku import HerokuEmbeddingProvider

        return HerokuEmbeddingProvider
    if provider == "github":
        from codeweaver.models.embedding_providers.github import GitHubEmbeddingProvider

        return GitHubEmbeddingProvider

    raise ValueError(f"Unknown embedding provider: {provider}")


def infer_provider(provider: LiteralString) -> EmbeddingProvider[Any]:
    """
    Infer the embedding provider from the provider name.

    Args:
        provider: The name of the embedding provider.

    Returns:
        An instance of the embedding provider.
    """
    provider_class = infer_embedding_provider_class(provider)
    return provider_class()
