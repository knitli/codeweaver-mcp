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
from typing import Any

from codeweaver._settings import Provider
from codeweaver.embedding.profiles import EmbeddingModelProfile


type NonClient = None


class EmbeddingProvider[InterfaceClient](ABC):
    """
    Abstract class for an embedding provider.

    This class mirrors `pydantic_ai.providers.Provider` class to make it simple to use
    existing implementations of `pydantic_ai.providers.Provider` as embedding providers.

    We chose to separate this from the `pydantic_ai.providers.Provider` class for clarity. That class is re-exported in `codeweaver.agent_providers.py` as `AgentProvider`, which is used for agent operations.
    We didn't want folks accidentally conflating agent operations with embedding operations. That's kind of a 'dogs and cats living together' 🐕🐈 situation.

    Each provider only supports a specific interface, but an interface can be used by multiple providers.

    The primary example of this one-to-many relationship is the OpenAI provider, which supports any OpenAI-compatible provider (Azure, Ollama, Fireworks, Heroku, Together, Github).
    """

    _client: InterfaceClient

    @property
    @abstractmethod
    def name(self) -> Provider:
        """Get the name of the embedding provider."""

    @property
    @abstractmethod
    def base_url(self) -> str | None:
        """Get the base URL of the embedding provider, if any."""

    @abstractmethod
    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents into vectors."""

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """Embed a query into a vector."""

    @abstractmethod
    def get_vector_name(self) -> str:
        """Get the name of the vector for the collection."""

    @abstractmethod
    def get_vector_size(self) -> int:
        """Get the size of the vector for the collection."""

    @property
    @abstractmethod
    def client(self) -> InterfaceClient:
        """Get the client for the embedding provider."""

    @property
    def model_profile(self) -> EmbeddingModelProfile | None:
        """Get the model profile for the embedding provider."""
        return None


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
