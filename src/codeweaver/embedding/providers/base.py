"""Base class for embedding providers."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from codeweaver._settings import Provider
from codeweaver.embedding.profiles import EmbeddingModelProfile


class EmbeddingProvider[EmbeddingClient](BaseModel, ABC):
    """
    Abstract class for an embedding provider.

    This class mirrors `pydantic_ai.providers.Provider` class to make it simple to use
    existing implementations of `pydantic_ai.providers.Provider` as embedding providers.

    We chose to separate this from the `pydantic_ai.providers.Provider` class for clarity. That class is re-exported in `codeweaver.agent_providers.py` as `AgentProvider`, which is used for agent operations.
    We didn't want folks accidentally conflating agent operations with embedding operations. That's kind of a 'dogs and cats living together' 🐕🐈 situation.

    Each provider only supports a specific interface, but an interface can be used by multiple providers.

    The primary example of this one-to-many relationship is the OpenAI provider, which supports any OpenAI-compatible provider (Azure, Ollama, Fireworks, Heroku, Together, Github).
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    _client: EmbeddingClient
    _provider: Provider
    _profile: EmbeddingModelProfile | None

    @property
    def name(self) -> Provider:
        """Get the name of the embedding provider."""
        return self._provider

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
    def client(self) -> EmbeddingClient:
        """Get the client for the embedding provider."""
        return self._client

    @property
    def model_profile(self) -> EmbeddingModelProfile | None:
        """Get the model profile for the embedding provider."""
        return self._profile
