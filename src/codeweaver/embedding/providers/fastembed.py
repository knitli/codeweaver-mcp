"""FastEmbed embedding provider implementation.

FastEmbed is a lightweight and efficient library for generating embeddings locally.
"""

import asyncio

from codeweaver._settings import Provider
from codeweaver.embedding.profiles import EmbeddingModelProfile
from codeweaver.embedding.providers import EmbeddingProvider, NonClient


try:
    from fastembed.common.model_description import DenseModelDescription
except ImportError as e:
    raise ImportError(
        "FastEmbed is not installed. Please install it with `pip install fastembed`."
    ) from e


class FastEmbedProvider(EmbeddingProvider[NonClient]):
    """
    FastEmbed implementation of the embedding provider.

    model_name: The name of the FastEmbed model to use.
    """

    _client: NonClient = None  # FastEmbed does not require a client
    _model_profile: EmbeddingModelProfile

    def __init__(self, embedding_model_profile: EmbeddingModelProfile) -> None:
        """Initialize the FastEmbed embedding provider."""
        self._model_profile = embedding_model_profile

    @property
    def name(self) -> Provider:
        """Get the enum member of the embedding provider."""
        return Provider.FASTEMBED

    @property
    def base_url(self) -> str | None:
        """FastEmbed does not use a base URL."""
        return None

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents into vectors."""
        # TODO: Adjust method calls to match EmbeddingModelProfile once we have a standard interface
        # Run in a thread pool since FastEmbed is synchronous
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self._model_profile.model.passage_embed(documents))
        )
        return [embedding.tolist() for embedding in embeddings]

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query into a vector."""
        # Run in a thread pool since FastEmbed is synchronous
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self._model_profile.model.query_embed([query]))
        )
        return embeddings[0].tolist()

    def get_vector_name(self) -> str:
        """
        Return the name of the vector for the Qdrant collection.
        Important: This is compatible with the FastEmbed logic used before 0.6.0.
        """
        model_name = self._model_profile.model.model_name.split("/")[-1].lower()
        return f"fast-{model_name}"

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        model_description: DenseModelDescription = self._model_profile.model._get_model_description(
            self._model_profile.model.model_name
        )
        return model_description.dim

    @property
    def client(self) -> NonClient:
        """Get the client for the embedding provider."""
        return self._client

    @property
    def model_profile(self) -> EmbeddingModelProfile | None:
        """Get the model profile for the embedding provider."""
        return self._model_profile
