# Copyright (c) 2024 to present Pydantic Services Inc
# SPDX-License-Identifier: MIT
# Applies to original code in this directory (`src/codeweaver/embedding_providers/`) from `pydantic_ai`.
#
# SPDX-FileCopyrightText: 2025 (c) 2025 Knitli Inc.
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
# applies to new/modified code in this directory (`src/codeweaver/embedding_providers/`)
"""VoyageAI embedding provider."""

from __future__ import annotations

from codeweaver._settings import Provider
from codeweaver.embedding.providers import EmbeddingProvider


try:
    from voyageai.client_async import AsyncClient
except ImportError as _import_error:
    raise ImportError(
        'Please install the `voyageai` package to use the Voyage provider, you can use the `voyage` optional group — `pip install "codeweaver[voyage]"`'
    ) from _import_error


class VoyageEmbeddingProvider(EmbeddingProvider[AsyncClient]):
    """VoyageAI embedding provider."""

    def __init__(self, api_key: str | None = None, client: AsyncClient | None = None) -> None:
        """Initialize the Voyage embedding provider.

        Args:
            api_key: The Voyage API key. If not provided, will use VOYAGE_API_KEY environment variable.
            client: An existing Voyage client. If provided, api_key will be ignored.
        """
        self._client = client if client is not None else AsyncClient(api_key=api_key)

    @property
    def name(self) -> Provider:
        """Get the name of the embedding provider."""
        return Provider.VOYAGE

    @property
    def base_url(self) -> str | None:
        """Get the base URL of the embedding provider."""
        return "https://api.voyageai.com/v1"

    @property
    def client(self) -> AsyncClient:
        """Get the client for the embedding provider."""
        return self._client

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents into vectors."""
        # TODO: Implement document embedding using Voyage API
        raise NotImplementedError("Document embedding not yet implemented for Voyage provider")

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query into a vector."""
        # TODO: Implement query embedding using Voyage API
        raise NotImplementedError("Query embedding not yet implemented for Voyage provider")

    def get_vector_name(self) -> str:
        """Get the name of the vector for the collection."""
        return "voyage-embeddings"

    def get_vector_size(self) -> int:
        """Get the size of the vector for the collection."""
        # TODO: Return actual vector size based on model
        return 1024  # Default size, should be configurable based on model
