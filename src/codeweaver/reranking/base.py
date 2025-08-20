# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Base class for reranking providers."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict

from codeweaver._data_structures import CodeChunk


class RerankingProvider[RerankingClient](BaseModel, ABC):
    """Base class for reranking providers."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    _client: RerankingClient
    _model: Any
    _prompt: str | None

    def __init__(self, client: RerankingClient, model: Any, prompt: str | None = None) -> None:
        """Initialize the RerankingProvider."""
        self._client = client
        self._model = model
        self._prompt = prompt

    @staticmethod
    def to_code_chunk(text: str | bytes | CodeChunk) -> CodeChunk:
        """Convert text to a CodeChunk."""
        return text if isinstance(text, CodeChunk) else CodeChunk.model_validate_json(text)

    @abstractmethod
    def rerank(
        self, query: str, documents: list[str | bytes | CodeChunk], top_k: int = 15
    ) -> list[CodeChunk]:
        """Rerank the given documents based on the query."""

    @property
    def client(self) -> RerankingClient:
        """Get the client for the reranking provider."""
        return self._client

    @property
    def model(self) -> Any:
        """Get the model for the reranking provider."""
        return self._model

    @property
    def prompt(self) -> str | None:
        """Get the prompt for the reranking provider."""
        return self._prompt
