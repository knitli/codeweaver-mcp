# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Base class for reranking providers."""

from abc import ABC
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple, cast, overload

from pydantic import BaseModel, ConfigDict

from codeweaver._data_structures import CodeChunk
from codeweaver._server import get_statistics
from codeweaver._settings import Provider
from codeweaver.reranking.models.base import RerankingModelCapabilities
from codeweaver.tokenizers import Tokenizer, get_tokenizer


type StructuredDataInput = str | bytes | bytearray | CodeChunk
type StructuredDataSequence = (
    Sequence[str] | Sequence[bytes] | Sequence[bytearray] | Sequence[CodeChunk]
)


class RerankingResult(NamedTuple):
    """Result of a reranking operation."""

    original_index: int
    batch_rank: int
    score: float
    chunk: CodeChunk


class RerankingProvider[RerankingClient](BaseModel, ABC):
    """Base class for reranking providers."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    _client: RerankingClient
    _provider: Provider
    _prompt: str | None
    _caps: RerankingModelCapabilities

    _rerank_kwargs: dict[str, Any]
    _input_transformer: Callable[[StructuredDataSequence | StructuredDataInput], Sequence[str]]
    _output_transformer: Callable[[Any, Sequence[CodeChunk]], Sequence[RerankingResult]]

    def __init__(
        self,
        client: RerankingClient,
        capabilities: RerankingModelCapabilities,
        prompt: str | None = None,
        **kwargs: dict[str, Any] | None,
    ) -> None:
        """Initialize the RerankingProvider."""
        self._client = client
        self._prompt = prompt
        self._caps = capabilities
        self.kwargs = {**(type(self)._rerank_kwargs or {}), **(kwargs or {})}

    async def _execute_rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_k: int = 40,
        **kwargs: dict[str, Any] | None,
    ) -> Any:
        """Execute the reranking process."""
        raise NotImplementedError

    async def rerank(
        self,
        query: str,
        documents: StructuredDataInput | StructuredDataSequence,
        top_k: int = 40,
        **kwargs: dict[str, Any] | None,
    ) -> Sequence[RerankingResult]:
        """Rerank the given documents based on the query."""
        processed_kwargs = self._set_kwargs(**kwargs)
        transformed_docs = self._process_documents(documents)
        transformed_docs = self._input_transformer(transformed_docs)
        reranked = await self._execute_rerank(
            query, transformed_docs, top_k=top_k, **processed_kwargs
        )
        return self._process_results(reranked, transformed_docs)

    @property
    def client(self) -> RerankingClient:
        """Get the client for the reranking provider."""
        return self._client

    @property
    def model_capabilities(self) -> RerankingModelCapabilities:
        """Get the model capabilities for the reranking provider."""
        return self._caps

    @property
    def prompt(self) -> str | None:
        """Get the prompt for the reranking provider."""
        return self._prompt

    def _tokenizer(self) -> Tokenizer[Any]:
        """Retrieves the tokenizer associated with the reranking model."""
        if tokenizer := self.model_capabilities.tokenizer:
            return get_tokenizer(
                tokenizer, self.model_capabilities.tokenizer_model or self.model_capabilities.name
            )
        return get_tokenizer("tiktoken", "cl100k_base")

    @property
    def tokenizer(self) -> Tokenizer[Any]:
        """Get the tokenizer for the reranking provider."""
        return self._tokenizer()

    def _set_kwargs(self, **kwargs: dict[str, Any] | None) -> dict[str, Any]:
        """Set the keyword arguments for the reranking provider."""
        return self.kwargs | (kwargs or {})

    @overload
    def _update_token_stats(self, *, token_count: int) -> None: ...
    @overload
    def _update_token_stats(
        self, *, from_docs: Sequence[str] | Sequence[Sequence[str]]
    ) -> None: ...
    def _update_token_stats(
        self,
        *,
        token_count: int | None = None,
        from_docs: Sequence[str] | Sequence[Sequence[str]] | None = None,
    ) -> None:
        """Update token statistics for the embedding provider."""
        statistics = get_statistics()
        if token_count is not None:
            statistics.add_token_usage(reranking_generated=token_count)
        elif from_docs and all(isinstance(doc, str) for doc in from_docs):
            token_count = (
                self.tokenizer.estimate_batch(from_docs)  # pyright: ignore[reportArgumentType]
                if all(isinstance(doc, str) for doc in from_docs)
                else sum(self.tokenizer.estimate_batch(item) for item in from_docs)  # type: ignore
            )
            statistics.add_token_usage(reranking_generated=token_count)

    def _process_documents(
        self, documents: StructuredDataInput | StructuredDataSequence
    ) -> Sequence[str]:
        """Process the input documents into a uniform format."""
        if isinstance(documents, list | tuple | set):
            return [
                doc.serialize() if isinstance(doc, CodeChunk) else str(doc) for doc in documents
            ]
        return [documents.serialize()] if isinstance(documents, CodeChunk) else [str(documents)]

    def _process_results(
        self, results: Any, transformed_docs: Sequence[str]
    ) -> Sequence[RerankingResult]:
        """Process the results from the reranking."""
        chunks = [self.to_code_chunk(doc) for doc in transformed_docs]
        return self._output_transformer(results, chunks)

    @staticmethod
    def to_code_chunk(
        text: StructuredDataInput | StructuredDataSequence,
    ) -> CodeChunk | tuple[CodeChunk, ...]:
        """Convert text to a CodeChunk."""
        if isinstance(text, list | tuple | set):
            return tuple(
                t if isinstance(t, CodeChunk) else CodeChunk.model_validate_json(t) for t in text
            )  # type: ignore[return-value]
        return (
            text
            if isinstance(text, CodeChunk)
            else CodeChunk.model_validate_json(cast(str | bytes | bytearray, text))
        )
