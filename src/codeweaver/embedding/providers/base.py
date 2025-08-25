"""Base class for embedding providers."""

import logging

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import ClassVar, overload

from mistralai import Any, cast
from pydantic import BaseModel, ConfigDict

from codeweaver._data_structures import CodeChunk
from codeweaver._server import get_statistics
from codeweaver._settings import Provider
from codeweaver.embedding.capabilities.base import EmbeddingModelCapabilities
from codeweaver.tokenizers import Tokenizer, get_tokenizer


logger = logging.getLogger(__name__)


def default_input_transformer(chunks: Sequence[CodeChunk]) -> Sequence[str]:
    """Default input transformer that serializes CodeChunks to strings."""
    return [chunk.serialize() for chunk in chunks]


def default_output_transformer(output: Any) -> Sequence[Sequence[float]] | Sequence[Sequence[int]]:
    """Default output transformer that ensures the output is in the correct format."""
    if isinstance(output, list | tuple | set) and (
        all(isinstance(i, list | set | tuple) for i in output)
        or (needs_wrapper := all(isinstance(i, int | float) for i in output))  # type: ignore
    ):
        return [output] if needs_wrapper else list(output)  # type: ignore
    logger.error(
        ("Received unexpected output format from embedding provider."),
        extra={"output_data": output},  # pyright: ignore[reportUnknownArgumentType]
    )
    raise ValueError("Unexpected output format from embedding provider.")


class EmbeddingProvider[EmbeddingClient](BaseModel, ABC):
    """
    Abstract class for an embedding provider. You must pass in a client and capabilities.

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
    _caps: EmbeddingModelCapabilities

    _input_transformer: ClassVar[Callable[[Sequence[CodeChunk]], Sequence[str]]] = (
        default_input_transformer
    )
    _output_transformer: ClassVar[
        Callable[[Any], Sequence[Sequence[float]] | Sequence[Sequence[int]]]
    ] = default_output_transformer
    _doc_kwargs: ClassVar[dict[str, Any]] = {}
    _query_kwargs: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        client: EmbeddingClient,
        caps: EmbeddingModelCapabilities,
        kwargs: dict[str, Any] | None,
    ) -> None:
        """Initialize the embedding provider."""
        self._client = client
        self._caps = caps
        if not self._provider:
            self._provider = caps.provider
        self.doc_kwargs = type(self)._doc_kwargs.copy() or {}
        self.query_kwargs = type(self)._query_kwargs.copy() or {}
        self._initialize()
        self._add_kwargs(kwargs or {})
        """Add any user-provided kwargs to the embedding provider, after we merge the defaults together."""

    def _add_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Add keyword arguments to the embedding provider."""
        if not kwargs:
            return
        self.doc_kwargs = {**self.doc_kwargs, **kwargs}
        self.query_kwargs = {**self.query_kwargs, **kwargs}

    @abstractmethod
    def _initialize(self) -> None:
        """Initialize the embedding provider.

        This method is called at the end of __init__ to allow for any additional setup.
        It should minimally set up `_doc_kwargs` and `_query_kwargs` if they are not already set.
        """

    @property
    def name(self) -> Provider:
        """Get the name of the embedding provider."""
        return self._provider

    @property
    @abstractmethod
    def base_url(self) -> str | None:
        """Get the base URL of the embedding provider, if any."""

    @abstractmethod
    async def embed_documents(
        self, documents: list[CodeChunk], **kwargs: dict[str, Any] | None
    ) -> Sequence[Sequence[float]] | Sequence[Sequence[int]]:
        """Embed a list of documents into vectors."""

    @abstractmethod
    async def embed_query(
        self, query: str | Sequence[str], **kwargs: dict[str, Any] | None
    ) -> Sequence[Sequence[float]] | Sequence[Sequence[int]]:
        """Embed a query into a vector."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the size of the vector for the collection."""

    @property
    def client(self) -> EmbeddingClient:
        """Get the client for the embedding provider."""
        return self._client

    @property
    def model_capabilities(self) -> EmbeddingModelCapabilities | None:
        """Get the model capabilities for the embedding provider."""
        return self._caps

    def _tokenizer(self) -> Tokenizer[Any]:
        """Get the tokenizer for the embedding provider."""
        if defined_tokenizer := self._caps.tokenizer:
            return get_tokenizer(defined_tokenizer, self._caps.tokenizer_model or self._caps.name)
        return get_tokenizer("tiktoken", "cl100k_base")

    @property
    def tokenizer(self) -> Tokenizer[Any]:
        """Get the tokenizer for the embedding provider."""
        return self._tokenizer()

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
            statistics.add_token_usage(embedding_generated=token_count)
        elif from_docs and all(isinstance(doc, str) for doc in from_docs):
            token_count = (
                self.tokenizer.estimate_batch(from_docs)  # pyright: ignore[reportArgumentType]
                if all(isinstance(doc, str) for doc in from_docs)
                else sum(self.tokenizer.estimate_batch(item) for item in from_docs)  # type: ignore
            )
            statistics.add_token_usage(embedding_generated=token_count)

    @staticmethod
    def _set_kwargs(
        instance_kwargs: dict[str, Any], passed_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Set keyword arguments for the embedding provider."""
        passed_kwargs = passed_kwargs or {}
        return instance_kwargs | passed_kwargs

    def _process_input(
        self, input_data: CodeChunk | Sequence[CodeChunk] | str | Sequence[str]
    ) -> Sequence[str]:
        """Process input data for embedding."""
        if isinstance(input_data, CodeChunk) or (
            isinstance(input_data, list | tuple | set)
            and all(isinstance(i, CodeChunk) for i in input_data)
        ):
            return self._handle_chunk_input(cast(Sequence[CodeChunk] | CodeChunk, input_data))
        return self._handle_string_input(cast(str | Sequence[str], input_data))

    def _handle_chunk_input(self, input_data: CodeChunk | Sequence[CodeChunk]) -> Sequence[str]:
        """Handle chunk input for embedding."""
        # If input is a single CodeChunk or a list of CodeChunks, serialize them
        preprocessed_input = [input_data] if isinstance(input_data, CodeChunk) else input_data
        return self._input_transformer(preprocessed_input)

    def _handle_string_input(self, input_data: str | Sequence[str]) -> Sequence[str]:
        """Handle string input for embedding."""
        # If input is a single string or a list of strings, return as is
        return [input_data] if isinstance(input_data, str) else input_data

    def _process_output(
        self, output_data: Any
    ) -> Sequence[Sequence[float]] | Sequence[Sequence[int]]:
        """Handle output data from embedding."""
        transformed_output = self._output_transformer(output_data)
        if not isinstance(transformed_output, list | tuple | set) or not all(
            isinstance(i, list | tuple | set) for i in transformed_output
        ):
            logger.error(
                ("Transformed output is not in the expected format."),
                extra={"output_data": output_data},  # pyright: ignore[reportUnknownArgumentType]
            )
            raise ValueError("Transformed output is not in the expected format.")
        return transformed_output
