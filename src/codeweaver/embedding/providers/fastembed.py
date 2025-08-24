# SPDX=FileCopyrightText: 2024-2025 (c) Qdrant Solutions GmBh
# SPDX-LicenseIdentifier: Apache-2.0
# This file is partly derived from code in the `mcp-server-qdrant` project
#
# SPDX-FileCopyrightText: 2025 (c) 2025 Knitli Inc.
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
"""FastEmbed embedding provider implementation.

FastEmbed is a lightweight and efficient library for generating embeddings locally.
"""

import asyncio
import multiprocessing

from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import numpy as np

from codeweaver._data_structures import CodeChunk
from codeweaver._settings import Provider
from codeweaver.embedding.models.base import EmbeddingModelCapabilities
from codeweaver.embedding.providers import EmbeddingProvider


try:
    from fastembed.text import TextEmbedding
except ImportError as e:
    raise ImportError(
        "FastEmbed is not installed. Please install it with `pip install fastembed`."
    ) from e


def fastembed_all_kwargs(**kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Get all possible kwargs for FastEmbed embedding methods."""
    default_kwargs: dict[str, Any] = {"threads": multiprocessing.cpu_count(), "lazy_load": True}
    if kwargs:
        device_ids: list[int] | None = kwargs.get("device_ids")  # pyright: ignore[reportAssignmentType]
        cuda: bool | None = kwargs.get("cuda")  # pyright: ignore[reportAssignmentType]
        if cuda == False:  # user **explicitly** disabled cuda  # noqa: E712
            return default_kwargs | kwargs
        cuda = bool(cuda)
        from codeweaver._utils import decide_fastembed_runtime

        decision = decide_fastembed_runtime(explicit_cuda=cuda, explicit_device_ids=device_ids)
        if isinstance(decision, tuple) and len(decision) == 2:
            cuda = True
            device_ids = decision[1]
        elif decision == "gpu":
            cuda = True
            device_ids = [0]
        else:
            cuda = False
            device_ids = None
        if cuda:
            kwargs["cuda"] = True  # pyright: ignore[reportArgumentType]
            kwargs["device_ids"] = device_ids  # pyright: ignore[reportArgumentType]
            kwargs["providers"] = ["CUDAExecutionProvider"]  # pyright: ignore[reportArgumentType]
    return default_kwargs


def fastembed_output_transformer(
    output: list[np.ndarray],
) -> Sequence[Sequence[float]] | Sequence[Sequence[int]]:
    """Transform the output of FastEmbed into a more usable format."""
    return [emb.tolist() for emb in output]


class FastEmbedProvider(EmbeddingProvider[TextEmbedding]):
    """
    FastEmbed implementation of the embedding provider.

    model_name: The name of the FastEmbed model to use.
    """

    _client: TextEmbedding
    _provider: Provider = Provider.FASTEMBED
    _caps: EmbeddingModelCapabilities

    _doc_kwargs: ClassVar[dict[str, Any]] = fastembed_all_kwargs()
    _query_kwargs: ClassVar[dict[str, Any]] = fastembed_all_kwargs()
    _output_transformer: ClassVar[
        Callable[[Any], Sequence[Sequence[float]] | Sequence[Sequence[int]]]
    ] = fastembed_output_transformer

    def _initialize(self) -> None:
        """Initialize the FastEmbed client."""
        if "model_name" not in self._doc_kwargs:
            model = self._caps.name
            type(self)._doc_kwargs["model_name"] = model
            type(self)._query_kwargs["model_name"] = model
        self._client = TextEmbedding(**self._doc_kwargs)

    @property
    def base_url(self) -> str | None:
        """FastEmbed does not use a base URL."""
        return None

    async def embed_documents(
        self, documents: Sequence[CodeChunk] | CodeChunk, **kwargs: dict[str, Any] | None
    ) -> Sequence[Sequence[float]] | Sequence[Sequence[int]]:
        """Embed a list of documents into vectors."""
        processed_kwargs: dict[str, Any] = self._set_kwargs(self.doc_kwargs, kwargs or {})
        processed_documents: Sequence[str] = self._process_input(documents)
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self._client.passage_embed(processed_documents, **processed_kwargs)
        )
        self._update_token_stats(from_docs=processed_documents)
        return list(self._process_output(embeddings))

    async def embed_query(
        self, query: str | Sequence[str], **kwargs: dict[str, Any] | None
    ) -> Sequence[Sequence[float]] | Sequence[Sequence[int]]:
        """Embed a query into a vector."""
        processed_kwargs: dict[str, Any] = self._set_kwargs(self.query_kwargs, kwargs or {})
        processed_query: Sequence[str] = self._process_input(query)
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self._client.query_embed(processed_query, **processed_kwargs))
        )
        self._update_token_stats(from_docs=processed_query)
        return list(self._process_output(embeddings))

    @property
    def dimension(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        return self._client.embedding_size
