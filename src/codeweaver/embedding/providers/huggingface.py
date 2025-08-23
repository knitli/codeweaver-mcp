# Copyright (c) 2024 to present Pydantic Services Inc
# SPDX-License-Identifier: MIT
# Applies to original code in this directory (`src/codeweaver/embedding_providers/`) from `pydantic_ai`.
#
# SPDX-FileCopyrightText: 2025 (c) 2025 Knitli Inc.
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
# applies to new/modified code in this directory (`src/codeweaver/embedding_providers/`)
"""HuggingFace embedding provider."""

import logging

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from codeweaver._data_structures import CodeChunk
from codeweaver._settings import Provider
from codeweaver.embedding.models.base import EmbeddingModelCapabilities
from codeweaver.embedding.providers.base import EmbeddingProvider


logger = logging.getLogger(__name__)


def huggingface_hub_input_transformer(chunks: Sequence[CodeChunk]) -> Sequence[str]:
    """Input transformer for Hugging Face Hub models."""
    # The hub client only takes a single string at a time, so we'll just use a generator here
    return [chunk.serialize() for chunk in chunks]


def huggingface_hub_output_transformer(
    output: Iterator[np.ndarray],
) -> Sequence[Sequence[float]] | Sequence[Sequence[int]]:
    """Output transformer for Hugging Face Hub models."""
    return [out.tolist() for out in output]


def huggingface_hub_embed_kwargs(**kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keyword arguments for Hugging Face Hub models."""
    kwargs = kwargs or {}
    return {"normalize": True, "prompt_name": "passage", **kwargs}


def huggingface_hub_query_kwargs(**kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keyword arguments for the query embedding method."""
    kwargs = kwargs or {}
    return {"normalize": True, "prompt_name": "query", **kwargs}


try:
    from huggingface_hub import AsyncInferenceClient

except ImportError as e:
    logger.debug("HuggingFace Hub is not installed.")
    raise ImportError(
        'Please install the `huggingface_hub` package to use the HuggingFace provider, you can use the `huggingface` optional group — `pip install "codeweaver[huggingface]"`'
    ) from e


class HuggingFaceEmbeddingProvider(EmbeddingProvider[AsyncInferenceClient]):
    """HuggingFace embedding provider."""

    _client: AsyncInferenceClient
    _provider: Provider = Provider.HUGGINGFACE
    _caps: EmbeddingModelCapabilities

    _input_transformer = staticmethod(huggingface_hub_input_transformer)
    _output_transformer = staticmethod(huggingface_hub_output_transformer)
    _doc_kwargs = huggingface_hub_embed_kwargs()
    _query_kwargs = huggingface_hub_query_kwargs()

    def _initialize(self) -> None:
        """We don't need to do anything here."""
        type(self)._doc_kwargs |= {"model": self._caps.name}
        type(self)._query_kwargs |= {"model": self._caps.name}

    @property
    def base_url(self) -> str | None:
        """Get the base URL of the embedding provider."""
        return "https://api.huggingface.co/"

    async def _embed_sequence(
        self, sequence: Sequence[str], **kwargs: dict[str, Any]
    ) -> Sequence[Sequence[float]] | Sequence[Sequence[int]]:
        """Embed a sequence of strings into vectors."""
        all_output: Sequence[Sequence[float]] | Sequence[Sequence[int]] = []
        for doc in sequence:
            output = await self._client.feature_extraction(doc, **kwargs)  # type: ignore
            all_output.append(output)  # type: ignore
        return all_output

    async def embed_documents(
        self, documents: list[CodeChunk], **kwargs: dict[str, Any] | None
    ) -> Sequence[Sequence[float]] | Sequence[Sequence[int]]:
        """Embed a list of documents into vectors."""
        processed_kwargs = self._set_kwargs(self.doc_kwargs, kwargs)
        transformed_input = self._process_input(documents)
        all_output = await self._embed_sequence(transformed_input, **processed_kwargs)
        self._update_token_stats(from_docs=transformed_input)
        return self._process_output(all_output)

    async def embed_query(
        self, query: str | Sequence[str], **kwargs: dict[str, Any] | None
    ) -> Sequence[Sequence[float]] | Sequence[Sequence[int]]:
        """Embed a query into a vector."""
        processed_kwargs = self._set_kwargs(self.query_kwargs, kwargs)
        if isinstance(query, str):
            query = [query]
        output = await self._embed_sequence(query, **processed_kwargs)
        self._update_token_stats(from_docs=query)
        return self._process_output(output)

    @property
    def dimension(self) -> int:
        """Get the size of the vector for the collection."""
        return self._caps.default_dimension or 1024
