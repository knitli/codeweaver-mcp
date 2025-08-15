# Copyright (c) 2024 to present Pydantic Services Inc
# SPDX-License-Identifier: MIT
# Applies to original code in this directory (`src/codeweaver/embedding_providers/`) from `pydantic_ai`.
#
# SPDX-FileCopyrightText: 2025 (c) 2025 Knitli Inc.
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
# applies to new/modified code in this directory (`src/codeweaver/embedding_providers/`)
# ruff: noqa: S101
"""OpenAI embedding provider."""

from __future__ import annotations as _annotations

import os

from typing import Literal

import httpx

from pydantic import SecretStr
from pydantic_ai.models import cached_async_http_client

from codeweaver.embedding.profiles.openai import openai_model_profile
from codeweaver.embedding.providers import EmbeddingProvider
from codeweaver.embedding.profiles import EmbeddingModelProfile


try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the OpenAI provider, \nyou can use the `openai` optional group — `pip install "codeweaver[openai]"`'
    ) from _import_error


class OpenAIEmbeddingProvider(EmbeddingProvider[AsyncOpenAI]):
    """Provider for OpenAI API."""

    @property
    def name(self) -> Literal["openai"]:
        """OpenAI."""
        return "openai"

    @property
    def base_url(self) -> str:
        """Base URL for OpenAI API."""
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        """The OpenAI async client."""
        return self._client

    def model_profile(self, model_name: str) -> EmbeddingModelProfile | None:
        """Get the model profile for the specified model name."""
        return openai_model_profile(model_name)

    def __init__(
        self,
        base_url: str | None = None,
        api_key: SecretStr | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new OpenAI provider. You can provide either a base URL (defaults to OpenAI's base URL) and an API key, or an instantiated OpenAI client, or a custom HTTP client.

        Args:
            base_url: The base url for the OpenAI requests. If not provided, the `OPENAI_BASE_URL` environment variable
                will be used if available. Otherwise, defaults to OpenAI's base url.
            api_key: The API key to use for authentication, if not provided, the `OPENAI_API_KEY` environment variable
                will be used if available.
            openai_client: An existing
                [`AsyncOpenAI`](https://github.com/openai/openai-python?tab=readme-ov-file#async-usage)
                client to use. If provided, `base_url`, `api_key`, and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        # This is a workaround for the OpenAI client requiring an API key, whilst locally served,
        # openai compatible models do not always need an API key, but a placeholder (non-empty) key is required.
        if (
            api_key is None
            and "OPENAI_API_KEY" not in os.environ
            and base_url is not None
            and openai_client is None
        ):
            api_key = SecretStr("api-key-not-set")

        if openai_client is not None:
            assert base_url is None, "Cannot provide both `openai_client` and `base_url`"
            assert http_client is None, "Cannot provide both `openai_client` and `http_client`"
            assert api_key is None, "Cannot provide both `openai_client` and `api_key`"
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)  # type: ignore
        else:
            http_client = cached_async_http_client(provider="openai")
            self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)  # type: ignore
