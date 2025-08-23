"""Azure embedding provider."""

from __future__ import annotations as _annotations

from pydantic_ai.profiles.cohere import cohere_model_profile

from codeweaver._settings import Provider


try:
    from openai import AsyncAzureOpenAI, AsyncOpenAI

    cohere_model_profile
    Provider
    AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        "Please install the `openai` package to use the Azure provider, "
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class AzureEmbeddingModelProfile[AsyncAzureOpenAI]: ...


class AzureEmbeddingProvider:
    """Azure embedding provider."""

    _client: AsyncAzureOpenAI
