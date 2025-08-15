"""Providers of all kinds, including vector stores, agents, embeddings, and rerankers."""

from __future__ import annotations

import contextlib

from types import FunctionType
from typing import TYPE_CHECKING, Self, cast

from codeweaver._common import BaseEnum


if TYPE_CHECKING:
    from codeweaver._settings import ProviderKind
    from codeweaver.exceptions import ConfigurationError


class Provider(BaseEnum):
    """Enumeration of available providers."""

    VOYAGE = "voyage"

    QDRANT = "qdrant"
    IN_MEMORY = "in_memory"  # Special case for in-memory vector stores

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    MISTRAL = "mistral"
    GOOGLE = "google"
    GROK = "grok"
    BEDROCK = "bedrock"
    HUGGINGFACE = "huggingface"

    # OpenAI Compatible with OpenAIModel
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"  # supports rerank, but not on OpenAI API
    OPENROUTER = "openrouter"
    VERCEL = "vercel"
    PERPLEXITY = "perplexity"
    MOONSHOT = "moonshot"
    FIREWORKS = "fireworks"
    TOGETHER = "together"
    AZURE = "azure"  # supports rerank, but not on OpenAI API
    HEROKU = "heroku"
    GITHUBMODELS = "githubmodels"

    DUCKDUCKGO = "duckduckgo"
    TAVILY = "tavily"

    _UNSET = "unset"

    @classmethod
    def validate(cls, value: str) -> Self:
        """Validate provider-specific settings."""
        with contextlib.suppress(AttributeError, KeyError, ValueError):
            if value_in_self := cls.from_string(value.strip()):
                return value_in_self
        # TODO: We need to allow for dynamic providers in the future, we would check if there's a provider class registered for the value, then register the provider here with `cls.add_member("NEW_PROVIDER", "new_provider")`.
        raise ConfigurationError(f"Invalid provider: {value}")

    @property
    def kinds(self) -> tuple[ProviderKind, ...]:
        """Get the kinds of this provider."""
        # NOTE: Azure and Ollama both *also* support reranking,
        # But our support for them currently is through OpenAI's API,
        # which doesn't expose reranking capabilities (because OpenAI doesn't have a reranking model).
        from codeweaver._settings import ProviderKind

        if self == Provider.VOYAGE:
            return cast(
                tuple[ProviderKind, ProviderKind], (ProviderKind.EMBEDDING, ProviderKind.RERANKING)
            )
        if self in (Provider.QDRANT, Provider.IN_MEMORY):
            return cast(tuple[ProviderKind], (ProviderKind.VECTOR_STORE,))
        if self in {Provider.COHERE, Provider.BEDROCK, Provider.HUGGINGFACE}:
            return cast(
                tuple[ProviderKind, ...],
                (ProviderKind.EMBEDDING, ProviderKind.RERANKING, ProviderKind.AGENT),
            )
        if self in {
            Provider.AZURE,
            Provider.FIREWORKS,
            Provider.GOOGLE,
            Provider.GITHUBMODELS,
            Provider.MISTRAL,
            Provider.HEROKU,
            Provider.OLLAMA,
            Provider.OPENAI,
            Provider.TOGETHER,
            Provider.VERCEL,
        }:
            return cast(
                tuple[ProviderKind, ProviderKind], (ProviderKind.AGENT, ProviderKind.EMBEDDING)
            )
        if self in {
            Provider.ANTHROPIC,
            Provider.DEEPSEEK,
            Provider.OPENROUTER,
            Provider.PERPLEXITY,
            Provider.MOONSHOT,
            Provider.GROK,
        }:
            return cast(tuple[ProviderKind], (ProviderKind.AGENT,))
        if self == Provider._UNSET:
            return cast(tuple[ProviderKind], (ProviderKind._UNSET,))  # type: ignore
        return cast(tuple[ProviderKind], (ProviderKind.DATA,))

    def get_handlers(self) -> tuple[FunctionType | type[object], ...]:
        """Get the handler function for this provider."""
        match self:
            case Provider._UNSET:
                raise ConfigurationError("Provider is not set.")
            case Provider.TAVILY:
                from codeweaver.tools import tavily_tool

                return (tavily_tool.TavilySearchTool,)
            case Provider.DUCKDUCKGO:
                from codeweaver.tools import duckduckgo_tool

                return (duckduckgo_tool.DuckDuckGoSearchTool,)
            case Provider.QDRANT:
                from codeweaver.vector_stores import get_store

                return (get_store,)
            case _:
                return self._get_all_handlers()

    def _get_all_handlers(self) -> tuple[FunctionType | type[object], ...]:
        """Get all handler functions for this provider."""
        handlers: list[FunctionType | type[object]] = []
        kinds = tuple(str(kind) for kind in self.kinds)
        if "agent" in kinds:
            from codeweaver.agent_providers import infer_agent_provider_class

            handlers.append(infer_agent_provider_class)
        if "embedding" in kinds:
            from codeweaver.embedding import get_embedding_model_provider

            handlers.append(get_embedding_model_provider)
        if "reranking" in kinds:
            from codeweaver.embedding import get_rerank_model_provider

            handlers.append(get_rerank_model_provider)
        if "vector_store" in kinds:
            from codeweaver.vector_stores import get_store

            handlers.append(get_store)
        return tuple(handlers)
