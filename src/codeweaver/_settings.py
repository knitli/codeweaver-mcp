# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
# We need to override our generic models with specific types, and type overrides for narrower values is a good thing.
# pyright: reportIncompatibleMethodOverride=false,reportIncompatibleVariableOverride=false
"""Core settings and provider definitions."""

from __future__ import annotations

import contextlib
import os
import platform

from pathlib import Path
from typing import Any, LiteralString, NotRequired, Required, Self, TypedDict

from pydantic_ai.settings import ModelSettings as AgentModelSettings

from codeweaver._common import BaseEnum
from codeweaver.exceptions import ConfigurationError


class BaseProviderSettings(TypedDict, total=False):
    """Base settings for all providers."""

    provider: Required[Provider]
    enabled: Required[bool]
    api_key: NotRequired[LiteralString | None]
    extra: NotRequired[dict[str, Any] | None]


class DataProviderSettings(BaseProviderSettings):
    """Settings for data providers."""


class EmbeddingModelSettings:
    """Embedding model settings stub."""


class RerankModelSettings:
    """Rerank model settings stub."""


class EmbeddingProviderSettings(BaseProviderSettings):
    """Settings for embedding models."""

    model: Required[str]
    model_settings: NotRequired[EmbeddingModelSettings | None]


class RerankProviderSettings(BaseProviderSettings):
    """Settings for re-ranking models."""

    models: Required[str | tuple[str, ...]]  # Tuple of model names
    """A model name or a tuple of model names to use for re-ranking in order of preference."""
    model_settings: NotRequired[RerankModelSettings | tuple[RerankModelSettings, ...] | None]
    """Settings for the re-ranking model(s)."""
    extra: NotRequired[dict[str, Any] | None]


class AgentProviderSettings(BaseProviderSettings):
    """Settings for agent models."""

    models: Required[str | tuple[str, ...]]
    """A model name or a tuple of model names to use for agent in order of preference."""
    model_settings: NotRequired[AgentModelSettings | tuple[AgentModelSettings, ...] | None]
    """Settings for the agent model(s)."""


class Provider(BaseEnum):
    """Enumeration of available providers."""

    VOYAGE = "voyage"
    FASTEMBED = "fastembed"

    QDRANT = "qdrant"
    FASTEMBED_VECTORSTORE = "fastembed_vectorstore"

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
    OLLAMA = "ollama"  # supports rerank, but not w/ OpenAI API
    OPENROUTER = "openrouter"
    VERCEL = "vercel"
    PERPLEXITY = "perplexity"
    MOONSHOT = "moonshot"
    FIREWORKS = "fireworks"
    TOGETHER = "together"
    AZURE = "azure"  # supports rerank, but not w/ OpenAI API
    HEROKU = "heroku"
    GITHUB = "github"

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


class ProviderKind(BaseEnum):
    """Enumeration of available provider kinds."""

    DATA = "data"
    """Provider for data retrieval and processing (e.g. Tavily)"""
    EMBEDDING = "embedding"
    """Provider for text embedding (e.g. Voyage)"""
    RERANKING = "reranking"
    """Provider for re-ranking (e.g. Voyage)"""
    VECTOR_STORE = "vector_store"
    """Provider for vector storage (e.g. Qdrant)"""
    AGENT = "agent"
    """Provider for agents (e.g. OpenAI or Anthropic)"""

    _UNSET = "unset"
    """A sentinel setting to identify when a `ProviderKind` is not set or is configured."""

    @property
    def settings_object(self) -> object:
        """Get the settings object for this provider kind."""
        if self == ProviderKind.DATA:
            return DataProviderSettings
        if self == ProviderKind.EMBEDDING:
            return EmbeddingProviderSettings
        if self == ProviderKind.RERANKING:
            return RerankProviderSettings
        if self == ProviderKind.AGENT:
            return AgentProviderSettings
        raise ConfigurationError(f"ProviderKind {self} does not have a settings object.")


def default_config_file_locations(
    *, as_yaml: bool = False, as_json: bool = False
) -> tuple[str, ...]:
    """Get default file locations for configuration files."""
    # Determine base extensions
    extensions = (
        ["yaml", "yml"] if not as_yaml and not as_json else ["yaml", "yml"] if as_yaml else ["json"]
    )
    # Get user config directory
    user_config_dir = (
        os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        if platform.system() == "Windows"
        else os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    )

    # Build file paths maintaining precedence order
    base_paths = [
        (Path.cwd(), ".codeweaver.local"),
        (Path.cwd(), ".codeweaver"),
        (Path(user_config_dir) / "codeweaver", "settings"),
    ]

    # Generate all file paths using list comprehension
    file_paths = [
        str(base_dir / f"{filename}.{ext}")
        for base_dir, filename in base_paths
        for ext in extensions
    ]

    return tuple(file_paths)
