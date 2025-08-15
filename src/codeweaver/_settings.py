# sourcery skip: no-complex-if-expressions
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
# We need to override our generic models with specific types, and type overrides for narrower values is a good thing.
# pyright: reportIncompatibleMethodOverride=false,reportIncompatibleVariableOverride=false
"""Unified configuration system for CodeWeaver.

Provides a centralized settings system using pydantic-settings with
clear precedence hierarchy and validation.
"""

from __future__ import annotations

import os
import platform

from pathlib import Path
from typing import TYPE_CHECKING, Any, LiteralString, NotRequired, Required, TypedDict

from codeweaver._common import BaseEnum
from codeweaver.exceptions import ConfigurationError


if TYPE_CHECKING:
    pass


class AgentModelSettings:
    """Agent model settings stub."""
    ...


class EmbeddingModelSettings:
    """Embedding model settings stub."""
    ...


class RerankModelSettings:
    """Rerank model settings stub."""
    ...


class Provider(BaseEnum):
    """Provider enumeration stub for _settings.py to avoid circular imports."""
    
    VOYAGE = "voyage"
    QDRANT = "qdrant"
    IN_MEMORY = "in_memory"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    MISTRAL = "mistral"
    GOOGLE = "google"
    GROK = "grok"
    BEDROCK = "bedrock"
    HUGGINGFACE = "huggingface"
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    VERCEL = "vercel"
    PERPLEXITY = "perplexity"
    MOONSHOT = "moonshot"
    FIREWORKS = "fireworks"
    TOGETHER = "together"
    AZURE = "azure"
    HEROKU = "heroku"
    GITHUBMODELS = "githubmodels"
    DUCKDUCKGO = "duckduckgo"
    TAVILY = "tavily"
    _UNSET = "unset"


class DataProviderSettings(TypedDict, total=False):
    """Settings for data providers."""

    provider: Required[Provider]
    enabled: Required[bool]
    api_key: NotRequired[LiteralString | None]
    settings: NotRequired[dict[str, Any] | None]


DefaultDataProviderSettings = (
    DataProviderSettings(provider=Provider.TAVILY, enabled=False, settings={"api_key": ""}),
    # DuckDuckGo
    DataProviderSettings(provider=Provider.DUCKDUCKGO, enabled=True, settings=None),
)


class EmbeddingProviderSettings(TypedDict, total=False):
    """Settings for embedding models."""

    provider: Required[Provider]
    model_name: Required[str]
    model_settings: NotRequired[EmbeddingModelSettings | None]
    api_key: NotRequired[LiteralString | None]
    settings: NotRequired[dict[str, Any] | None]


class RerankProviderSettings(TypedDict, total=False):
    """Settings for re-ranking models."""

    provider: Required[Provider]
    model_name: Required[str]
    model_settings: NotRequired[RerankModelSettings | None]
    api_key: NotRequired[LiteralString | None]
    settings: NotRequired[dict[str, Any] | None]


class AgentProviderSettings(TypedDict, total=False):
    """Settings for agent models."""

    provider: Required[Provider]
    model_name: Required[str]
    model_settings: NotRequired[AgentModelSettings | None]
    api_key: NotRequired[LiteralString | None]
    settings: NotRequired[dict[str, Any] | None]


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
