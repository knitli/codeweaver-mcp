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

import contextlib
import os
import platform

from pathlib import Path
from typing import Self, cast

from codeweaver._common import BaseEnum
from codeweaver.exceptions import ConfigurationError


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


class Provider(BaseEnum):
    """Enumeration of available providers."""

    VOYAGE = "voyage"  # RR

    QDRANT = "qdrant"

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"  # RR
    MISTRAL = "mistral"
    GOOGLE = "google"
    GROK = "grok"
    BEDROCK = "bedrock"  # RR
    HUGGINGFACE = "huggingface"  # RR

    # OpenAI Compatible with OpenAIModel
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"  # RR
    OPENROUTER = "openrouter"
    VERCEL = "vercel"
    PERPLEXITY = "perplexity"
    MOONSHOT = "moonshot"
    FIREWORKS = "fireworks"
    TOGETHER = "together"
    AZURE = "azure"  # RR
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
        if self == Provider.VOYAGE:
            return cast(
                tuple[ProviderKind, ProviderKind], (ProviderKind.EMBEDDING, ProviderKind.RERANKING)
            )
        if self == Provider.QDRANT:
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
