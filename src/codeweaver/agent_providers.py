# Copyright (c) 2024 to present Pydantic Services Inc
# SPDX-License-Identifier: MIT
# Applies to original code in this directory (`src/codeweaver/embedding_providers/`) from `pydantic_ai`.
#
# SPDX-FileCopyrightText: 2025 (c) 2025 Knitli Inc.
# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
# applies to new/modified code in this directory (`src/codeweaver/embedding_providers/`)
"""This module re-exports agentic model providers and associated utilities from Pydantic AI."""

from typing import Any, LiteralString, TypeVar

from pydantic_ai.providers import Provider as AgentProvider
from pydantic_ai.toolsets import (
    AbstractToolset,
    CombinedToolset,
    DeferredToolset,
    FilteredToolset,
    FunctionToolset,
    PrefixedToolset,
    PreparedToolset,
    RenamedToolset,
    ToolsetTool,
    WrapperToolset,
)

from codeweaver.providers import Provider as ProviderEnum


InterfaceClient = TypeVar("InterfaceClient", bound="AgentProvider[Any]")


def get_agent_model_provider(provider: ProviderEnum) -> type[AgentProvider[InterfaceClient]]:  # type: ignore  # noqa: C901
    # sourcery skip: low-code-quality, no-long-functions
    """Get the agent model provider."""
    if provider == ProviderEnum.OPENAI:
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIProvider
    if provider == ProviderEnum.DEEPSEEK:
        from pydantic_ai.providers.deepseek import DeepSeekProvider

        return DeepSeekProvider
    if provider == ProviderEnum.OPENROUTER:
        from pydantic_ai.providers.openrouter import OpenRouterProvider

        return OpenRouterProvider
    if provider == ProviderEnum.VERCEL:
        from pydantic_ai.providers.vercel import VercelProvider

        return VercelProvider
    if provider == ProviderEnum.AZURE:
        from pydantic_ai.providers.azure import AzureProvider

        return AzureProvider

    # NOTE: We don't test for auth because there are many ways the `boto3.client` can retrieve the credentials.
    if provider == ProviderEnum.BEDROCK:
        from pydantic_ai.providers.bedrock import BedrockProvider

        return BedrockProvider
    if provider == ProviderEnum.GOOGLE:
        from pydantic_ai.providers.google import GoogleProvider

        return GoogleProvider
    if provider == ProviderEnum.GROK:
        from pydantic_ai.providers.grok import GrokProvider

        return GrokProvider
    if provider == ProviderEnum.ANTHROPIC:
        from pydantic_ai.providers.anthropic import AnthropicProvider

        return AnthropicProvider
    if provider == ProviderEnum.MISTRAL:
        from pydantic_ai.providers.mistral import MistralProvider

        return MistralProvider
    if provider == ProviderEnum.COHERE:
        from pydantic_ai.providers.cohere import CohereProvider

        return CohereProvider
    if provider == ProviderEnum.MOONSHOTAI:
        from pydantic_ai.providers.moonshotai import MoonshotAIProvider

        return MoonshotAIProvider
    if provider == ProviderEnum.FIREWORKS:
        from pydantic_ai.providers.fireworks import FireworksProvider

        return FireworksProvider
    if provider == ProviderEnum.TOGETHER:
        from pydantic_ai.providers.together import TogetherProvider

        return TogetherProvider
    if provider == ProviderEnum.HEROKU:
        from pydantic_ai.providers.heroku import HerokuProvider

        return HerokuProvider
    if provider == ProviderEnum.HUGGINGFACE:
        from pydantic_ai.providers.huggingface import HuggingFaceProvider

        return HuggingFaceProvider
    if provider == ProviderEnum.GITHUB:
        from pydantic_ai.providers.github import GitHubProvider

        return GitHubProvider
    # pragma: no cover
    raise ValueError(f"Unknown provider: {provider}")


def infer_agent_provider_class(
    provider: LiteralString | ProviderEnum,
) -> AgentProvider[InterfaceClient]:  # type: ignore
    """Infer the provider from the provider name."""
    if not isinstance(provider, ProviderEnum):
        provider = ProviderEnum.from_string(provider)
    provider_class: type[AgentProvider[InterfaceClient]] = get_agent_model_provider(provider)  # type: ignore
    return provider_class()


__all__ = (
    "AbstractToolset",
    "AgentProvider",
    "CombinedToolset",
    "DeferredToolset",
    "FilteredToolset",
    "FunctionToolset",
    "PrefixedToolset",
    "PreparedToolset",
    "RenamedToolset",
    "ToolsetTool",
    "WrapperToolset",
    "get_agent_model_provider",
    "infer_agent_provider_class",
)
