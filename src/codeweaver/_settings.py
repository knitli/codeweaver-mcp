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

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, LiteralString, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    HttpUrl,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    model_validator,
)
from pydantic.dataclasses import dataclass

from codeweaver._common import BaseEnum
from codeweaver.exceptions import ConfigurationError


@dataclass(
    frozen=True,
    order=True,
    kw_only=True,
    config=ConfigDict(extra="allow", str_strip_whitespace=True),
)
class ProviderAIModel:
    """Configuration for an AI provider model (chat/agent, embedding, reranking, etc.). Most fields are optional."""

    model_name: LiteralString
    variants: Annotated[
        tuple[str, ...] | None,
        Field(
            description="List of model variants. These are other official names for the model, typically representing versions or training dates like `gpt-5-2025-08-07`."
        ),
    ] = None
    max_tokens: Annotated[
        int | None,
        Field(description="For ingest. Maximum number of tokens the model can receive in a call."),
    ] = None
    temperature: Annotated[
        float | None,
        Field(description="Sampling temperature for the model's responses, if configurable."),
    ] = None
    top_p: Annotated[
        float | None,
        Field(description="Nucleus sampling parameter for the model's responses, if configurable."),
    ] = None
    tokenizer: Annotated[
        LiteralString | None, Field(description="Tokenizer library used by the model.")
    ] = None
    token_encoding: Annotated[
        LiteralString | None, Field(description="Token encoding used by the model.")
    ] = None


@dataclass(
    frozen=True,
    order=True,
    kw_only=True,
    config=ConfigDict(extra="allow", str_strip_whitespace=True),
)
class AiProviderSettings(BaseModel):
    models: Annotated[
        tuple[ProviderAIModel, ...], Field(description="List of models available for the provider.")
    ] = ()
    url: Annotated[
        HttpUrl | None,
        Field(
            description="Base URL for the provider API. Not required for all providers, such as local providers."
        ),
    ] = None
    client = Annotated[
        LiteralString | None,
        Field(
            description="The name of the client application to use with the provider. This should be a full **python path** to the client.",
            examples=["voyageai.AsyncClient"],
        ),
    ] = None


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
    CHAT = "chat"
    """Provider for chat or agents (e.g. OpenAI or Anthropic)"""

    _UNSET = "unset"
    """A sentinel setting to identify when a `ProviderKind` is not set or is configured."""


class Provider(BaseEnum):
    """Enumeration of available providers."""

    VOYAGE = "voyage"
    QDRANT = "qdrant"

    # from pydantic-ai once tied in:
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    GEMINI = "gemini"
    GROK = "grok"
    AWSBEDROCK = "awsbedrock"
    HUGGINGFACE = "huggingface"

    # OpenAI Compatible with OpenAIModel
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    VERCEL = "vercel"
    PERPLEXITY = "perplexity"
    FIREWORKS = "fireworks"
    TOGETHER = "together"
    AZUREFOUNDRY = "azurefoundry"
    HEROKU = "heroku"
    GITHUBMODELS = "githubmodels"

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
    def kind(self) -> tuple[ProviderKind]:
        match self:
            case Provider.VOYAGE:
                return (ProviderKind.EMBEDDING, ProviderKind.RERANKING)
            case Provider.QDRANT:
                return (ProviderKind.VECTOR_STORE,)
            case Provider.COHERE:
                return (ProviderKind.CHAT, ProviderKind.EMBEDDING, ProviderKind.RERANKING)


class BaseProvider(BaseModel):
    """Base class for all providers."""

    model_config = ConfigDict(use_enum_values=True, extra="allow")

    provider: Annotated[
        type[Provider] | None,
        Field(description="Name of the provider."),
        BeforeValidator(Provider.validate),
    ] = None
    provider_kind: Annotated[
        type[ProviderKind] | None, Field(description="Kind of the provider.")
    ] = None

    batch_size: Annotated[PositiveInt, Field(le=128)] = 32
    timeout_seconds: Annotated[PositiveFloat, Field(gt=3.0)] = 30.0

    api_key: Annotated[
        SecretStr | None, Field(description="API key for the provider", repr=False, exclude=True)
    ] = None
    _api_key_required: Annotated[
        bool, Field(description="Whether the provider requires an API key")
    ] = False

    @model_validator(mode="after")
    def _instance_validate(self) -> Self:
        """Validate instance-level configuration. Don't override this method. Use `validate` instead."""
        if self._api_key_required and not self.api_key:
            raise ConfigurationError(
                f"You must provide an API key for {self.provider!s}.",
                suggestions=[
                    f"You can set the API key in a CodeWeaver configuration file or as an environment variable. To set an environment variable, use `CODEWEAVER_{self.provider_kind.name.upper() if self.provider_kind else 'PROVIDER_KIND'}__{self.provider.name.upper() if self.provider else 'PROVIDER'}__API_KEY=your_api_key`."
                ],
            )
        return self

    def __call__(self, *args: Sequence[Any], **kwds: dict[str, Any]) -> BaseProvider:
        """Create a new instance of the provider."""
        return super().model_validate(*args, **kwds)

    @model_validator(mode="after")
    def validate(self) -> Self:
        """
        Optional validator to implement provider-specific validation.

        Pydantic will call this method *after* the model is initialized and all fields are validated.

        If you want to implement provider-specific validation, such as two settings that cannot be used together, you can override this method in your subclass.

        The method must return `self` to maintain the chain of validation, or raise an error if validation fails.
        """
        return self

    @property
    def extra(self) -> dict[str, Any]:
        """Return extra configuration options. Use this to pass additional settings to a custom provider, or to pass settings to the provider's API that are not defined in the model."""
        return self.__pydantic_extra__ or {}


class AIProviderConfigMixin:
    """Mixin for AI provider configuration."""

    use_model: Annotated[LiteralString, Field(description="Model to use for the provider.")] = (
        "default"
    )
    tokenizer: Annotated[
        Literal["tiktoken", "voyage"] | None, Field(description="Tokenizer to use.")
    ] = "tiktoken"
    tokenizer_encoder: Annotated[LiteralString, Field(description="Tokenizer encoder to use.")] = (
        "c1100k_base"
    )
    ignore_model_checks: Annotated[
        bool,
        Field(
            description="Ignore model validation checks. You can use this to allow a model that is valid for the provider, but not yet in CodeWeaver's list of valid models."
        ),
    ] = False
    valid_models: ClassVar[tuple[LiteralString, ...]] = ("default",)


class BaseAIProviderConfig(BaseProvider, AIProviderConfigMixin):
    """Base configuration for any model-centric provider. Think of it as anything that uses tokens and has models that consume them."""


class BaseEmbeddingConfig(BaseAIProviderConfig):
    """Configuration for embedding providers."""

    chunk_size: Annotated[PositiveInt, Field(le=32_768)] = 2_048
    provider_kind: type[ProviderKind] | LiteralString = ProviderKind.EMBEDDING
    embedding_dimension: Annotated[
        Literal[256, 384, 512, 768, 1024, 2048, 3084],
        Field(
            description="Dimension for vector output. This does not check if it's a valid dimension for the selected model. Please make sure it is if you aren't using default values."
        ),
    ] = 1024
    output_dtype: Annotated[
        Literal["float", "int8", "uint8", "binary", "ubinary"],
        Field(description="Data type for the output embeddings."),
    ] = "float"


class VoyageConfig(BaseEmbeddingConfig):
    """Voyage AI embedding configuration."""

    import os

    provider: type[Provider] | LiteralString = Provider.VOYAGE
    api_key: Annotated[SecretStr | None, Field(description="Voyage API key. Required.")] = (
        SecretStr(os.environ.get("VOYAGE_API_KEY", "")) or None
    )
    use_model: Literal["voyage-code-3", "voyage-3-large", "voyage-3.5"] = "voyage-code-3"
    embedding_dimension = 1024
    output_dtype = "float"


class BaseVectorStoreConfig(BaseProvider):
    """Base configuration for vector store providers."""

    provider: type[Provider] | LiteralString = Provider.QDRANT
    provider_kind: type[ProviderKind] | LiteralString = ProviderKind.VECTOR_STORE
    collection_name: Annotated[
        str | None,
        Field(
            description="Collection name for the vector store. Defaults to the name of the repository."
        ),
    ] = None
    stored_dimension: Annotated[
        PositiveInt, Field(description="Dimension of the stored vectors.")
    ] = 1024
    # For the default providers, Voyage AI's embeddings are normalized to length 1
    # So the results of dot-product *are* cosine similarity, but can be computed much faster.
    # It also means euclidean would produce the same results as cosine similarity.
    distance_metric: Annotated[
        Literal["cosine", "euclidean", "dot-product", "hamming", "jaccard"],
        Field(description="Distance metric to use for vector similarity."),
    ] = "dot-product"
    url: Annotated[
        HttpUrl, Field(description="URL for the vector store. May be local or remote.")
    ] = "http://localhost:6333"
    batch_size: Annotated[PositiveInt, Field(ge=10, le=1000)] = 100


def _default_config_file_locations(
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
