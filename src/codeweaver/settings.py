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
from typing import Annotated, Any, ClassVar, Literal, LiteralString, Self, Sequence

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
from pydantic_settings import BaseSettings, SettingsConfigDict

from codeweaver._common import BaseEnum
from codeweaver._constants import DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_EXTENSIONS
from codeweaver._utils import walk_down_to_git_root
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
    CHAT = "chat"
    """Provider for chat or agents (e.g. OpenAI or Anthropic)"""

    _UNSET = "unset"
    """A sentinel setting to identify when a `ProviderKind` is not set or is configured."""


class Provider(BaseEnum):
    """Enumeration of available providers."""

    VOYAGE = "voyage"
    QDRANT = "qdrant"

    # from pydantic-ai once tied in:
    # OPENAI = "openai"
    # ANTHROPIC = "anthropic"
    # COHERE = "cohere"
    # GEMINI = "gemini"
    # GROK = "grok"
    # BEDROCK = "bedrock"
    # HUGGINGFACE = "huggingface"

    # OpenAI Compatible with OpenAIModel
    # DEEPSEEK = "deepseek"
    # OLLAMA = "ollama"
    # OPENROUTER = "openrouter"
    # VERCEL = "vercel"
    # PERPLEXITY = "perplexity"
    # FIREWORKS = "fireworks"
    # TOGETHER = "together"
    # AZUREFOUNDRY = "azurefoundry"
    # HEROKU = "heroku"
    # GITHUBMODELS = "githubmodels"

    _UNSET = "unset"

    @classmethod
    def validate(cls, value: str) -> Self:
        """Validate provider-specific settings."""
        with contextlib.suppress(AttributeError, KeyError, ValueError):
            if value_in_self := cls.from_string(value.strip()):
                return value_in_self
        # TODO: We need to allow for dynamic providers in the future, we would check if there's a provider class registered for the value, then register the provider here with `cls.add_member("NEW_PROVIDER", "new_provider")`.
        raise ConfigurationError(f"Invalid provider: {value}")


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

    def __call__(self, *args: Sequence[Any], **kwds: dict[str, Any]) -> type[BaseProvider]:
        """Create a new instance of the provider."""
        return super().__call__(*args: Sequence[Any], **kwds)  # type: ignore

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


class EmbeddingConfig(BaseProvider, AIProviderConfigMixin):
    """Configuration for embedding providers."""

    chunk_size: Annotated[PositiveInt, Field(le=32_768)] = 2_048
    provider_kind: type[ProviderKind] = ProviderKind.EMBEDDING
    embedding_dimension: Annotated[
        Literal[256, 384, 512, 768, 1024, 2048, 3084],
        Field(
            description="Dimension to output vector embeddings. This does not check if it's a valid dimension for the selected model. Please make sure it is if you aren't using default values."
        ),
    ] = 1024
    output_dtype: Annotated[
        Literal["float", "int8", "uint8", "binary", "ubinary"],
        Field(description="Data type for the output embeddings."),
    ] = "float"


class VoyageConfig(EmbeddingConfig):
    """Voyage AI embedding configuration."""

    import os

    provider: type[Provider] = Provider.VOYAGE
    api_key: Annotated[SecretStr | None, Field(description="Voyage API key. Required.")] = (
        SecretStr(os.environ.get("VOYAGE_API_KEY", "")) or None
    )
    use_model: Literal["voyage-code-3", "voyage-3-large", "voyage-3.5"] = "voyage-code-3"
    embedding_dimension = 1024
    output_dtype = "float"


class VectorStoreConfig(BaseProvider):
    """Base configuration for vector store providers."""

    provider: type[Provider] = Provider.QDRANT
    provider_kind: type[ProviderKind] = ProviderKind.VECTOR_STORE
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


QdrantConfig = VectorStoreConfig()


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


class CodeWeaverSettings(BaseSettings):
    """Main configuration model following pydantic-settings patterns.

    Configuration precedence (highest to lowest):
    1. Environment variables (CODEWEAVER_*)
    2. Local config (.codeweaver.local.toml (or .yaml, .yml, .json) in current directory)
    3. Project config (.codeweaver.toml (or .yaml, .yml, .json) in project root)
    4. User config (~/.codeweaver.toml (or .yaml, .yml, .json))
    5. Global config (/etc/codeweaver.toml (or .yaml, .yml, .json))
    6. Defaults
    """

    model_config = SettingsConfigDict(
        env_prefix="CODEWEAVER_",
        env_nested_delimiter="__",
        env_file=(".codeweaver.local.env", ".env", ".codeweaver.env"),
        toml_file=_default_config_file_locations(),
        yaml_file=_default_config_file_locations(as_yaml=True),
        json_file=_default_config_file_locations(as_json=True),
        case_sensitive=False,
        validate_assignment=True,
        cli_kebab_case=True,
        extra="allow",  # Allow extra fields in the configuration for plugins/extensions
    )

    # Core settings
    project_path: Annotated[
        Path,
        Field(
            description="Root path of the codebase to analyze. CodeWeaver will try to detect the project root automatically if you don't provide one."
        ),
    ] = walk_down_to_git_root()

    project_name: Annotated[
        str | None, Field(description="Project name (auto-detected from directory if None)")
    ] = None

    # Performance settings
    token_limit: Annotated[
        PositiveInt, Field(le=130_000, description="Maximum tokens per response")
    ] = 10_000
    max_file_size: Annotated[
        PositiveInt, Field(ge=51_200, description="Maximum file size to process in bytes")
    ] = 1_048_576  # 1 MB
    max_results: Annotated[
        PositiveInt,
        Field(
            le=500,
            description="Maximum code matches to return. Because CodeWeaver primarily indexes ast-nodes, a page can return multiple matches per file, so this is not the same as the number of files returned. This is the maximum number of code matches returned in a single response.",
        ),
    ] = 75

    # File filtering
    excluded_dirs: Annotated[
        frozenset[LiteralString], Field(description="Directories to exclude from analysis")
    ] = DEFAULT_EXCLUDED_DIRS
    excluded_extensions: Annotated[
        frozenset[LiteralString],
        Field(description="File extensions to exclude from search and indexing"),
    ] = DEFAULT_EXCLUDED_EXTENSIONS

    # Provider configuration
    embedding: Annotated[
        tuple[type[EmbeddingConfig], ...], Field(description="Embedding provider configuration")
    ] = (VoyageConfig(),)
    vector_store: Annotated[
        VectorStoreConfig,
        Field(default_factory=QdrantConfig, description="Vector store provider configuration"),
    ] = (QdrantConfig(),)

    # Feature flags
    enable_background_indexing: Annotated[
        bool,
        Field(
            description="Enable automatic background indexing (default behavior and recommended)"
        ),
    ] = True
    enable_telemetry: Annotated[
        bool,
        Field(
            description="Enable privacy-friendly usage telemetry. On by default. We do not collect any identifying information -- we hash all file and directory paths, repository names, and other identifiers to ensure privacy while still gathering useful aggregate data for improving CodeWeaver. You can see exactly what we collect, and how we collect it [here](services/telemetry.py). You can disable this if you prefer not to send any data. You can also provide your own PostHog Project Key to collect your own telemetry data. We will not use this information for anything else -- it is only used to improve CodeWeaver."
        ),
    ] = True
    allow_identifying_telemetry: Annotated[
        bool,
        Field(
            description="DISABLED BY DEFAULT. If you want to *really* help us improve CodeWeaver, you can allow us to collect potentially identifying telemetry data. If it's enabled, we *won't hash file and repository names. We'll still try our best to screen out potential secrets, as well as names and emails, but we can't guarantee complete anonymity. This helps us by giving us real-world usage patterns and information on queries and results. We can use that to make everyone's results better. Like with the default telemetry, we **will not use it for anything else**."
        ),
    ] = False
    enable_ai_intent_analysis: Annotated[
        bool, Field(description="Enable AI-powered intent analysis via FastMCP sampling")
    ] = False  # ! Phase 2 feature, switch to True when implemented

    __version__: Annotated[
        str,
        Field(
            description="Schema version for CodeWeaver settings",
            pattern=r"\d{1,2}\.\d{1,3}\.\d{1,3}",
        ),
    ] = "1.0.0"

    def model_post_init(self, __context: Any, /) -> None:
        """Post-initialization validation."""
        # Ensure project path exists and is readable
        if not self.project_path.exists():
            raise ConfigurationError(
                f"Project path does not exist: {self.project_path}",
                suggestions=[
                    "Check the project_path setting or CODEWEAVER_PROJECT_PATH environment variable or in your configuration file"
                ],
            )

        if not self.project_path.is_dir():
            raise ConfigurationError(
                f"Project path is not a directory: {self.project_path}",
                suggestions=["Ensure project_path points to a directory"],
            )

        if not self.project_name:
            self.project_name = self.project_path.name

    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return self.project_path.resolve()


# Global settings instance
_settings: CodeWeaverSettings | None = None


def get_settings(path: Path | None = None) -> CodeWeaverSettings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = CodeWeaverSettings()
    return _settings


def reload_settings() -> CodeWeaverSettings:
    """Reload settings from configuration sources."""
    global _settings
    _settings = None
    return get_settings()
