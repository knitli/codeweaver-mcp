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

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, LiteralString

from pydantic import BaseModel, Field, PositiveInt
from pydantic_ai.settings import ModelSettings as AgentModelSettings
from pydantic_ai.settings import merge_model_settings
from pydantic_settings import BaseSettings, SettingsConfigDict

from codeweaver._constants import DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_EXTENSIONS
from codeweaver._settings import Provider, default_config_file_locations
from codeweaver._utils import walk_down_to_git_root
from codeweaver.exceptions import ConfigurationError, MissingValueError


if TYPE_CHECKING:
    from codeweaver._settings import DataProviderSettings, ProviderKind


DefaultDataProviderSettings = (
    DataProviderSettings(provider=Provider.TAVILY, enabled=False, api_key=None, extra=None),
    # DuckDuckGo
    DataProviderSettings(provider=Provider.DUCKDUCKGO, enabled=True, api_key=None, extra=None),
)


def merge_agent_model_settings(
    base: AgentModelSettings | None, override: AgentModelSettings | None
) -> AgentModelSettings | None:
    """A convenience re-export of `merge_model_settings` for agent model settings."""
    return merge_model_settings(base, override)


class FileFilterSettings(BaseModel):
    """Settings for file filtering."""

    excluded_dirs: Annotated[
        frozenset[LiteralString], Field(description="Directories to exclude from analysis")
    ] = DEFAULT_EXCLUDED_DIRS
    excluded_extensions: Annotated[
        frozenset[LiteralString],
        Field(description="File extensions to exclude from search and indexing"),
    ] = DEFAULT_EXCLUDED_EXTENSIONS
    use_gitignore: Annotated[bool, Field(description="Whether to use .gitignore for filtering")] = (
        True
    )
    use_other_ignore_files: Annotated[
        bool,
        Field(
            description="Whether to read *other* ignore files (besides .gitignore) for filtering"
        ),
    ] = False
    ignore_hidden: Annotated[
        bool, Field(description="Whether to ignore hidden files (starting with .) for filtering")
    ] = True


class ProviderSettings(BaseModel):
    """Settings for provider configuration."""

    data: Annotated[
        tuple[DataProviderSettings, ...] | None, Field(description="Data provider configuration")
    ] = DefaultDataProviderSettings

    """ COMMENTED OUT WHILE IMPLEMENTING...
    embedding: Annotated[
        tuple[EmbeddingProviderSettings, ...], Field(description="Embedding provider configuration")
    ] = (VoyageEmbeddingProviderSettings(),)

    vector: Annotated[
        tuple[BaseVectorStoreConfig, ...],
        Field(default_factory=QdrantVectorStore, description="Vector store provider configuration"),
    ] = QdrantConfig()

    agent: Annotated[
        tuple[AgentProviderSettings, ...] | None, Field(description="Agent provider configuration")
    ] = (DefaultAgentProviderSettings(),) = None
    """


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
        toml_file=default_config_file_locations(),
        yaml_file=default_config_file_locations(as_yaml=True),
        json_file=default_config_file_locations(as_json=True),
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

    filter_settings: Annotated[FileFilterSettings, Field(description="File filtering settings")] = (
        FileFilterSettings()
    )

    """ # Disabled while implementing...
    # Provider configuration
    embedding: Annotated[
        tuple[EmbeddingConfig, ...], Field(description="Embedding provider configuration")
    ] = (VoyageConfig(),)
    vector_store: Annotated[
        BaseVectorStoreConfig,
        Field(default_factory=QdrantConfig, description="Vector store provider configuration"),
    ] = QdrantConfig()
    """
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
            description="DISABLED BY DEFAULT. If you want to *really* help us improve CodeWeaver, you can allow us to collect potentially identifying telemetry data. It's not intrusive, it's more like what *most* telemetry collects. If it's enabled, we *won't hash file and repository names. We'll still try our best to screen out potential secrets, as well as names and emails, but we can't guarantee complete anonymity. This helps us by giving us real-world usage patterns and information on queries and results. We can use that to make everyone's results better. Like with the default telemetry, we **will not use it for anything else**."
        ),
    ] = False
    enable_ai_intent_analysis: Annotated[
        bool, Field(description="Enable AI-powered intent analysis via FastMCP sampling")
    ] = False  # ! Phase 2 feature, switch to True when implemented
    enable_precontext: Annotated[
        bool,
        Field(
            description="Enable precontext code generation. Recommended, but requires you set up an agent model. This allows CodeWeaver to call an agent model outside of an MCP tool request (it still requires either a CLI call from you or a hook you setup). This is required for our recommended *precontext workflow*. This setting dictionary is a `pydantic_ai.settings.ModelSettings` object. If you already use `pydantic_ai.settings.ModelSettings`, then you can provide the same settings here."
        ),
    ] = False  # ! Phase 2 feature, switch to True when implemented

    agent_settings: Annotated[
        AgentModelSettings | None,
        Field(description="Model settings for ai agents. Required for `enable_precontext`"),
    ] = None

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
        if not self.project_path:
            self.project_path = walk_down_to_git_root()
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


def get_provider_settings(provider_kind: ProviderKind | LiteralString) -> Any:
    """Check a setting value by a tuple of keys (the path to the setting)."""
    if isinstance(provider_kind, str):
        provider_kind = ProviderKind.from_string(provider_kind)
    if provider_kind == ProviderKind._UNSET:  # type: ignore
        raise MissingValueError(
            "Provider kind cannot be _UNSET",
            "settings.get_provider_settings: `provider_kind` is _UNSET",
            None,
            ["This may be a bug in CodeWeaver, please report it."],
        )
