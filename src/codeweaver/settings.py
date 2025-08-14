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

from pathlib import Path
from typing import Annotated, Any, LiteralString, TypedDict

from pydantic import Field, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict

from codeweaver._constants import DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_EXTENSIONS
from codeweaver._settings import Provider, default_config_file_locations
from codeweaver._utils import walk_down_to_git_root
from codeweaver.exceptions import ConfigurationError


class ProviderModelConfig(TypedDict, total=False):
    """
    Configuration for an AI provider model (chat/agent, embedding, reranking, etc.).

    If a field is not applicable to the model, it should be set to None. If it can be configured, set it to a default value, and mark it as `configurable: True` in the JSON schema extra.
    """

    model_name: Provider
    variants: Annotated[
        tuple[str, ...] | None,
        Field(
            description="List of model variants. These are other official names for the model, typically representing versions or training dates like `gpt-5-2025-08-07`."
        ),
    ]
    max_tokens: Annotated[
        int,
        Field(description="For ingest. Maximum number of tokens the model can receive in a call."),
    ]
    n_completions: Annotated[
        int | None,
        Field(
            description="Number of completions to generate for each prompt, if configurable.",
            json_schema_extra={"configurable": False},
        ),
    ]
    temperature: Annotated[
        float | None,
        Field(
            description="Sampling temperature for the model's responses, if configurable.",
            json_schema_extra={"configurable": False},
        ),
    ]
    top_p: Annotated[
        float | None,
        Field(
            description="Nucleus sampling parameter for the model's responses, if configurable.",
            json_schema_extra={"configurable": False},
        ),
    ]
    presence_penalty: Annotated[
        float | None,
        Field(
            description="Presence penalty for the model's responses, if configurable.",
            json_schema_extra={"configurable": False},
        ),
    ]
    frequency_penalty: Annotated[
        float | None,
        Field(
            description="Frequency penalty for the model's responses, if configurable.",
            json_schema_extra={"configurable": False},
        ),
    ]


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

    # File filtering
    excluded_dirs: Annotated[
        frozenset[LiteralString], Field(description="Directories to exclude from analysis")
    ] = DEFAULT_EXCLUDED_DIRS
    excluded_extensions: Annotated[
        frozenset[LiteralString],
        Field(description="File extensions to exclude from search and indexing"),
    ] = DEFAULT_EXCLUDED_EXTENSIONS
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
