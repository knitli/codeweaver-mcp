"""
Configuration management for Code Weaver MCP server.

Provides comprehensive TOML-based configuration using pydantic-settings with multiple location support:
- Workspace local: .local.codeweaver.toml
- Repository: .codeweaver/.codeweaver.toml or .codeweaver.toml
- User: ~/.config/codeweaver/config.toml or %LOCALAPPDATA%/codeweaver/config.toml
"""

import logging

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from codeweaver.backends.config import BackendConfigExtended
from codeweaver.providers.config import (
    CohereConfig,
    CombinedProviderConfig,
    EmbeddingProviderConfig,
    HuggingFaceConfig,
    OpenAICompatibleConfig,
    OpenAIConfig,
    RerankingProviderConfig,
    SentenceTransformersConfig,
    VoyageConfig,
)
from codeweaver.types import ComponentType, ServicesConfig


logger = logging.getLogger(__name__)


class ChunkingConfig(BaseModel):
    """Configuration for code chunking."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)
    max_chunk_size: Annotated[int, Field(default=1500, ge=100, le=10000)] = Field(
        description="Maximum chunk size in characters"
    )
    min_chunk_size: Annotated[int, Field(default=50, ge=10, le=500)] = Field(
        description="Minimum chunk size in characters"
    )
    max_file_size_mb: Annotated[int, Field(default=1, ge=1, le=100)] = Field(
        description="Skip files larger than this (MB)"
    )
    language_settings: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Language-specific chunking settings"
    )

    @model_validator(mode="after")
    def validate_chunk_sizes(self) -> "ChunkingConfig":
        """Validate chunk size constraints."""
        if self.max_chunk_size <= self.min_chunk_size:
            raise ValueError("max_chunk_size must be greater than min_chunk_size")
        return self


class IndexingConfig(BaseModel):
    """Configuration for codebase indexing."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)
    use_gitignore: bool = Field(default=True, description="Use .gitignore patterns for filtering")
    additional_ignore_patterns: list[str] = Field(
        default_factory=lambda: [
            "node_modules",
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            "target",
            "build",
            "dist",
            ".next",
            ".nuxt",
            "coverage",
            ".pytest_cache",
            ".mypy_cache",
            ".vscode",
            ".idea",
        ],
        description="Additional patterns to ignore during indexing",
    )
    enable_auto_reindex: bool = Field(
        default=False, description="Enable automatic reindexing on file changes"
    )
    watch_debounce_seconds: Annotated[float, Field(default=2.0, ge=0.1, le=60.0)] = Field(
        description="Debounce time for file watching"
    )
    batch_size: Annotated[int, Field(default=8, ge=1, le=100)] = Field(
        description="Batch size for indexing operations"
    )
    max_concurrent_files: Annotated[int, Field(default=10, ge=1, le=50)] = Field(
        description="Maximum concurrent files to process"
    )


class RateLimitConfig(BaseModel):
    """Configuration for API rate limiting and backoff."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)
    voyage_requests_per_minute: Annotated[int, Field(default=100, ge=1, le=10000)] = Field(
        description="Voyage AI requests per minute"
    )
    voyage_tokens_per_minute: Annotated[int, Field(default=1000000, ge=1000, le=10000000)] = Field(
        description="Voyage AI tokens per minute"
    )
    openai_requests_per_minute: Annotated[int, Field(default=5000, ge=1, le=50000)] = Field(
        description="OpenAI requests per minute"
    )
    openai_tokens_per_minute: Annotated[int, Field(default=1000000, ge=1000, le=10000000)] = Field(
        description="OpenAI tokens per minute"
    )
    qdrant_requests_per_second: Annotated[int, Field(default=100, ge=1, le=1000)] = Field(
        description="Qdrant requests per second"
    )
    initial_backoff_seconds: Annotated[float, Field(default=1.0, ge=0.1, le=10.0)] = Field(
        description="Initial backoff time"
    )
    max_backoff_seconds: Annotated[float, Field(default=60.0, ge=1.0, le=300.0)] = Field(
        description="Maximum backoff time"
    )
    backoff_multiplier: Annotated[float, Field(default=2.0, ge=1.1, le=5.0)] = Field(
        description="Backoff multiplier factor"
    )
    max_retries: Annotated[int, Field(default=5, ge=1, le=20)] = Field(
        description="Maximum number of retries"
    )


class ServerConfig(BaseModel):
    """Configuration for the MCP server."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)
    server_name: str = Field(default="codeweaver-mcp", description="MCP server name")
    server_version: str = Field(default="1.0.0", description="MCP server version")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    enable_request_logging: bool = Field(
        default=False, description="Enable request/response logging"
    )
    max_search_results: Annotated[int, Field(default=50, ge=1, le=1000)] = Field(
        description="Maximum search results to return"
    )


class ProviderConfig(BaseModel):
    """Modern provider configuration using pydantic models."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)
    embedding: EmbeddingProviderConfig | CombinedProviderConfig | None = Field(
        default=None, description="Primary embedding provider configuration"
    )
    reranking: RerankingProviderConfig | CombinedProviderConfig | None = Field(
        default=None, description="Primary reranking provider configuration"
    )
    voyage_ai: VoyageConfig | None = Field(
        default=None, description="Voyage AI provider configuration"
    )
    openai: OpenAIConfig | None = Field(default=None, description="OpenAI provider configuration")
    cohere: CohereConfig | None = Field(default=None, description="Cohere provider configuration")
    huggingface: HuggingFaceConfig | None = Field(
        default=None, description="HuggingFace provider configuration"
    )
    sentence_transformers: SentenceTransformersConfig | None = Field(
        default=None, description="Sentence Transformers provider configuration"
    )
    openai_compatible: OpenAICompatibleConfig | None = Field(
        default=None, description="OpenAI-compatible provider configuration"
    )

    def get_active_embedding_provider(
        self,
    ) -> EmbeddingProviderConfig | CombinedProviderConfig | None:
        """Get the active embedding provider configuration."""
        return self.embedding

    def get_active_reranking_provider(
        self,
    ) -> RerankingProviderConfig | CombinedProviderConfig | None:
        """Get the active reranking provider configuration."""
        return self.reranking


class DefaultsConfig(BaseModel):
    """Configuration for default behavior and profiles."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)
    profile: Annotated[
        str, Field(default="codeweaver_original", description="Configuration profile to use")
    ]
    auto_configure: Annotated[
        bool, Field(default=True, description="Automatically configure based on profile")
    ]
    validate_setup: Annotated[
        bool, Field(default=True, description="Validate configuration during startup")
    ]
    strict_validation: Annotated[
        bool, Field(default=False, description="Use strict validation mode")
    ]


class PluginRegistryConfig(BaseModel):
    """Plugin registry configuration for controlling plugin behavior."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)
    enabled_plugins: Annotated[
        list[str],
        Field(default=["*"], description="List of enabled plugins (* means all discovered)"),
    ]
    disabled_plugins: Annotated[
        list[str], Field(default_factory=list, description="List of disabled plugins")
    ]
    plugin_priority_order: Annotated[
        list[str], Field(default_factory=list, description="Priority order for plugin resolution")
    ]
    auto_resolve_conflicts: Annotated[
        bool, Field(default=True, description="Automatically resolve plugin conflicts")
    ]
    require_explicit_enable: Annotated[
        bool, Field(default=False, description="Require explicit enabling of all plugins")
    ]


class CustomPluginConfig(BaseModel):
    """Configuration for a custom plugin."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)
    enabled: Annotated[bool, Field(default=True, description="Whether plugin is enabled")]
    plugin_type: Annotated[ComponentType, Field(description="Type of plugin")]
    module_path: Annotated[str, Field(description="Python module path")]
    class_name: Annotated[str, Field(description="Plugin class name")]
    entry_point: Annotated[
        str | None,
        Field(default=None, description="Entry point name (alternative to module_path/class_name)"),
    ]
    priority: Annotated[
        int, Field(default=50, ge=0, le=100, description="Plugin priority (0=lowest, 100=highest)")
    ]
    config: Annotated[
        dict[str, Any], Field(default_factory=dict, description="Plugin-specific configuration")
    ]
    dependencies: Annotated[
        list[str], Field(default_factory=list, description="Required dependencies")
    ]
    tags: Annotated[
        list[str], Field(default_factory=list, description="Plugin tags for categorization")
    ]


class PluginsConfig(BaseModel):
    """Enhanced plugin system configuration."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)
    enabled: Annotated[bool, Field(default=True, description="Enable plugin system")]
    auto_discover: Annotated[
        bool, Field(default=True, description="Automatically discover plugins")
    ]
    plugin_directories: Annotated[
        list[str],
        Field(
            default_factory=lambda: ["~/.codeweaver/plugins", "./plugins", "./codeweaver_plugins"],
            description="Directories to search for plugins",
        ),
    ]
    entry_point_groups: Annotated[
        list[str],
        Field(
            default_factory=lambda: [
                "codeweaver.backends",
                "codeweaver.providers",
                "codeweaver.sources",
                "codeweaver.services",
            ],
            description="Entry point groups to scan",
        ),
    ]
    registry: Annotated[
        PluginRegistryConfig,
        Field(default_factory=PluginRegistryConfig, description="Plugin registry configuration"),
    ]
    custom: Annotated[
        dict[str, CustomPluginConfig],
        Field(default_factory=dict, description="Custom plugin configurations"),
    ]
    development_mode: Annotated[
        bool, Field(default=False, description="Enable development mode for plugin debugging")
    ]
    validation_strict: Annotated[
        bool, Field(default=True, description="Use strict validation for plugins")
    ]


class ProfileConfig(BaseModel):
    """Configuration profile definition."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)
    name: Annotated[str, Field(description="Profile name")]
    description: Annotated[str, Field(description="Profile description")]
    data_sources: Annotated[
        dict[str, Any],
        Field(default_factory=dict, description="Data sources configuration overrides"),
    ]
    services: Annotated[
        dict[str, Any], Field(default_factory=dict, description="Services configuration overrides")
    ]
    providers: Annotated[
        dict[str, Any], Field(default_factory=dict, description="Providers configuration overrides")
    ]
    backend: Annotated[
        dict[str, Any], Field(default_factory=dict, description="Backend configuration overrides")
    ]
    indexing: Annotated[
        dict[str, Any], Field(default_factory=dict, description="Indexing configuration overrides")
    ]
    plugins: Annotated[
        dict[str, Any], Field(default_factory=dict, description="Plugin configuration overrides")
    ]
    factory: Annotated[
        dict[str, Any], Field(default_factory=dict, description="Factory configuration overrides")
    ]


class FactoryConfig(BaseModel):
    """Factory system configuration."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)
    enable_dependency_injection: Annotated[
        bool, Field(default=True, description="Enable dependency injection")
    ]
    enable_plugin_discovery: Annotated[
        bool, Field(default=True, description="Enable plugin discovery")
    ]
    validate_configurations: Annotated[
        bool, Field(default=True, description="Validate configurations during creation")
    ]
    lazy_initialization: Annotated[
        bool, Field(default=False, description="Use lazy initialization for components")
    ]
    enable_graceful_shutdown: Annotated[
        bool, Field(default=True, description="Enable graceful shutdown handling")
    ]
    shutdown_timeout: Annotated[
        float, Field(default=30.0, gt=0, description="Shutdown timeout in seconds")
    ]
    enable_health_checks: Annotated[
        bool, Field(default=True, description="Enable component health checks")
    ]
    health_check_interval: Annotated[
        float, Field(default=60.0, gt=0, description="Health check interval in seconds")
    ]
    enable_metrics: Annotated[
        bool, Field(default=True, description="Enable factory metrics collection")
    ]


class DataSourceConfig(BaseModel):
    """Data source configuration."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)
    enabled: bool = Field(default=True, description="Enable data sources system")
    default_source_type: str = Field(default="filesystem", description="Default source type")
    max_concurrent_sources: Annotated[int, Field(default=5, ge=1, le=20)] = Field(
        description="Maximum concurrent sources"
    )
    enable_content_deduplication: bool = Field(
        default=True, description="Enable content deduplication"
    )
    content_cache_ttl_hours: Annotated[int, Field(default=24, ge=1, le=168)] = Field(
        description="Content cache TTL in hours"
    )
    enable_metadata_extraction: bool = Field(default=True, description="Enable metadata extraction")
    sources: list[dict[str, Any]] = Field(
        default_factory=list, description="List of configured data sources"
    )


class CodeWeaverConfig(BaseSettings):
    """Main configuration for Code Weaver MCP server.

    Uses pydantic-settings for TOML-based configuration with hierarchical loading
    and environment variable support.
    """

    model_config = SettingsConfigDict(
        env_prefix="CW_",
        env_nested_delimiter="__",
        env_ignore_empty=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
        validate_assignment=True,
        str_strip_whitespace=True,
        case_sensitive=False,
    )

    @classmethod
    def settings_customize_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize configuration source priority and TOML file loading.

        Priority order (highest to lowest):
        1. Init settings (direct instantiation parameters)
        2. Environment variables (CW_* with __ nested delimiter)
        3. TOML configuration files (.local.codeweaver.toml > .codeweaver.toml > ~/.config/codeweaver/config.toml)
        4. .env files
        5. Secret files (lowest priority)
        """
        toml_settings = CustomTomlSource(settings_cls)
        return (init_settings, env_settings, toml_settings, dotenv_settings, file_secret_settings)

    backend: BackendConfigExtended = Field(
        default_factory=BackendConfigExtended, description="Vector database backend configuration"
    )
    providers: ProviderConfig = Field(
        default_factory=ProviderConfig, description="Provider configuration"
    )
    data_sources: DataSourceConfig = Field(
        default_factory=DataSourceConfig, description="Data source configuration"
    )
    defaults: DefaultsConfig = Field(
        default_factory=DefaultsConfig, description="Default behavior and profile configuration"
    )
    plugins: PluginsConfig = Field(
        default_factory=PluginsConfig, description="Plugin system configuration"
    )
    factory: FactoryConfig = Field(
        default_factory=FactoryConfig, description="Factory system configuration"
    )
    services: ServicesConfig = Field(
        default_factory=ServicesConfig, description="Service layer configuration"
    )
    profiles: dict[str, ProfileConfig] = Field(
        default_factory=dict, description="Available configuration profiles"
    )
    chunking: ChunkingConfig = Field(
        default_factory=ChunkingConfig, description="Code chunking configuration"
    )
    indexing: IndexingConfig = Field(
        default_factory=IndexingConfig, description="Codebase indexing configuration"
    )
    rate_limiting: RateLimitConfig = Field(
        default_factory=RateLimitConfig, description="API rate limiting configuration"
    )
    server: ServerConfig = Field(
        default_factory=ServerConfig, description="MCP server configuration"
    )

    @model_validator(mode="after")
    def apply_profile_configuration(self) -> "CodeWeaverConfig":
        """Apply selected profile configuration."""
        if (
            self.defaults.auto_configure
            and self.defaults.profile
            and (profile := self._get_profile(self.defaults.profile))
        ):
            self._apply_profile_overrides(profile)
        return self

    @model_validator(mode="after")
    def setup_original_defaults(self) -> "CodeWeaverConfig":
        """Set up original CodeWeaver defaults if using default profile."""
        if self.defaults.profile == "codeweaver_original":
            self._setup_original_defaults()
        return self

    @model_validator(mode="after")
    def setup_default_data_sources(self) -> "CodeWeaverConfig":
        """Set up default data sources if none are configured."""
        if not self.data_sources.sources:
            self.data_sources.sources = [
                {
                    "type": "filesystem",
                    "enabled": True,
                    "priority": 1,
                    "source_id": "default_filesystem",
                    "config": {
                        "root_path": ".",
                        "use_gitignore": True,
                        "additional_ignore_patterns": self.indexing.additional_ignore_patterns,
                        "max_file_size_mb": self.chunking.max_file_size_mb,
                    },
                }
            ]
        return self

    @field_validator("backend", mode="before")
    @classmethod
    def setup_backend_from_env(cls, value: Any) -> Any:
        """Set up backend configuration from environment variables if needed."""
        return value

    def get_effective_embedding_provider(self) -> str:
        """Get the effective embedding provider name."""
        if self.providers.embedding:
            return getattr(self.providers.embedding, "provider_name", "unknown")
        return "unknown"

    def get_effective_backend_provider(self) -> str:
        """Get the effective backend provider."""
        return self.backend.provider

    def get_effective_backend_url(self) -> str | None:
        """Get the effective backend URL."""
        return self.backend.url

    def get_effective_backend_api_key(self) -> str | None:
        """Get the effective backend API key."""
        return self.backend.api_key

    def _get_profile(self, profile_name: str) -> ProfileConfig | None:
        """Get profile configuration by name."""
        if profile_name in self.profiles:
            return self.profiles[profile_name]
        built_in_profiles = self._get_builtin_profiles()
        return built_in_profiles.get(profile_name)

    def _get_builtin_profiles(self) -> dict[str, ProfileConfig]:
        """Get built-in configuration profiles."""
        return {
            "codeweaver_original": ProfileConfig(
                name="codeweaver_original",
                description="Original CodeWeaver design: ast-grep + chunking + Voyage AI + Qdrant + auto-watch",
                data_sources={
                    "default_source_type": "ast_grep",
                    "sources": [
                        {
                            "type": "ast_grep",
                            "enabled": True,
                            "priority": 1,
                            "source_id": "primary_codebase",
                            "config": {
                                "root_path": ".",
                                "use_gitignore": True,
                                "enable_structural_chunking": True,
                                "auto_watch_changes": True,
                            },
                        }
                    ],
                },
                services={
                    "chunking": {
                        "provider": "ast_grep_chunking",
                        "enable_structural_chunking": True,
                        "respect_code_structure": True,
                    },
                    "filtering": {"provider": "gitignore_filtering", "use_gitignore": True},
                },
                providers={
                    "embedding": {"provider_type": "voyage_ai", "model": "voyage-code-3"},
                    "reranking": {"provider_type": "voyage_ai", "model": "voyage-rerank-2"},
                },
                backend={"provider": "qdrant", "enable_hybrid_search": False},
                indexing={"enable_auto_reindex": True, "batch_size": 8},
            ),
            "minimal": ProfileConfig(
                name="minimal",
                description="Minimal setup for quick testing",
                services={"chunking": {"provider": "simple_chunking", "max_chunk_size": 1000}},
                providers={
                    "embedding": {
                        "provider_type": "sentence_transformers",
                        "model": "all-MiniLM-L6-v2",
                    }
                },
                backend={"provider": "memory"},
                indexing={"enable_auto_reindex": False, "batch_size": 4},
            ),
            "performance": ProfileConfig(
                name="performance",
                description="Optimized for large codebases",
                indexing={"batch_size": 16, "max_concurrent_files": 20},
                services={"chunking": {"performance_mode": "fast", "max_chunk_size": 2000}},
                backend={"enable_hybrid_search": True, "enable_sparse_vectors": True},
            ),
        }

    def _apply_profile_overrides(self, profile: ProfileConfig) -> None:
        """Apply profile configuration overrides."""
        self._apply_data_sources_overrides(profile)
        self._apply_services_overrides_if_present(profile)
        self._apply_provider_overrides_if_present(profile)
        self._apply_backend_overrides(profile)
        self._apply_indexing_overrides(profile)

    def _apply_data_sources_overrides(self, profile: ProfileConfig) -> None:
        """Apply data sources configuration overrides."""
        if profile.data_sources:
            for key, value in profile.data_sources.items():
                if hasattr(self.data_sources, key):
                    setattr(self.data_sources, key, value)

    def _apply_services_overrides_if_present(self, profile: ProfileConfig) -> None:
        """Apply services configuration overrides if present."""
        if profile.services:
            self._apply_services_overrides(profile.services)

    def _apply_provider_overrides_if_present(self, profile: ProfileConfig) -> None:
        """Apply provider configuration overrides if present."""
        if profile.providers:
            self._apply_provider_overrides(profile.providers)

    def _apply_backend_overrides(self, profile: ProfileConfig) -> None:
        """Apply backend configuration overrides."""
        if profile.backend:
            for key, value in profile.backend.items():
                if hasattr(self.backend, key):
                    setattr(self.backend, key, value)

    def _apply_indexing_overrides(self, profile: ProfileConfig) -> None:
        """Apply indexing configuration overrides."""
        if profile.indexing:
            for key, value in profile.indexing.items():
                if hasattr(self.indexing, key):
                    setattr(self.indexing, key, value)

    def _apply_services_overrides(self, services_config: dict[str, Any]) -> None:
        """Apply services configuration overrides."""
        for service_name, service_config in services_config.items():
            if hasattr(self.services, service_name):
                service_obj = getattr(self.services, service_name)
                for key, value in service_config.items():
                    if hasattr(service_obj, key):
                        setattr(service_obj, key, value)

    def _apply_provider_overrides(self, providers_config: dict[str, Any]) -> None:
        """Apply provider configuration overrides."""
        for provider_name, provider_config in providers_config.items():
            if hasattr(self.providers, provider_name):
                provider_obj = getattr(self.providers, provider_name)
                if provider_obj is None:
                    logger.warning(
                        "Cannot create provider %s from profile - not implemented", provider_name
                    )
                else:
                    for key, value in provider_config.items():
                        if hasattr(provider_obj, key):
                            setattr(provider_obj, key, value)

    def _setup_original_defaults(self) -> None:
        """Set up original CodeWeaver defaults."""
        if not self.data_sources.sources:
            self.data_sources.sources = [
                {
                    "type": "ast_grep",
                    "enabled": True,
                    "priority": 1,
                    "source_id": "primary_codebase",
                    "config": {
                        "root_path": ".",
                        "use_gitignore": True,
                        "enable_structural_chunking": True,
                        "auto_watch_changes": True,
                    },
                }
            ]
        if self.services.chunking.provider == "fastmcp_chunking":
            self.services.chunking.provider = "ast_grep_chunking"
        if (
            not hasattr(self.indexing, "enable_auto_reindex")
            or not self.indexing.enable_auto_reindex
        ):
            self.indexing.enable_auto_reindex = True


class CustomTomlSource(TomlConfigSettingsSource):
    """Custom TOML source with multiple search paths and enhanced error handling."""

    model_config = SettingsConfigDict(extra="allow")

    def __init__(self, settings_cls: type[BaseSettings], toml_file: str | Path | None = None):
        """Initialize with custom search paths or explicit file.

        Args:
            settings_cls: The settings class
            toml_file: Optional specific TOML file path. If None, searches default locations.
        """
        self.settings_cls = settings_cls
        self.loaded_from = "None"
        self.initialization_error = None
        self._local_search_paths = None
        self._search_paths = None
        if toml_file is not None:
            self.loaded_from = str(toml_file)
            try:
                super().__init__(settings_cls, toml_file=toml_file)
            except Exception as e:
                self.initialization_error = e
                super().__init__(settings_cls, toml_file=None)
        else:
            import os
            import platform

            local_search_paths = [
                Path(".local.codeweaver.toml"),
                Path(".codeweaver") / "local.config.toml",
                Path(".codeweaver") / ".local.config.toml",
                Path(".codeweaver.toml"),
                Path(".codeweaver") / "config.toml",
                Path(".codeweaver") / ".codeweaver.toml",
            ]
            search_paths = [
                Path(os.environ.get("CW_CONFIG_FILE", "")),
                *local_search_paths,
                Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
                / "codeweaver"
                / "config.toml",
            ]
            if platform.system() == "Windows":
                search_paths.append(
                    Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
                    / "codeweaver"
                    / "config.toml"
                )
            if search_paths[0] == "":
                search_paths.pop(0)
            found_file = next((path for path in search_paths if path.exists()), None)
            self.loaded_from = str(found_file) if found_file else "None"
            try:
                super().__init__(settings_cls, toml_file=found_file)
            except Exception as e:
                self.initialization_error = e
                super().__init__(settings_cls, toml_file=None)

    def __call__(self) -> dict[str, Any]:
        """Load TOML data with enhanced error handling, logging, and migration support."""
        if self.initialization_error:
            logger.warning(
                "Failed to load TOML configuration from %s: %s",
                self.loaded_from,
                self.initialization_error,
            )
            return {}
        try:
            data = super().__call__()
            if data and hasattr(self, "toml_file") and self.toml_file:
                logger.info("Loaded configuration from TOML: %s", self.loaded_from)
        except FileNotFoundError:
            logger.debug("TOML configuration file not found: %s", self.loaded_from)
            return {}
        except Exception as e:
            logger.warning("Failed to load TOML configuration from %s: %s", self.loaded_from, e)
            return {}
        else:
            return data


class ConfigManager:
    """Simplified configuration manager using pydantic-settings native capabilities."""

    def __init__(self, config_path: str | Path | None = None):
        """Initialize the config manager with an optional specific config path.

        Args:
            config_path: Optional path to a specific configuration file.
        """
        self.config_path = config_path
        self._config: CodeWeaverConfig | None = None
        self._default_user_config_location = self._get_default_user_config_location()
        self._default_system_config_location = self._get_default_system_config_location()

    def _get_default_user_config_location(self) -> Path:
        """Get the default user configuration location."""
        import os
        import platform

        if platform.system() == "Windows":
            return (
                Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
                / "codeweaver"
                / "config.toml"
            )
        return (
            Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
            / "codeweaver"
            / "config.toml"
        )

    def _get_default_system_config_location(self) -> Path:
        """Get the default system-wide configuration location."""
        import os
        import platform

        if platform.system() == "Windows":
            return (
                Path(os.environ.get("PROGRAMDATA", Path.home() / "ProgramData"))
                / "codeweaver"
                / "config.toml"
            )
        return Path("/etc/codeweaver/config.toml")

    def get_config(self) -> CodeWeaverConfig:
        """Get the current configuration, loading it if necessary."""
        if self._config is None:
            self._config = self._load_config()
        return self._config

    def _load_config(self) -> CodeWeaverConfig:
        """Load configuration using pydantic-settings with custom source."""
        try:
            if self.config_path:
                return CodeWeaverConfigWithFile(toml_file=self.config_path)
        except Exception:
            logger.exception("Failed to load configuration")
            logger.info("Using default configuration")
            return CodeWeaverConfig(_env_file=None)
        else:
            return CodeWeaverConfig()

    def reload_config(self) -> CodeWeaverConfig:
        """Reload configuration, clearing cache."""
        self._config = None
        return self.get_config()

    def save_config(self, config: CodeWeaverConfig, config_path: str | Path | None = None) -> Path:
        """Save configuration to TOML file.

        Args:
            config: Configuration to save
            config_path: Optional path to save to. If None, uses the default user config location.

        Returns:
            Path where the configuration was saved
        """
        if config_path is None:
            save_path = Path.home() / ".config" / "codeweaver" / "config.toml"
        else:
            save_path = Path(config_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        config_data = config.model_dump(exclude_unset=True)
        try:
            import tomli_w

            with save_path.open("wb") as f:
                tomli_w.dump(config_data, f)
            logger.info("Saved configuration to: %s", save_path)
        except ImportError:
            logger.exception(
                "tomli_w is required for saving TOML files. Install with: pip install tomli_w"
            )
            raise
        else:
            return save_path

    def validate_config(self, config_path: str | Path) -> dict[str, Any]:
        """Validate a specific configuration file.

        Args:
            config_path: Path to the configuration file to validate

        Returns:
            Validation results dictionary
        """
        path = Path(config_path)
        result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "file_exists": path.exists(),
            "file_path": str(path),
            "loaded_from": None,
        }
        if not path.exists():
            result["errors"].append(f"Configuration file does not exist: {path}")
            return result
        try:
            self._load_from_default_path(path, result)
        except Exception as e:
            result["errors"].append(f"Configuration validation failed: {e}")
        return result

    def _load_from_default_path(self, path, result) -> None:
        """Load configuration from a specific path and validate it."""
        temp_manager = ConfigManager(config_path=path)
        config = temp_manager.get_config()
        result["valid"] = True
        result["loaded_from"] = str(path)
        result["warnings"].append("Configuration file loaded and validated successfully")
        result["summary"] = {
            "backend_provider": config.get_effective_backend_provider(),
            "embedding_provider": config.get_effective_embedding_provider(),
            "data_sources_count": len(config.data_sources.sources),
        }


class CodeWeaverConfigWithFile(CodeWeaverConfig):
    """CodeWeaver configuration with explicit TOML file specification."""

    def __init__(self, toml_file: str | Path, **kwargs):
        """Initialize with explicit TOML file.

        Args:
            toml_file: Path to specific TOML configuration file
            **kwargs: Additional configuration parameters
        """
        type(self)._current_toml_file = toml_file
        super().__init__(**kwargs)

    @classmethod
    def settings_customize_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize sources to use explicit TOML file."""
        toml_file = getattr(cls, "_current_toml_file", None)
        toml_settings = CustomTomlSource(settings_cls, toml_file=toml_file)
        return (init_settings, env_settings, toml_settings, dotenv_settings, file_secret_settings)


class ConfigurationError(Exception):
    """Configuration-related errors."""


class ProfileError(ConfigurationError):
    """Profile-related configuration errors."""


class PluginConfigurationError(ConfigurationError):
    """Plugin configuration errors."""


def setup_development_config() -> CodeWeaverConfig:
    """Set up development configuration with optimized settings for development work."""
    return CodeWeaverConfig(
        server={"log_level": "DEBUG", "enable_request_logging": True},
        indexing={"batch_size": 4, "enable_auto_reindex": True},
        chunking={"max_chunk_size": 1000},
    )


def setup_production_config(**kwargs) -> CodeWeaverConfig:
    """Set up production configuration with performance and security optimizations."""
    return CodeWeaverConfig(
        server={"log_level": "WARNING", "enable_request_logging": False, "max_search_results": 200},
        indexing={"batch_size": 16, "enable_auto_reindex": False},
        chunking={"max_chunk_size": 2000},
        rate_limiting={"max_retries": 3, "initial_backoff_seconds": 2.0},
        **kwargs,
    )


def setup_testing_config() -> CodeWeaverConfig:
    """Set up testing configuration with minimal settings for unit tests."""
    return CodeWeaverConfig(
        server={"log_level": "CRITICAL", "enable_request_logging": False},
        indexing={"batch_size": 1, "enable_auto_reindex": False},
        chunking={"max_chunk_size": 500, "min_chunk_size": 25},
        data_sources={"enabled": False},
    )


class ConfigValidator:
    """Validates configuration settings and provides recommendations."""

    def __init__(self, config: CodeWeaverConfig):
        """Initialize with configuration to validate."""
        self.config = config

    def validate(self) -> dict[str, Any]:
        """Validate the configuration and return results."""
        results = {"valid": True, "errors": [], "warnings": [], "recommendations": []}
        backend_validation = self._validate_backend()
        results["errors"].extend(backend_validation["errors"])
        results["warnings"].extend(backend_validation["warnings"])
        provider_validation = self._validate_providers()
        results["errors"].extend(provider_validation["errors"])
        results["warnings"].extend(provider_validation["warnings"])
        performance_validation = self._validate_performance()
        results["warnings"].extend(performance_validation["warnings"])
        results["recommendations"].extend(performance_validation["recommendations"])
        results["valid"] = len(results["errors"]) == 0
        return results

    def _validate_backend(self) -> dict[str, list[str]]:
        """Validate backend configuration."""
        errors = []
        warnings = []
        backend = self.config.backend
        if not backend.url:
            errors.append("Backend URL is required")
        if backend.provider == "qdrant" and (not backend.api_key):
            warnings.append("Qdrant API key is not set - may be required for authentication")
        if not backend.collection_name:
            warnings.append("Backend collection name is not set")
        return {"errors": errors, "warnings": warnings}

    def _validate_providers(self) -> dict[str, list[str]]:
        """Validate provider configuration."""
        errors = []
        warnings = []
        providers = self.config.providers
        if not providers.embedding:
            warnings.append("No embedding provider configured")
        provider_configs = [
            providers.voyage_ai,
            providers.openai,
            providers.cohere,
            providers.huggingface,
            providers.sentence_transformers,
            providers.openai_compatible,
        ]
        if not any(provider_configs):
            errors.append("No provider configurations found")
        return {"errors": errors, "warnings": warnings}

    def _validate_performance(self) -> dict[str, list[str]]:
        """Validate performance settings."""
        warnings = []
        recommendations = []
        chunking = self.config.chunking
        indexing = self.config.indexing
        if chunking.max_chunk_size > 3000:
            warnings.append("Large chunk sizes may impact embedding performance")
        if chunking.min_chunk_size < 25:
            warnings.append("Very small chunks may produce poor search results")
        if indexing.batch_size > 20:
            recommendations.append("Consider reducing batch size for better memory usage")
        if indexing.batch_size < 4:
            recommendations.append("Consider increasing batch size for better performance")
        return {"warnings": warnings, "recommendations": recommendations}


class ConfigSchema:
    """Utilities for generating and validating configuration schemas."""

    @staticmethod
    def generate_example_config(format_type: str = "toml") -> str:
        """Generate an example configuration in the specified format."""
        if format_type.lower() == "toml":
            return create_example_config()
        raise ValueError(f"Unsupported format: {format_type}")

    @staticmethod
    def get_schema_dict() -> dict[str, Any]:
        """Get the configuration schema as a dictionary."""
        return CodeWeaverConfig.model_json_schema()

    @staticmethod
    def validate_schema(config_dict: dict[str, Any]) -> dict[str, Any]:
        """Validate a configuration dictionary against the schema."""
        try:
            CodeWeaverConfig(**config_dict)
        except Exception as e:
            return {"valid": False, "errors": [str(e)]}
        else:
            return {"valid": True, "errors": []}


class BackendConfigBuilder:
    """Builder for backend configurations."""

    def __init__(self, backend_type: str):
        """Initialize with backend type."""
        self.backend_type = backend_type.lower()
        self.config_data = {}

    def url(self, url: str) -> "BackendConfigBuilder":
        """Set backend URL."""
        self.config_data["url"] = url
        return self

    def api_key(self, api_key: str) -> "BackendConfigBuilder":
        """Set backend API key."""
        self.config_data["api_key"] = api_key
        return self

    def collection_name(self, name: str) -> "BackendConfigBuilder":
        """Set collection name."""
        self.config_data["collection_name"] = name
        return self

    def hybrid_search(self, *, enabled: bool = True) -> "BackendConfigBuilder":
        """Enable/disable hybrid search."""
        self.config_data["enable_hybrid_search"] = enabled
        return self

    def build(self) -> dict[str, Any]:
        """Build the backend configuration."""
        config = {"provider": self.backend_type, **self.config_data}
        if self.backend_type == "qdrant":
            config.setdefault("url", "http://localhost:6333")
            config.setdefault("collection_name", "codeweaver-embeddings")
        elif self.backend_type == "pinecone":
            config.setdefault("collection_name", "codeweaver-index")
        elif self.backend_type == "weaviate":
            config.setdefault("url", "http://localhost:8080")
            config.setdefault("collection_name", "CodeWeaver")
        return config


class ProviderConfigBuilder:
    """Builder for provider configurations."""

    def __init__(self, provider_type: str):
        """Initialize with provider type."""
        self.provider_type = provider_type.lower()
        self.config_data = {}

    def api_key(self, api_key: str) -> "ProviderConfigBuilder":
        """Set provider API key."""
        self.config_data["api_key"] = api_key
        return self

    def model(self, model: str) -> "ProviderConfigBuilder":
        """Set model name."""
        self.config_data["model"] = model
        return self

    def embedding_model(self, model: str) -> "ProviderConfigBuilder":
        """Set embedding model."""
        self.config_data["embedding_model"] = model
        return self

    def reranking_model(self, model: str) -> "ProviderConfigBuilder":
        """Set reranking model."""
        self.config_data["reranking_model"] = model
        return self

    def enable_embeddings(self, *, enabled: bool = True) -> "ProviderConfigBuilder":
        """Enable/disable embeddings."""
        self.config_data["enable_embeddings"] = enabled
        return self

    def enable_reranking(self, *, enabled: bool = True) -> "ProviderConfigBuilder":
        """Enable/disable reranking."""
        self.config_data["enable_reranking"] = enabled
        return self

    def build(self) -> dict[str, Any]:
        """Build the provider configuration."""
        config = {**self.config_data}
        match self.provider_type:
            case "voyage":
                config.setdefault("model", "voyage-code-3")
                config.setdefault("embedding_model", "voyage-code-3")
                config.setdefault("reranking_model", "voyage-rerank-2")
                config.setdefault("enable_embeddings", True)
                config.setdefault("enable_reranking", True)
            case "openai":
                config.setdefault("model", "text-embedding-3-small")
                config.setdefault("enable_embeddings", True)
                config.setdefault("enable_reranking", False)
            case "cohere":
                config.setdefault("model", "embed-english-v3.0")
                config.setdefault("reranking_model", "rerank-english-v3.0")
                config.setdefault("enable_embeddings", True)
                config.setdefault("enable_reranking", True)
            case "huggingface":
                config.setdefault("model", "sentence-transformers/all-MiniLM-L6-v2")
                config.setdefault("enable_embeddings", True)
                config.setdefault("enable_reranking", False)
            case "sentence_transformers":
                config.setdefault("model", "sentence-transformers/all-MiniLM-L6-v2")
                config.setdefault("enable_embeddings", True)
                config.setdefault("enable_reranking", False)
        return config


_config_manager: ConfigManager | None = None


def get_config_manager(config_path: str | Path | None = None) -> ConfigManager:
    """Get the global config manager instance."""
    global _config_manager
    if _config_manager is None or config_path is not None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config() -> CodeWeaverConfig:
    """Get the current configuration."""
    return get_config_manager().get_config()


def create_example_config() -> str:
    """Create an example TOML configuration."""
    return '# Code Weaver MCP Server Configuration\n# Place this file in one of these locations (highest precedence first):\n# 1. .local.codeweaver.toml (workspace local)\n# 2. .codeweaver.toml (repository)\n# 3. ~/.config/codeweaver/config.toml (user)\n\n# Vector Database Backend Configuration\n[backend]\nprovider = "qdrant"\nurl = "YOUR_BACKEND_URL"  # Can also use CW_BACKEND__URL env var\napi_key = "YOUR_BACKEND_API_KEY"  # Can also use CW_BACKEND__API_KEY env var\ncollection_name = "code-embeddings"\nenable_hybrid_search = false\nenable_sparse_vectors = false\n\n# Provider Configuration\n[providers]\n\n# Voyage AI Configuration (Combined embedding + reranking)\n[providers.voyage_ai]\napi_key = "YOUR_VOYAGE_API_KEY"  # Can also use CW_PROVIDERS__VOYAGE_AI__API_KEY env var\nmodel = "voyage-code-3"\nembedding_model = "voyage-code-3"\nreranking_model = "voyage-rerank-2"\nenable_embeddings = true\nenable_reranking = true\n\n# Set active providers\n[providers.embedding]\napi_key = "YOUR_VOYAGE_API_KEY"\nmodel = "voyage-code-3"\n\n[providers.reranking]\napi_key = "YOUR_VOYAGE_API_KEY"\nmodel = "voyage-rerank-2"\n\n# Data Sources Configuration\n[data_sources]\nenabled = true\ndefault_source_type = "filesystem"\n\n# File System Source\n[[data_sources.sources]]\ntype = "filesystem"\nenabled = true\npriority = 1\nsource_id = "main_codebase"\n\n[data_sources.sources.config]\nroot_path = "."\nuse_gitignore = true\nmax_file_size_mb = 1\n\n# Shared Configuration\n[chunking]\nmax_chunk_size = 1500\nmin_chunk_size = 50\nmax_file_size_mb = 1\n\n[indexing]\nuse_gitignore = true\nenable_auto_reindex = false\nbatch_size = 8\n\n[rate_limiting]\nvoyage_requests_per_minute = 100\nvoyage_tokens_per_minute = 1000000\nmax_retries = 5\n\n[server]\nserver_name = "codeweaver-mcp"\nlog_level = "INFO"\nmax_search_results = 50\n'


def get_effective_config_summary() -> dict[str, Any]:
    """Get a summary of the effective configuration."""
    try:
        config = get_config()
    except Exception as e:
        return {"error": str(e)}
    else:
        return {
            "backend": {
                "provider": config.get_effective_backend_provider(),
                "url": config.get_effective_backend_url(),
                "has_api_key": bool(config.get_effective_backend_api_key()),
                "collection_name": config.backend.collection_name,
                "hybrid_search_enabled": config.backend.enable_hybrid_search,
            },
            "embedding": {
                "provider": config.get_effective_embedding_provider(),
                "has_active_provider": config.providers.embedding is not None,
            },
            "data_sources": {
                "enabled": config.data_sources.enabled,
                "source_count": len(config.data_sources.sources),
                "source_types": list({
                    source.get("type", "unknown") for source in config.data_sources.sources
                }),
            },
        }
