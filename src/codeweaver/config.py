# sourcery skip: lambdas-should-be-short
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

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
from pydantic_settings import BaseSettings, SettingsConfigDict

# Import configuration components
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

    # Language-specific settings
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

    # File filtering
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

    # File watching
    enable_auto_reindex: bool = Field(
        default=False, description="Enable automatic reindexing on file changes"
    )
    watch_debounce_seconds: Annotated[float, Field(default=2.0, ge=0.1, le=60.0)] = Field(
        description="Debounce time for file watching"
    )

    # Performance
    batch_size: Annotated[int, Field(default=8, ge=1, le=100)] = Field(
        description="Batch size for indexing operations"
    )
    max_concurrent_files: Annotated[int, Field(default=10, ge=1, le=50)] = Field(
        description="Maximum concurrent files to process"
    )


class RateLimitConfig(BaseModel):
    """Configuration for API rate limiting and backoff."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    # Voyage AI rate limiting
    voyage_requests_per_minute: Annotated[int, Field(default=100, ge=1, le=10000)] = Field(
        description="Voyage AI requests per minute"
    )
    voyage_tokens_per_minute: Annotated[int, Field(default=1000000, ge=1000, le=10000000)] = Field(
        description="Voyage AI tokens per minute"
    )

    # OpenAI rate limiting
    openai_requests_per_minute: Annotated[int, Field(default=5000, ge=1, le=50000)] = Field(
        description="OpenAI requests per minute"
    )
    openai_tokens_per_minute: Annotated[int, Field(default=1000000, ge=1000, le=10000000)] = Field(
        description="OpenAI tokens per minute"
    )

    # Qdrant rate limiting
    qdrant_requests_per_second: Annotated[int, Field(default=100, ge=1, le=1000)] = Field(
        description="Qdrant requests per second"
    )

    # Exponential backoff
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

    # Performance settings
    enable_request_logging: bool = Field(
        default=False, description="Enable request/response logging"
    )
    max_search_results: Annotated[int, Field(default=50, ge=1, le=1000)] = Field(
        description="Maximum search results to return"
    )


class ProviderConfig(BaseModel):
    """Modern provider configuration using pydantic models."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    # Primary embedding provider
    embedding: EmbeddingProviderConfig | CombinedProviderConfig | None = Field(
        default=None, description="Primary embedding provider configuration"
    )

    # Primary reranking provider
    reranking: RerankingProviderConfig | CombinedProviderConfig | None = Field(
        default=None, description="Primary reranking provider configuration"
    )

    # Provider-specific configurations
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


class DataSourceConfig(BaseModel):
    """Data source configuration."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    enabled: bool = Field(default=True, description="Enable data sources system")
    default_source_type: str = Field(default="filesystem", description="Default source type")
    max_concurrent_sources: Annotated[int, Field(default=5, ge=1, le=20)] = Field(
        description="Maximum concurrent sources"
    )

    # Content processing
    enable_content_deduplication: bool = Field(
        default=True, description="Enable content deduplication"
    )
    content_cache_ttl_hours: Annotated[int, Field(default=24, ge=1, le=168)] = Field(
        description="Content cache TTL in hours"
    )
    enable_metadata_extraction: bool = Field(default=True, description="Enable metadata extraction")

    # Source definitions
    sources: list[dict[str, Any]] = Field(
        default_factory=list, description="List of configured data sources"
    )


class CodeWeaverConfig(BaseSettings):
    """Main configuration for Code Weaver MCP server.

    Uses pydantic-settings for TOML-based configuration with hierarchical loading
    and environment variable support.
    """

    model_config = SettingsConfigDict(
        # TOML file settings
        toml_file=[
            ".local.codeweaver.toml",
            ".codeweaver.toml",
            Path.home() / ".config" / "codeweaver" / "config.toml",
        ],
        # Environment variable settings
        env_prefix="CW_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        # Extra configuration
        extra="allow",
        validate_assignment=True,
        str_strip_whitespace=True,
        case_sensitive=False,
    )

    # Core configuration sections
    backend: BackendConfigExtended = Field(
        default_factory=BackendConfigExtended, description="Vector database backend configuration"
    )
    providers: ProviderConfig = Field(
        default_factory=ProviderConfig, description="Provider configuration"
    )
    data_sources: DataSourceConfig = Field(
        default_factory=DataSourceConfig, description="Data source configuration"
    )

    # Shared configuration
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


class ConfigManager:
    """Manages configuration loading and caching."""

    def __init__(self, config_path: str | Path | None = None):
        """Initialize the config manager with an optional specific config path.

        Args:
            config_path: Optional path to a specific configuration file.
                        If provided, this will be added to the search locations.
        """
        self.config_path = Path(config_path) if config_path else None
        self._config: CodeWeaverConfig | None = None

    def get_config(self) -> CodeWeaverConfig:
        """Get the current configuration, loading it if necessary."""
        if self._config is None:
            self._config = self._load_config()
        return self._config

    def _load_config(self) -> CodeWeaverConfig:
        """Load configuration using manual TOML loading with pydantic-settings."""
        # Define search locations
        search_paths = []
        if self.config_path:
            search_paths.append(self.config_path)

        search_paths.extend([
            Path(".local.codeweaver.toml"),
            Path(".codeweaver.toml"),
            Path.home() / ".config" / "codeweaver" / "config.toml",
        ])

        # Load TOML data from first existing file
        toml_data = {}
        for config_file in search_paths:
            if config_file.exists():
                try:
                    import tomllib

                    with config_file.open("rb") as f:
                        toml_data = tomllib.load(f)
                    logger.info("Loaded configuration from: %s", config_file)
                    break
                except Exception as e:
                    logger.warning("Failed to load config from %s: %s", config_file, e)

        return CodeWeaverConfig(**toml_data)

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
            # Use the user config location as default for saving
            save_path = Path.home() / ".config" / "codeweaver" / "config.toml"
        else:
            save_path = Path(config_path)

        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Use pydantic's model_dump_json for serialization
        config_data = config.model_dump(exclude_unset=True)

        # Convert to TOML format
        import tomli_w

        with save_path.open("wb") as f:
            tomli_w.dump(config_data, f)

        logger.info("Saved configuration to: %s", save_path)
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
        }

        if not path.exists():
            result["errors"].append(f"Configuration file does not exist: {path}")
            return result

        try:
            # Try to load the config with the specific path
            temp_manager = ConfigManager(config_path=path)
            temp_manager.get_config()
            result["valid"] = True
            result["warnings"].append("Configuration file loaded and validated successfully")
        except Exception as e:
            result["errors"].append(f"Configuration validation failed: {e}")

        return result


# Global config manager instance
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
    return """# Code Weaver MCP Server Configuration
# Place this file in one of these locations (highest precedence first):
# 1. .local.codeweaver.toml (workspace local)
# 2. .codeweaver.toml (repository)
# 3. ~/.config/codeweaver/config.toml (user)

# Vector Database Backend Configuration
[backend]
provider = "qdrant"
url = "YOUR_BACKEND_URL"  # Can also use CW_BACKEND__URL env var
api_key = "YOUR_BACKEND_API_KEY"  # Can also use CW_BACKEND__API_KEY env var
collection_name = "code-embeddings"
enable_hybrid_search = false
enable_sparse_vectors = false

# Provider Configuration
[providers]

# Voyage AI Configuration (Combined embedding + reranking)
[providers.voyage_ai]
api_key = "YOUR_VOYAGE_API_KEY"  # Can also use CW_PROVIDERS__VOYAGE_AI__API_KEY env var
model = "voyage-code-3"
embedding_model = "voyage-code-3"
reranking_model = "voyage-rerank-2"
enable_embeddings = true
enable_reranking = true

# Set active providers
[providers.embedding]
api_key = "YOUR_VOYAGE_API_KEY"
model = "voyage-code-3"

[providers.reranking]
api_key = "YOUR_VOYAGE_API_KEY"
model = "voyage-rerank-2"

# Data Sources Configuration
[data_sources]
enabled = true
default_source_type = "filesystem"

# File System Source
[[data_sources.sources]]
type = "filesystem"
enabled = true
priority = 1
source_id = "main_codebase"

[data_sources.sources.config]
root_path = "."
use_gitignore = true
max_file_size_mb = 1

# Shared Configuration
[chunking]
max_chunk_size = 1500
min_chunk_size = 50
max_file_size_mb = 1

[indexing]
use_gitignore = true
enable_auto_reindex = false
batch_size = 8

[rate_limiting]
voyage_requests_per_minute = 100
voyage_tokens_per_minute = 1000000
max_retries = 5

[server]
server_name = "codeweaver-mcp"
log_level = "INFO"
max_search_results = 50
"""


def get_effective_config_summary() -> dict[str, Any]:
    """Get a summary of the effective configuration."""
    try:
        config = get_config()
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
    except Exception as e:
        return {"error": str(e)}
