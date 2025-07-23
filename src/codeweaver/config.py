# sourcery skip: lambdas-should-be-short
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Configuration management for Code Weaver MCP server.

Provides comprehensive TOML-based configuration with multiple location support:
- Workspace local: .local.code-weaver.toml
- Repository: .code-weaver.toml
- User: ~/.config/code-weaver/config.toml
"""

import logging
import os

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar


try:
    import tomllib
except ImportError:
    # Python < 3.11 fallback
    import tomli as tomllib

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding providers."""

    provider: str = "voyage"  # "voyage" or "openai"
    api_key: str | None = None
    model: str = "voyage-code-3"
    dimension: int = 1024
    batch_size: int = 8

    # OpenAI-compatible provider settings
    base_url: str | None = None
    custom_headers: dict[str, str] = field(default_factory=dict)


@dataclass
class QdrantConfig:
    """Configuration for Qdrant vector database."""

    url: str | None = None
    api_key: str | None = None
    collection_name: str = "code-embeddings"

    # Hybrid search settings
    enable_sparse_vectors: bool = False
    sparse_vector_name: str = "sparse"


@dataclass
class ChunkingConfig:
    """Configuration for code chunking."""

    max_chunk_size: int = 1500
    min_chunk_size: int = 50
    max_file_size_mb: int = 1  # Skip files larger than this

    # Language-specific settings
    language_settings: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class IndexingConfig:
    """Configuration for codebase indexing."""

    # File filtering
    use_gitignore: bool = True
    additional_ignore_patterns: list[str] = field(
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
        ]
    )

    # File watching
    enable_auto_reindex: bool = False
    watch_debounce_seconds: float = 2.0

    # Performance
    batch_size: int = 8
    max_concurrent_files: int = 10


@dataclass
class RateLimitConfig:
    """Configuration for API rate limiting and backoff."""

    # Voyage AI rate limiting
    voyage_requests_per_minute: int = 100
    voyage_tokens_per_minute: int = 1000000

    # OpenAI rate limiting
    openai_requests_per_minute: int = 5000
    openai_tokens_per_minute: int = 1000000

    # Qdrant rate limiting
    qdrant_requests_per_second: int = 100

    # Exponential backoff
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 60.0
    backoff_multiplier: float = 2.0
    max_retries: int = 5


@dataclass
class ServerConfig:
    """Configuration for the MCP server."""

    server_name: str = "code-weaver-mcp"
    server_version: str = "2.0.0"
    log_level: str = "INFO"

    # Performance settings
    enable_request_logging: bool = False
    max_search_results: int = 50


@dataclass
class CodeWeaverConfig:
    """Main configuration for Code Weaver MCP server."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    rate_limiting: RateLimitConfig = field(default_factory=RateLimitConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    def merge_from_dict(self, config_dict: dict[str, Any]) -> None:
        """Merge configuration from a dictionary."""
        for section_name, section_data in config_dict.items():
            if hasattr(self, section_name) and isinstance(section_data, dict):
                section = getattr(self, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
                    else:
                        logger.warning("Unknown config key: %s.%s", section_name, key)
            else:
                logger.warning("Unknown config section: %s", section_name)

    def merge_from_env(self) -> None:
        """Merge configuration from environment variables."""
        # Embedding configuration
        if api_key := os.getenv("VOYAGE_API_KEY"):
            self.embedding.api_key = api_key
        if model := os.getenv("VOYAGE_MODEL"):
            self.embedding.model = model
        if provider := os.getenv("EMBEDDING_PROVIDER"):
            self.embedding.provider = provider.lower()
        if base_url := os.getenv("OPENAI_BASE_URL"):
            self.embedding.base_url = base_url
        if (api_key := os.getenv("OPENAI_API_KEY")) and self.embedding.provider == "openai":
            self.embedding.api_key = api_key

        # Qdrant configuration
        if url := os.getenv("QDRANT_URL"):
            self.qdrant.url = url
        if api_key := os.getenv("QDRANT_API_KEY"):
            self.qdrant.api_key = api_key
        if collection := os.getenv("COLLECTION_NAME"):
            self.qdrant.collection_name = collection

        # Server configuration
        if log_level := os.getenv("LOG_LEVEL"):
            self.server.log_level = log_level.upper()


class ConfigManager:
    """Manages configuration loading and merging from multiple sources."""

    CONFIG_LOCATIONS: ClassVar[list[str | Path]] = [
        ".local.code-weaver.toml",  # Workspace local
        ".code-weaver.toml",  # Repository
        Path.home() / ".config" / "code-weaver" / "config.toml",  # User
    ]

    def __init__(self, config_path: str | Path | None = None):
        """Initialize the config manager with an optional specific config path.

        Args:
            config_path: Optional path to a specific configuration file.
                        If None, uses the default search locations.
        """
        self.config_path = Path(config_path) if config_path else None
        self._config: CodeWeaverConfig | None = None

    def load_config(self) -> CodeWeaverConfig:
        """Load configuration from all sources with proper precedence."""
        if self._config is not None:
            return self._config

        config = CodeWeaverConfig()

        # Load from config files (lowest to highest precedence)
        config_files = list(reversed(self.CONFIG_LOCATIONS))
        if self.config_path:
            config_files.append(self.config_path)

        for config_file in config_files:
            config_path = Path(config_file) if isinstance(config_file, str) else config_file

            if config_path.exists():
                try:
                    with config_path.open("rb") as f:
                        config_dict = tomllib.load(f)
                    config.merge_from_dict(config_dict)
                    logger.info("Loaded configuration from: %s", config_path)
                except Exception as e:
                    logger.warning("Failed to load config from %s: %s", config_path, e)

        # Environment variables have highest precedence
        config.merge_from_env()

        # Validate configuration
        self._validate_config(config)

        self._config = config
        return config

    def _validate_config(self, config: CodeWeaverConfig) -> None:
        """Validate the configuration and raise errors for missing required values."""
        # Check required API keys based on provider
        if config.embedding.provider == "voyage":
            if not config.embedding.api_key:
                raise ValueError("VOYAGE_API_KEY is required when using Voyage AI embeddings")
        elif config.embedding.provider == "openai":
            if not config.embedding.api_key:
                raise ValueError("OPENAI_API_KEY is required when using OpenAI embeddings")
        else:
            raise ValueError(f"Unknown embedding provider: {config.embedding.provider}")

        # Check Qdrant configuration
        if not config.qdrant.url:
            raise ValueError("QDRANT_URL is required")

        # Validate rate limiting values
        if config.rate_limiting.max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if config.rate_limiting.initial_backoff_seconds <= 0:
            raise ValueError("initial_backoff_seconds must be > 0")

    def get_example_config(self) -> str:
        """Get an example TOML configuration file."""
        return """# Code Weaver MCP Server Configuration
# Place this file in one of these locations (highest precedence first):
# 1. .local.code-weaver.toml (workspace local)
# 2. .code-weaver.toml (repository)
# 3. ~/.config/code-weaver/config.toml (user)

[embedding]
provider = "voyage"  # "voyage" or "openai"
api_key = "YOUR_VOYAGE_API_KEY"  # Can also use VOYAGE_API_KEY env var
model = "voyage-code-3"
dimension = 1024
batch_size = 8

# For OpenAI provider, use:
# provider = "openai"
# api_key = "YOUR_OPENAI_API_KEY"  # Can also use OPENAI_API_KEY env var
# model = "text-embedding-3-small"  # or "text-embedding-3-large"
# dimension = 1536  # Optional: reduce dimensions from native (1536 for small, 3072 for large)

# For OpenAI-compatible providers (e.g., local models):
# base_url = "http://localhost:8000/v1"
# custom_headers = { "Authorization" = "Bearer your-token" }

[qdrant]
url = "YOUR_QDRANT_URL"  # Can also use QDRANT_URL env var
api_key = "YOUR_QDRANT_API_KEY"  # Can also use QDRANT_API_KEY env var
collection_name = "code-embeddings"

# Hybrid search (experimental)
enable_sparse_vectors = false
sparse_vector_name = "sparse"

[chunking]
max_chunk_size = 1500
min_chunk_size = 50
max_file_size_mb = 1

# Language-specific chunking settings
# [chunking.language_settings.python]
# max_chunk_size = 2000

[indexing]
use_gitignore = true
additional_ignore_patterns = [
    "node_modules", ".git", ".venv", "venv", "__pycache__",
    "target", "build", "dist", ".next", ".nuxt", "coverage"
]

# Auto-reindexing on file changes
enable_auto_reindex = false
watch_debounce_seconds = 2.0

# Performance
batch_size = 8
max_concurrent_files = 10

[rate_limiting]
# Voyage AI limits
voyage_requests_per_minute = 100
voyage_tokens_per_minute = 1000000

# OpenAI limits
openai_requests_per_minute = 5000
openai_tokens_per_minute = 1000000

# Qdrant limits
qdrant_requests_per_second = 100

# Exponential backoff
initial_backoff_seconds = 1.0
max_backoff_seconds = 60.0
backoff_multiplier = 2.0
max_retries = 5

[server]
server_name = "code-weaver-mcp"
server_version = "2.0.0"
log_level = "INFO"
enable_request_logging = false
max_search_results = 50
"""


# Global config manager instance
_config_manager: ConfigManager | None = None


def get_config_manager(config_path: str | Path | None = None) -> ConfigManager:
    """Get the global config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config() -> CodeWeaverConfig:
    """Get the current configuration."""
    return get_config_manager().load_config()
