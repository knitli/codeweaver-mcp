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
from typing import Any, ClassVar, Literal


try:
    import tomllib
except ImportError:
    # Python < 3.11 fallback
    import tomli as tomllib

# Import new configuration components
try:
    from codeweaver.backends.config import BackendConfigExtended, create_backend_config_from_legacy
    from codeweaver.providers.factory import ProviderConfig
    from codeweaver.sources.config import DataSourcesConfig, extend_config_with_data_sources

    _EXTENDED_CONFIGS_AVAILABLE = True
except ImportError:
    # Fallback for when extended configurations are not available
    _EXTENDED_CONFIGS_AVAILABLE = False
    BackendConfigExtended = None
    ProviderConfig = None
    DataSourcesConfig = None

logger = logging.getLogger(__name__)


@dataclass
class LegacyEmbeddingConfig:
    """Legacy embedding configuration for backward compatibility."""

    provider: str = "voyage"  # "voyage", "openai", "cohere", "sentence-transformers", "huggingface"
    api_key: str | None = None
    model: str = "voyage-code-3"
    dimension: int = 1024
    batch_size: int = 8

    # OpenAI-compatible provider settings
    base_url: str | None = None
    custom_headers: dict[str, str] = field(default_factory=dict)

    # Reranking configuration
    rerank_provider: str | None = None  # If None, uses same as embedding provider if available
    rerank_model: str | None = None  # If None, uses provider default

    # Local model settings (for sentence-transformers, huggingface local)
    use_local: bool = False
    device: str = "auto"  # "cpu", "cuda", "auto"
    normalize_embeddings: bool = True


@dataclass
class EmbeddingConfig:
    """Configuration for embedding providers."""

    provider: str = "voyage"  # "voyage", "openai", "cohere", "sentence-transformers", "huggingface"
    api_key: str | None = None
    model: str = "voyage-code-3"
    dimension: int = 1024
    batch_size: int = 8

    # OpenAI-compatible provider settings
    base_url: str | None = None
    custom_headers: dict[str, str] = field(default_factory=dict)

    # Reranking configuration
    rerank_provider: str | None = None  # If None, uses same as embedding provider if available
    rerank_model: str | None = None  # If None, uses provider default

    # Local model settings (for sentence-transformers, huggingface local)
    use_local: bool = False
    device: str = "auto"  # "cpu", "cuda", "auto"
    normalize_embeddings: bool = True


@dataclass
class LegacyQdrantConfig:
    """Legacy Qdrant configuration for backward compatibility."""

    url: str | None = None
    api_key: str | None = None
    collection_name: str = "code-embeddings"

    # Hybrid search settings
    enable_sparse_vectors: bool = False
    sparse_vector_name: str = "sparse"


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
class BackendConfig:
    """Vector database backend configuration."""

    provider: Literal[
        "qdrant", "pinecone", "chroma", "weaviate", "pgvector", "milvus", "elasticsearch"
    ] = "qdrant"
    url: str | None = None
    api_key: str | None = None
    collection_name: str = "code-embeddings"

    # Feature capabilities
    enable_hybrid_search: bool = False
    enable_sparse_vectors: bool = False
    enable_streaming: bool = False

    # Performance settings
    batch_size: int = 100
    connection_timeout: float = 30.0
    request_timeout: float = 60.0
    max_connections: int = 10

    # Provider-specific options (key-value pairs)
    provider_options: dict[str, Any] = field(default_factory=dict)

    # Advanced settings
    prefer_disk: bool = False
    enable_failover: bool = False
    replica_urls: list[str] = field(default_factory=list)


@dataclass
class ProviderConfig:
    """Embedding and reranking provider configuration."""

    # Embedding provider
    embedding_provider: Literal[
        "voyage", "openai", "cohere", "sentence-transformers", "huggingface"
    ] = "voyage"
    embedding_api_key: str | None = None
    embedding_model: str = "voyage-code-3"
    embedding_dimension: int = 1024
    embedding_batch_size: int = 8

    # Reranking provider
    rerank_provider: str | None = None  # If None, uses embedding provider if available
    rerank_api_key: str | None = None  # If None, uses embedding API key
    rerank_model: str | None = None  # If None, uses provider default

    # Provider-specific settings
    base_url: str | None = None
    custom_headers: dict[str, str] = field(default_factory=dict)

    # Local model settings
    use_local: bool = False
    device: str = "auto"  # "cpu", "cuda", "auto"
    normalize_embeddings: bool = True

    # Advanced settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600


@dataclass
class DataSourceConfig:
    """Data source configuration."""

    enabled: bool = True
    default_source_type: str = "filesystem"
    max_concurrent_sources: int = 5

    # Content processing
    enable_content_deduplication: bool = True
    content_cache_ttl_hours: int = 24
    enable_metadata_extraction: bool = True

    # Source definitions
    sources: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CodeWeaverConfig:
    """Main configuration for Code Weaver MCP server."""

    # New extensible configuration (primary)
    backend: BackendConfig = field(default_factory=BackendConfig)
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    data_sources: DataSourceConfig = field(default_factory=DataSourceConfig)

    # Legacy configuration (for backward compatibility)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)

    # Shared configuration
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    rate_limiting: RateLimitConfig = field(default_factory=RateLimitConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    # Migration and compatibility settings
    _config_version: str = "2.0"  # Track configuration schema version
    _migrated_from_legacy: bool = False  # Track if config was migrated

    def merge_from_dict(self, config_dict: dict[str, Any]) -> None:
        """Merge configuration from a dictionary with migration support."""
        # Check if this is a legacy configuration
        has_legacy_sections = any(key in config_dict for key in ["embedding", "qdrant"])
        has_new_sections = any(
            key in config_dict for key in ["backend", "provider", "data_sources"]
        )

        # If we have legacy sections but no new sections, perform migration
        if has_legacy_sections and not has_new_sections:
            logger.info("Detected legacy configuration, performing migration")
            self._migrate_legacy_config(config_dict)
            self._migrated_from_legacy = True

        # Process configuration sections
        for section_name, section_data in config_dict.items():
            if hasattr(self, section_name) and isinstance(section_data, dict):
                section = getattr(self, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
                    else:
                        logger.warning("Unknown config key: %s.%s", section_name, key)
            elif section_name.startswith("_"):
                # Handle private configuration fields
                if hasattr(self, section_name):
                    setattr(self, section_name, section_data)
            else:
                logger.warning("Unknown config section: %s", section_name)

        # Handle special case for configuration version tracking
        if "_config_version" in config_dict:
            self._config_version = config_dict["_config_version"]
        if "_migrated_from_legacy" in config_dict:
            self._migrated_from_legacy = config_dict["_migrated_from_legacy"]

    def _migrate_legacy_config(self, config_dict: dict[str, Any]) -> None:
        """Migrate legacy configuration to new format."""
        # Migrate embedding configuration to provider configuration
        if "embedding" in config_dict:
            embedding_config = config_dict["embedding"]
            if isinstance(embedding_config, dict):
                provider_dict = {
                    "embedding_provider": embedding_config.get("provider", "voyage"),
                    "embedding_api_key": embedding_config.get("api_key"),
                    "embedding_model": embedding_config.get("model", "voyage-code-3"),
                    "embedding_dimension": embedding_config.get("dimension", 1024),
                    "embedding_batch_size": embedding_config.get("batch_size", 8),
                    "rerank_provider": embedding_config.get("rerank_provider"),
                    "rerank_model": embedding_config.get("rerank_model"),
                    "base_url": embedding_config.get("base_url"),
                    "custom_headers": embedding_config.get("custom_headers", {}),
                    "use_local": embedding_config.get("use_local", False),
                    "device": embedding_config.get("device", "auto"),
                    "normalize_embeddings": embedding_config.get("normalize_embeddings", True),
                }
                config_dict["provider"] = provider_dict

        # Migrate Qdrant configuration to backend configuration
        if "qdrant" in config_dict:
            qdrant_config = config_dict["qdrant"]
            if isinstance(qdrant_config, dict):
                backend_dict = {
                    "provider": "qdrant",
                    "url": qdrant_config.get("url"),
                    "api_key": qdrant_config.get("api_key"),
                    "collection_name": qdrant_config.get("collection_name", "code-embeddings"),
                    "enable_sparse_vectors": qdrant_config.get("enable_sparse_vectors", False),
                    "enable_hybrid_search": qdrant_config.get("enable_sparse_vectors", False),
                }
                config_dict["backend"] = backend_dict

        # Create default data sources configuration if not present
        if "data_sources" not in config_dict:
            data_sources_dict = {
                "enabled": True,
                "default_source_type": "filesystem",
                "sources": [
                    {
                        "type": "filesystem",
                        "enabled": True,
                        "priority": 1,
                        "source_id": "default_filesystem",
                        "config": {
                            "root_path": ".",
                            "use_gitignore": config_dict.get("indexing", {}).get(
                                "use_gitignore", True
                            ),
                            "additional_ignore_patterns": config_dict.get("indexing", {}).get(
                                "additional_ignore_patterns", []
                            ),
                            "max_file_size_mb": config_dict.get("chunking", {}).get(
                                "max_file_size_mb", 1
                            ),
                        },
                    }
                ],
            }
            config_dict["data_sources"] = data_sources_dict

        logger.info("Successfully migrated legacy configuration to new format")

    def merge_from_env(self) -> None:
        """Merge configuration from environment variables with new and legacy support."""
        # New provider configuration
        if provider := os.getenv("EMBEDDING_PROVIDER"):
            self.provider.embedding_provider = provider.lower()
        if api_key := os.getenv("EMBEDDING_API_KEY"):
            self.provider.embedding_api_key = api_key
        if model := os.getenv("EMBEDDING_MODEL"):
            self.provider.embedding_model = model
        if dimension := os.getenv("EMBEDDING_DIMENSION"):
            self.provider.embedding_dimension = int(dimension)

        # Provider-specific API keys
        if api_key := os.getenv("VOYAGE_API_KEY"):
            if self.provider.embedding_provider == "voyage":
                self.provider.embedding_api_key = api_key
            # Also set legacy for backward compatibility
            self.embedding.api_key = api_key
        if api_key := os.getenv("OPENAI_API_KEY"):
            if self.provider.embedding_provider == "openai":
                self.provider.embedding_api_key = api_key
            if self.embedding.provider == "openai":
                self.embedding.api_key = api_key
        if api_key := os.getenv("COHERE_API_KEY"):
            if self.provider.embedding_provider == "cohere":
                self.provider.embedding_api_key = api_key
            if self.embedding.provider == "cohere":
                self.embedding.api_key = api_key
        if api_key := os.getenv("HUGGINGFACE_API_KEY"):
            if self.provider.embedding_provider == "huggingface":
                self.provider.embedding_api_key = api_key
            if self.embedding.provider == "huggingface":
                self.embedding.api_key = api_key

        # Reranking configuration
        if rerank_provider := os.getenv("RERANK_PROVIDER"):
            self.provider.rerank_provider = rerank_provider.lower()
            self.embedding.rerank_provider = rerank_provider.lower()
        if rerank_model := os.getenv("RERANK_MODEL"):
            self.provider.rerank_model = rerank_model
            self.embedding.rerank_model = rerank_model

        # Local model settings
        if use_local := os.getenv("USE_LOCAL_MODELS"):
            value = use_local.lower() in ("true", "1", "yes")
            self.provider.use_local = value
            self.embedding.use_local = value
        if device := os.getenv("MODEL_DEVICE"):
            self.provider.device = device.lower()
            self.embedding.device = device.lower()

        # New backend configuration
        if backend_provider := os.getenv("VECTOR_BACKEND_PROVIDER"):
            self.backend.provider = backend_provider.lower()
        if backend_url := os.getenv("VECTOR_BACKEND_URL"):
            self.backend.url = backend_url
        if backend_key := os.getenv("VECTOR_BACKEND_API_KEY"):
            self.backend.api_key = backend_key

        # Legacy backend configuration (Qdrant)
        if url := os.getenv("QDRANT_URL"):
            self.backend.url = url  # New config
            self.qdrant.url = url  # Legacy config
        if api_key := os.getenv("QDRANT_API_KEY"):
            self.backend.api_key = api_key  # New config
            self.qdrant.api_key = api_key  # Legacy config
        if collection := os.getenv("COLLECTION_NAME"):
            self.backend.collection_name = collection
            self.qdrant.collection_name = collection

        # Feature flags
        if hybrid := os.getenv("ENABLE_HYBRID_SEARCH"):
            value = hybrid.lower() in ("true", "1", "yes")
            self.backend.enable_hybrid_search = value
        if sparse := os.getenv("ENABLE_SPARSE_VECTORS"):
            value = sparse.lower() in ("true", "1", "yes")
            self.backend.enable_sparse_vectors = value
            self.qdrant.enable_sparse_vectors = value

        # Legacy embedding configuration (for backward compatibility)
        if model := os.getenv("VOYAGE_MODEL"):
            self.embedding.model = model
            if self.provider.embedding_provider == "voyage":
                self.provider.embedding_model = model
        if provider := os.getenv("EMBEDDING_PROVIDER"):
            self.embedding.provider = provider.lower()
        if base_url := os.getenv("OPENAI_BASE_URL"):
            self.embedding.base_url = base_url
            self.provider.base_url = base_url

        # Server configuration
        if log_level := os.getenv("LOG_LEVEL"):
            self.server.log_level = log_level.upper()

        # Sync legacy and new configurations
        self._sync_legacy_and_new_configs()

    def get_effective_embedding_provider(self) -> str:
        """Get the effective embedding provider (prefer new config)."""
        return self.provider.embedding_provider or self.embedding.provider

    def get_effective_backend_provider(self) -> str:
        """Get the effective backend provider."""
        return self.backend.provider

    def get_effective_backend_url(self) -> str | None:
        """Get the effective backend URL (prefer new config)."""
        return self.backend.url or self.qdrant.url

    def get_effective_backend_api_key(self) -> str | None:
        """Get the effective backend API key (prefer new config)."""
        return self.backend.api_key or self.qdrant.api_key

    def is_legacy_config(self) -> bool:
        """Check if this is primarily a legacy configuration."""
        return (self.embedding.api_key or self.qdrant.url) and not (
            self.provider.embedding_api_key or self.backend.url
        )

    def to_new_format_dict(self) -> dict[str, Any]:
        """Convert configuration to new format dictionary."""
        return {
            "backend": {
                "provider": self.get_effective_backend_provider(),
                "url": self.get_effective_backend_url(),
                "api_key": self.get_effective_backend_api_key(),
                "collection_name": self.backend.collection_name,
                "enable_hybrid_search": self.backend.enable_hybrid_search,
                "enable_sparse_vectors": self.backend.enable_sparse_vectors,
            },
            "provider": {
                "embedding_provider": self.get_effective_embedding_provider(),
                "embedding_api_key": self.provider.embedding_api_key or self.embedding.api_key,
                "embedding_model": self.provider.embedding_model or self.embedding.model,
                "embedding_dimension": self.provider.embedding_dimension
                or self.embedding.dimension,
                "rerank_provider": self.provider.rerank_provider or self.embedding.rerank_provider,
                "rerank_model": self.provider.rerank_model or self.embedding.rerank_model,
            },
            "data_sources": {
                "enabled": self.data_sources.enabled,
                "sources": self.data_sources.sources,
            },
            "_config_version": "2.0",
        }

    def _sync_legacy_and_new_configs(self) -> None:
        """Synchronize legacy and new configuration formats."""
        # Sync provider config with legacy embedding config
        if self.provider.embedding_api_key and not self.embedding.api_key:
            self.embedding.api_key = self.provider.embedding_api_key
        elif self.embedding.api_key and not self.provider.embedding_api_key:
            self.provider.embedding_api_key = self.embedding.api_key

        # Sync backend config with legacy Qdrant config
        if self.backend.url and not self.qdrant.url:
            self.qdrant.url = self.backend.url
        elif self.qdrant.url and not self.backend.url:
            self.backend.url = self.qdrant.url

        if self.backend.api_key and not self.qdrant.api_key:
            self.qdrant.api_key = self.backend.api_key
        elif self.qdrant.api_key and not self.backend.api_key:
            self.backend.api_key = self.qdrant.api_key


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

    def _validate_config(self, config: "CodeWeaverConfig") -> None:
        """Validate the configuration and raise errors for missing required values."""
        # Get effective provider
        provider = config.get_effective_embedding_provider().lower()

        # Validate embedding provider
        if provider in ["voyage", "openai", "cohere", "huggingface"]:
            # These providers require API keys (unless using local models)
            api_key = config.provider.embedding_api_key or config.embedding.api_key
            use_local = config.provider.use_local or config.embedding.use_local

            if provider == "huggingface" and use_local:
                # Local HuggingFace models don't require API key
                pass
            elif provider == "sentence-transformers":
                # Local models don't require API key
                pass
            elif not api_key:
                provider_env_map = {
                    "voyage": "VOYAGE_API_KEY",
                    "openai": "OPENAI_API_KEY",
                    "cohere": "COHERE_API_KEY",
                    "huggingface": "HUGGINGFACE_API_KEY",
                }
                env_var = provider_env_map.get(provider, f"{provider.upper()}_API_KEY")
                raise ValueError(f"{env_var} is required when using {provider} embeddings")
        elif provider == "sentence-transformers":
            # Local provider, no API key required
            pass
        else:
            # Try to use provider factory to validate
            try:
                from codeweaver.providers import get_provider_factory

                factory = get_provider_factory()
                if not factory.registry.is_embedding_provider_available(provider):
                    available = list(factory.registry.get_available_embedding_providers().keys())
                    raise ValueError(
                        f"Unknown or unavailable embedding provider: {provider}. "
                        f"Available providers: {', '.join(available)}"
                    )
            except ImportError:
                # Provider system not available, use legacy validation
                logger.warning(
                    "Provider validation unavailable, provider %s may not work", provider
                )

        # Validate backend configuration
        backend_url = config.get_effective_backend_url()
        if not backend_url:
            backend_provider = config.get_effective_backend_provider()
            if backend_provider == "qdrant":
                raise ValueError("QDRANT_URL or VECTOR_BACKEND_URL is required")
            raise ValueError(f"Backend URL is required for {backend_provider} provider")

        # Validate backend provider
        backend_provider = config.get_effective_backend_provider()
        supported_backends = [
            "qdrant",
            "pinecone",
            "chroma",
            "weaviate",
            "pgvector",
            "milvus",
            "elasticsearch",
        ]
        if backend_provider not in supported_backends:
            logger.warning(
                "Backend provider %s may not be supported. Supported: %s",
                backend_provider,
                ", ".join(supported_backends),
            )

        # Validate data sources
        if config.data_sources.enabled and not config.data_sources.sources:
            logger.warning("Data sources are enabled but no sources are configured")

        # Validate rate limiting values
        if config.rate_limiting.max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if config.rate_limiting.initial_backoff_seconds <= 0:
            raise ValueError("initial_backoff_seconds must be > 0")

    def get_example_config(self, format: Literal["new", "legacy", "migration"] = "new") -> str:
        """Get an example TOML configuration file."""
        if format == "new":
            return self._get_new_format_example()
        if format == "legacy":
            return self._get_legacy_format_example()
        # migration
        return self._get_migration_example()

    def _get_new_format_example(self) -> str:
        """Get example configuration in new format."""
        return """# Code Weaver MCP Server Configuration (v2.0)
# Place this file in one of these locations (highest precedence first):
# 1. .local.code-weaver.toml (workspace local)
# 2. .code-weaver.toml (repository)
# 3. ~/.config/code-weaver/config.toml (user)

# Vector Database Backend Configuration
[backend]
provider = "qdrant"  # qdrant, pinecone, chroma, weaviate, pgvector, milvus, elasticsearch
url = "YOUR_BACKEND_URL"  # Can also use VECTOR_BACKEND_URL or QDRANT_URL env var
api_key = "YOUR_BACKEND_API_KEY"  # Can also use VECTOR_BACKEND_API_KEY or QDRANT_API_KEY env var
collection_name = "code-embeddings"

# Feature capabilities
enable_hybrid_search = false
enable_sparse_vectors = false
enable_streaming = false

# Performance settings
batch_size = 100
connection_timeout = 30.0
request_timeout = 60.0
max_connections = 10

# Provider-specific options
[backend.provider_options]
# For Pinecone:
# environment = "us-west1-gcp"
# For Weaviate:
# timeout_config = [30.0, 60.0]

# Embedding and Reranking Provider Configuration
[provider]
embedding_provider = "voyage"  # voyage, openai, cohere, sentence-transformers, huggingface
embedding_api_key = "YOUR_EMBEDDING_API_KEY"  # Can also use VOYAGE_API_KEY, OPENAI_API_KEY, etc.
embedding_model = "voyage-code-3"
embedding_dimension = 1024
embedding_batch_size = 8

# Reranking configuration
rerank_provider = "voyage"  # If None, uses embedding provider if available
rerank_model = "voyage-rerank-2"  # If None, uses provider default

# Provider-specific settings
# base_url = "http://localhost:8000/v1"  # For OpenAI-compatible providers
# use_local = false  # For local models (sentence-transformers, huggingface)
# device = "auto"  # cpu, cuda, auto
normalize_embeddings = true

# Advanced provider settings
enable_caching = true
cache_ttl_seconds = 3600

# Data Sources Configuration
[data_sources]
enabled = true
default_source_type = "filesystem"
max_concurrent_sources = 5
enable_content_deduplication = true
content_cache_ttl_hours = 24
enable_metadata_extraction = true

# File System Source (primary)
[[data_sources.sources]]
type = "filesystem"
enabled = true
priority = 1
source_id = "main_codebase"

[data_sources.sources.config]
root_path = "."
use_gitignore = true
additional_ignore_patterns = [
    "node_modules", ".git", ".venv", "venv", "__pycache__",
    "target", "build", "dist", ".next", ".nuxt", "coverage"
]
max_file_size_mb = 1
batch_size = 8
enable_change_watching = false
change_check_interval_seconds = 60

# Git Repository Source (example)
# [[data_sources.sources]]
# type = "git"
# enabled = false
# priority = 2
# source_id = "external_repo"
#
# [data_sources.sources.config]
# repository_url = "https://github.com/user/repo.git"
# branch = "main"
# local_clone_path = "/tmp/codeweaver_repos/external_repo"

# Legacy Configuration (for backward compatibility)
[embedding]
provider = "voyage"  # Kept for backward compatibility
api_key = "YOUR_VOYAGE_API_KEY"  # Synced with provider.embedding_api_key
model = "voyage-code-3"  # Synced with provider.embedding_model

[qdrant]
url = "YOUR_QDRANT_URL"  # Synced with backend.url
api_key = "YOUR_QDRANT_API_KEY"  # Synced with backend.api_key
collection_name = "code-embeddings"

# Shared Configuration
[chunking]
max_chunk_size = 1500
min_chunk_size = 50
max_file_size_mb = 1

[indexing]
use_gitignore = true
additional_ignore_patterns = [
    "node_modules", ".git", ".venv", "venv", "__pycache__",
    "target", "build", "dist", ".next", ".nuxt", "coverage"
]
enable_auto_reindex = false
watch_debounce_seconds = 2.0
batch_size = 8
max_concurrent_files = 10

[rate_limiting]
voyage_requests_per_minute = 100
voyage_tokens_per_minute = 1000000
openai_requests_per_minute = 5000
openai_tokens_per_minute = 1000000
qdrant_requests_per_second = 100
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

    def _get_legacy_format_example(self) -> str:
        """Get example configuration in legacy format."""
        return """# Code Weaver MCP Server Configuration (Legacy Format)
# This format is supported for backward compatibility

[embedding]
provider = "voyage"  # "voyage" or "openai"
api_key = "YOUR_VOYAGE_API_KEY"  # Can also use VOYAGE_API_KEY env var
model = "voyage-code-3"
dimension = 1024
batch_size = 8

[qdrant]
url = "YOUR_QDRANT_URL"  # Can also use QDRANT_URL env var
api_key = "YOUR_QDRANT_API_KEY"  # Can also use QDRANT_API_KEY env var
collection_name = "code-embeddings"
enable_sparse_vectors = false

[chunking]
max_chunk_size = 1500
min_chunk_size = 50
max_file_size_mb = 1

[indexing]
use_gitignore = true
additional_ignore_patterns = [
    "node_modules", ".git", ".venv", "venv", "__pycache__",
    "target", "build", "dist", ".next", ".nuxt", "coverage"
]
enable_auto_reindex = false
watch_debounce_seconds = 2.0
batch_size = 8
max_concurrent_files = 10

[rate_limiting]
voyage_requests_per_minute = 100
voyage_tokens_per_minute = 1000000
openai_requests_per_minute = 5000
openai_tokens_per_minute = 1000000
qdrant_requests_per_second = 100
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

    def _get_migration_example(self) -> str:
        """Get example showing migration from legacy to new format."""
        return """# Code Weaver Configuration Migration Example
# This shows both legacy and new formats side by side

# =============================================================================
# LEGACY FORMAT (will be automatically migrated)
# =============================================================================

# [embedding]
# provider = "voyage"
# api_key = "your-voyage-key"
# model = "voyage-code-3"
#
# [qdrant]
# url = "https://your-cluster.qdrant.io"
# api_key = "your-qdrant-key"
# collection_name = "code-embeddings"

# =============================================================================
# NEW FORMAT (recommended for new installations)
# =============================================================================

[backend]
provider = "qdrant"
url = "https://your-cluster.qdrant.io"
api_key = "your-qdrant-key"
collection_name = "code-embeddings"
enable_hybrid_search = false

[provider]
embedding_provider = "voyage"
embedding_api_key = "your-voyage-key"
embedding_model = "voyage-code-3"
embedding_dimension = 1024

[data_sources]
enabled = true

[[data_sources.sources]]
type = "filesystem"
enabled = true
priority = 1

[data_sources.sources.config]
root_path = "."
use_gitignore = true

# =============================================================================
# MIGRATION NOTES
# =============================================================================
#
# 1. Legacy configurations are automatically detected and migrated
# 2. Both formats can coexist during transition
# 3. Environment variables work with both formats
# 4. New format enables multi-backend and multi-source support
# 5. Run with LOG_LEVEL=DEBUG to see migration details

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


def create_example_configs() -> dict[str, str]:
    """Create example configurations for different scenarios."""
    config_manager = get_config_manager()

    return {
        "new_format": config_manager.get_example_config("new"),
        "legacy_format": config_manager.get_example_config("legacy"),
        "migration_guide": config_manager.get_example_config("migration"),
    }


def validate_environment_variables() -> dict[str, Any]:
    """Validate current environment variables for configuration."""
    validation_results = {"valid": True, "warnings": [], "errors": [], "detected_vars": {}}

    # Check for provider-specific API keys
    provider_vars = {
        "VOYAGE_API_KEY": "Voyage AI",
        "OPENAI_API_KEY": "OpenAI",
        "COHERE_API_KEY": "Cohere",
        "HUGGINGFACE_API_KEY": "HuggingFace",
    }

    for var, provider in provider_vars.items():
        if os.getenv(var):
            validation_results["detected_vars"][var] = f"{provider} API key detected"

    # Check backend configuration
    backend_vars = {
        "QDRANT_URL": "Qdrant URL (legacy)",
        "QDRANT_API_KEY": "Qdrant API key (legacy)",
        "VECTOR_BACKEND_URL": "Backend URL (new)",
        "VECTOR_BACKEND_API_KEY": "Backend API key (new)",
        "VECTOR_BACKEND_PROVIDER": "Backend provider (new)",
    }

    for var, description in backend_vars.items():
        if os.getenv(var):
            validation_results["detected_vars"][var] = description

    # Check for required variables
    if not any(os.getenv(var) for var in ["QDRANT_URL", "VECTOR_BACKEND_URL"]):
        validation_results["errors"].append(
            "No backend URL found. Set QDRANT_URL or VECTOR_BACKEND_URL"
        )
        validation_results["valid"] = False

    # Check for embedding provider API key
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "voyage").lower()
    use_local = os.getenv("USE_LOCAL_MODELS", "false").lower() in ("true", "1", "yes")

    if not use_local and embedding_provider in ["voyage", "openai", "cohere", "huggingface"]:
        provider_key_map = {
            "voyage": "VOYAGE_API_KEY",
            "openai": "OPENAI_API_KEY",
            "cohere": "COHERE_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
        }

        required_key = provider_key_map.get(embedding_provider)
        if required_key and not os.getenv(required_key):
            validation_results["errors"].append(
                f"No API key found for {embedding_provider} provider. Set {required_key}"
            )
            validation_results["valid"] = False

    # Check for mixed legacy/new configuration
    has_legacy = any(os.getenv(var) for var in ["QDRANT_URL", "VOYAGE_API_KEY"])
    has_new = any(os.getenv(var) for var in ["VECTOR_BACKEND_URL", "EMBEDDING_API_KEY"])

    if has_legacy and has_new:
        validation_results["warnings"].append(
            "Mixed legacy and new environment variables detected. "
             "Consider migrating to new format for consistency."
        )

    return validation_results


def get_effective_config_summary() -> dict[str, Any]:
    """Get a summary of the effective configuration."""
    try:
        config = get_config()

        return {
            "config_version": getattr(config, "_config_version", "1.0"),
            "migrated_from_legacy": getattr(config, "_migrated_from_legacy", False),
            "is_legacy_config": config.is_legacy_config(),
            "backend": {
                "provider": config.get_effective_backend_provider(),
                "url": config.get_effective_backend_url(),
                "has_api_key": bool(config.get_effective_backend_api_key()),
                "collection_name": config.backend.collection_name,
                "hybrid_search_enabled": config.backend.enable_hybrid_search,
            },
            "embedding": {
                "provider": config.get_effective_embedding_provider(),
                "model": config.provider.embedding_model or config.embedding.model,
                "dimension": config.provider.embedding_dimension or config.embedding.dimension,
                "has_api_key": bool(config.provider.embedding_api_key or config.embedding.api_key),
                "use_local": config.provider.use_local or config.embedding.use_local,
            },
            "data_sources": {
                "enabled": config.data_sources.enabled,
                "source_count": len(config.data_sources.sources),
                "source_types": list(
                    {source.get("type", "unknown") for source in config.data_sources.sources}
                ),
            },
        }
    except Exception as e:
        return {"error": str(e)}


# Utility functions for configuration management
def is_configuration_migrated() -> bool:
    """Check if configuration has been migrated from legacy format."""
    try:
        config = get_config()
        return getattr(config, "_migrated_from_legacy", False)
    except Exception:
        return False


def suggest_configuration_improvements() -> list[str]:
    """Suggest improvements to the current configuration."""
    suggestions = []

    try:
        config = get_config()

        # Check if using legacy format
        if config.is_legacy_config():
            suggestions.append(
                "Consider migrating to the new configuration format for access to "
                "multi-backend support and enhanced features."
            )

        # Check for hybrid search opportunities
        backend = config.get_effective_backend_provider()
        if backend in ["qdrant", "weaviate", "milvus"] and not config.backend.enable_hybrid_search:
            suggestions.append(
                f"Your {backend} backend supports hybrid search. "
                "Consider enabling it for better search quality."
            )

        # Check for reranking opportunities
        provider = config.get_effective_embedding_provider()
        rerank_provider = config.provider.rerank_provider or config.embedding.rerank_provider
        if provider in ["voyage", "cohere"] and not rerank_provider:
            suggestions.append(
                f"Your {provider} provider supports reranking. "
                "Consider enabling it for improved search relevance."
            )

        # Check data sources configuration
        if config.data_sources.enabled and len(config.data_sources.sources) == 1:
            suggestions.append(
                "You're using only one data source. Consider adding additional sources "
                "like Git repositories or APIs for comprehensive code search."
            )

        # Check for performance optimizations
        if config.backend.batch_size < 50:
            suggestions.append(
                "Consider increasing backend batch_size to 100+ for better performance "
                "with large codebases."
            )

    except Exception as e:
        suggestions.append(f"Unable to analyze configuration: {e}")

    return suggestions
