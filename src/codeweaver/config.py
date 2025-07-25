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

from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

import tomlkit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# Import new configuration components
try:
    from codeweaver._types.provider_enums import ProviderType
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
    from codeweaver.sources.config import DataSourcesConfig

    _EXTENDED_CONFIGS_AVAILABLE = True
except ImportError:
    # Fallback for when extended configurations are not available
    _EXTENDED_CONFIGS_AVAILABLE = False
    BackendConfigExtended = None
    EmbeddingProviderConfig = None
    RerankingProviderConfig = None
    CombinedProviderConfig = None
    VoyageConfig = None
    OpenAIConfig = None
    OpenAICompatibleConfig = None
    CohereConfig = None
    HuggingFaceConfig = None
    SentenceTransformersConfig = None
    DataSourcesConfig = None
    ProviderType = None

logger = logging.getLogger(__name__)


class EmbeddingConfig(BaseModel):
    """Configuration for embedding providers."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    provider: str = Field(
        default="voyage-ai", description="Embedding provider (voyage, openai, cohere, etc.)"
    )
    api_key: str | None = Field(default=None, description="API key for the embedding provider")
    model: str = Field(default="voyage-code-3", description="Embedding model to use")
    dimension: int = Field(default=1024, ge=1, le=4096, description="Embedding dimension")
    batch_size: int = Field(default=8, ge=1, le=100, description="Batch size for embeddings")

    # OpenAI-compatible provider settings
    base_url: str | None = Field(
        default=None, description="Base URL for OpenAI-compatible providers"
    )
    custom_headers: dict[str, str] = Field(
        default_factory=dict, description="Custom headers for API requests"
    )

    # Reranking configuration
    rerank_provider: str | None = Field(
        default=None, description="Reranking provider (if different from embedding)"
    )
    rerank_model: str | None = Field(default=None, description="Reranking model to use")

    # Local model settings (for sentence-transformers, huggingface local)
    use_local: bool = Field(default=False, description="Use local models instead of API")
    device: str = Field(default="auto", description="Device for local models (cpu, cuda, auto)")
    normalize_embeddings: bool = Field(default=True, description="Whether to normalize embeddings")


class QdrantConfig(BaseModel):
    """Configuration for Qdrant vector database."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    url: str | None = Field(default=None, description="Qdrant server URL")
    api_key: str | None = Field(default=None, description="Qdrant API key")
    collection_name: str = Field(default="code-embeddings", description="Qdrant collection name")

    # Hybrid search settings
    enable_sparse_vectors: bool = Field(
        default=False, description="Enable sparse vectors for hybrid search"
    )
    sparse_vector_name: str = Field(default="sparse", description="Name of the sparse vector field")


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

    @field_validator("max_chunk_size", "min_chunk_size")
    @classmethod
    def validate_chunk_sizes(cls, v: int, info) -> int:
        """Validate chunk size constraints."""
        if (
            info.field_name == "max_chunk_size"
            and hasattr(info.data, "min_chunk_size")
            and v <= info.data.get("min_chunk_size", 50)
        ):
            raise ValueError("max_chunk_size must be greater than min_chunk_size")
        return v


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

    server_name: str = Field(default="code-weaver-mcp", description="MCP server name")
    server_version: str = Field(default="2.0.0", description="MCP server version")
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


# Temporarily commented out to resolve circular dependency during testing
# TODO: Move to separate module or restructure imports
# class ModernProviderConfig(BaseModel):
#     """Modern provider configuration using new Pydantic models."""
#
#     model_config = ConfigDict(extra="allow", validate_assignment=True)
#
#     # Primary embedding provider
#     embedding: "EmbeddingProviderConfig | CombinedProviderConfig | None" = Field(
#         default=None, description="Primary embedding provider configuration"
#     )
#
#     # Primary reranking provider (optional, can be same as embedding if CombinedProvider)
#     reranking: "RerankingProviderConfig | CombinedProviderConfig | None" = Field(
#         default=None, description="Primary reranking provider configuration"
#     )
#
#     # Provider-specific configurations
#     voyage: "VoyageConfig | None" = Field(default=None, description="Voyage AI provider configuration")
#     openai: "OpenAIConfig | None" = Field(default=None, description="OpenAI provider configuration")
#     cohere: "CohereConfig | None" = Field(default=None, description="Cohere provider configuration")
#     huggingface: "HuggingFaceConfig | None" = Field(default=None, description="HuggingFace provider configuration")
#     sentence_transformers: "SentenceTransformersConfig | None" = Field(
#         default=None, description="SentenceTransformers provider configuration"
#     )
#
#     # Advanced settings
#     enable_caching: bool = Field(default=True, description="Enable provider response caching")
#     cache_ttl_seconds: Annotated[int, Field(default=3600, ge=60, le=86400)] = Field(description="Cache TTL in seconds")
#
#     def get_provider_config(self, provider_type: "ProviderType") -> "EmbeddingProviderConfig | RerankingProviderConfig | CombinedProviderConfig | None":
#         """Get configuration for a specific provider type."""
#         provider_map = {
#             ProviderType.VOYAGE_AI: self.voyage,
#             ProviderType.OPENAI: self.openai,
#             ProviderType.COHERE: self.cohere,
#             ProviderType.HUGGINGFACE: self.huggingface,
#             ProviderType.SENTENCE_TRANSFORMERS: self.sentence_transformers,
#         }
#         return provider_map.get(provider_type)
#
#     def get_active_embedding_provider(self) -> EmbeddingProviderConfig | CombinedProviderConfig | None:
#         """Get the active embedding provider configuration."""
#         return self.embedding
#
#     def get_active_reranking_provider(self) -> RerankingProviderConfig | CombinedProviderConfig | None:
#         """Get the active reranking provider configuration."""
#         return self.reranking


# Modern provider configuration using new Pydantic models
class ModernProviderConfig(BaseModel):
    """Modern provider configuration using new Pydantic models."""

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
    voyage: VoyageConfig | None = Field(
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

    def get_provider_config(self, provider_type: ProviderType) -> Any | None:
        """Get configuration for a specific provider type."""
        provider_map = {
            ProviderType.VOYAGE_AI: self.voyage,
            ProviderType.OPENAI: self.openai,
            ProviderType.COHERE: self.cohere,
            ProviderType.HUGGINGFACE: self.huggingface,
            ProviderType.SENTENCE_TRANSFORMERS: self.sentence_transformers,
            ProviderType.OPENAI_COMPATIBLE: self.openai_compatible,
        }
        return provider_map.get(provider_type)

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


class CodeWeaverConfig(BaseModel):
    """Main configuration for Code Weaver MCP server.

    Provides comprehensive TOML-based configuration with hierarchical loading,
    validation, and backward compatibility support.
    """

    model_config = ConfigDict(
        extra="allow", validate_assignment=True, str_strip_whitespace=True, use_enum_values=True
    )

    # New extensible configuration (primary)
    backend: BackendConfigExtended = Field(
        default_factory=BackendConfigExtended, description="Vector database backend configuration"
    )
    providers: ModernProviderConfig = Field(
        default_factory=ModernProviderConfig, description="Modern provider configuration"
    )
    data_sources: DataSourceConfig = Field(
        default_factory=DataSourceConfig, description="Data source configuration"
    )

    # Legacy configuration (maintained for compatibility during transition)
    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig, description="Legacy embedding configuration"
    )
    qdrant: QdrantConfig = Field(
        default_factory=QdrantConfig, description="Legacy Qdrant configuration"
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

    # Migration and compatibility settings
    config_version: str = Field(
        default="2.0", alias="_config_version", description="Configuration schema version"
    )
    migrated_from_legacy: bool = Field(
        default=False,
        alias="_migrated_from_legacy",
        description="Whether config was migrated from legacy format",
    )

    @model_validator(mode="after")
    def validate_config_consistency(self) -> "CodeWeaverConfig":
        """Validate configuration consistency across sections."""
        # Ensure chunk sizes are consistent
        if self.chunking.max_chunk_size <= self.chunking.min_chunk_size:
            raise ValueError("max_chunk_size must be greater than min_chunk_size")

        # Modern provider configs are now the primary configuration
        # Legacy initialization no longer needed

        return self

    def merge_from_dict(self, parsed_config: dict[str, Any]) -> None:
        """Merge configuration from a dictionary with migration support."""
        # Check if this is a legacy configuration
        has_legacy_sections = any(key in parsed_config for key in ["embedding", "qdrant"])
        has_new_sections = any(
            key in parsed_config for key in ["backend", "providers", "data_sources"]
        )
        has_mixed_provider_sections = "provider" in parsed_config and "providers" in parsed_config

        # If we have legacy sections but no new sections, perform migration
        if has_legacy_sections and not has_new_sections:
            logger.info("Detected legacy configuration, performing migration")
            self._migrate_legacy_config(parsed_config)
            self.migrated_from_legacy = True

        # Handle mixed provider configurations (both old "provider" and new "providers")
        if has_mixed_provider_sections:
            logger.info("Detected mixed provider configuration, prioritizing new format")
            # Legacy provider config is kept for backward compatibility

        # Process configuration sections using Pydantic model updates
        for section_name, section_data in parsed_config.items():
            if hasattr(self, section_name) and isinstance(section_data, dict):
                # Get the current section model
                current_section = getattr(self, section_name)
                if isinstance(current_section, BaseModel):
                    # Use Pydantic's model_validate to merge with validation
                    try:
                        # Create a new instance with merged data
                        current_data = current_section.model_dump()
                        current_data.update(section_data)
                        updated_section = current_section.__class__.model_validate(current_data)
                        setattr(self, section_name, updated_section)
                        logger.debug("Updated section: %s", section_name)
                    except Exception as e:
                        logger.warning("Failed to update section %s: %s", section_name, e)
                        # Fallback to individual field updates
                        for key, value in section_data.items():
                            if hasattr(current_section, key):
                                try:
                                    setattr(current_section, key, value)
                                except Exception as field_error:
                                    logger.warning(
                                        "Failed to set %s.%s: %s", section_name, key, field_error
                                    )
                            else:
                                logger.warning("Unknown config key: %s.%s", section_name, key)
                else:
                    # Fallback for non-Pydantic sections
                    for key, value in section_data.items():
                        if hasattr(current_section, key):
                            setattr(current_section, key, value)
                        else:
                            logger.warning("Unknown config key: %s.%s", section_name, key)
            elif section_name.startswith("_") or section_name == "_metadata":
                # Handle private configuration fields and metadata
                if section_name == "_metadata" and isinstance(section_data, dict):
                    # Extract metadata fields
                    if "_config_version" in section_data:
                        self.config_version = section_data["_config_version"]
                    if "_migrated_from_legacy" in section_data:
                        self.migrated_from_legacy = section_data["_migrated_from_legacy"]
                elif hasattr(self, section_name):
                    setattr(self, section_name, section_data)
            else:
                logger.warning("Unknown config section: %s", section_name)

        # Handle special case for configuration version tracking (legacy format)
        if "_config_version" in parsed_config:
            self.config_version = parsed_config["_config_version"]
        if "_migrated_from_legacy" in parsed_config:
            self.migrated_from_legacy = parsed_config["_migrated_from_legacy"]

    def merge_from_env(self) -> None:
        """Merge configuration from environment variables with new and legacy support."""
        # New provider configuration
        if provider := os.getenv("CW_EMBEDDING_PROVIDER"):
            self.provider.embedding_provider = provider.lower()
            # Also update modern provider config if applicable
            if hasattr(self.providers, provider.lower()):
                setattr(self.providers, provider.lower(), True)
        if api_key := os.getenv("CW_EMBEDDING_API_KEY"):
            self.provider.CW_EMBEDDING_API_KEY = api_key
            # Update modern provider configs
            if self.providers.embedding:
                self.providers.embedding.api_key = api_key
        if model := os.getenv("CW_EMBEDDING_MODEL"):
            self.embedding.model = model
            if self.providers.embedding:
                self.providers.embedding.model = model
        if dimension := os.getenv("CW_EMBEDDING_DIMENSION"):
            self.embedding.dimension = int(dimension)
            if self.providers.embedding:
                self.providers.embedding.dimension = int(dimension)

        # Provider-specific API keys
        if api_key := os.getenv("CW_VOYAGE_API_KEY"):
            # Set legacy embedding config for backward compatibility
            if self.embedding.provider == "voyage-ai":
                self.embedding.api_key = api_key
            # Update modern provider config
            if self.providers.voyage:
                self.providers.voyage.api_key = api_key
        if api_key := os.getenv("CW_OPENAI_API_KEY"):
            if self.embedding.provider == "openai":
                self.embedding.api_key = api_key
            # Update modern provider config
            if self.providers.openai:
                self.providers.openai.api_key = api_key
        if api_key := os.getenv("CW_COHERE_API_KEY"):
            if self.provider.embedding_provider == "cohere":
                self.provider.CW_EMBEDDING_API_KEY = api_key
            if self.embedding.provider == "cohere":
                self.embedding.api_key = api_key
            # Update modern provider config
            if self.providers.cohere:
                self.providers.cohere.api_key = api_key
        if api_key := os.getenv("CW_HUGGINGFACE_API_KEY"):
            if self.provider.embedding_provider == "huggingface":
                self.provider.CW_EMBEDDING_API_KEY = api_key
            if self.embedding.provider == "huggingface":
                self.embedding.api_key = api_key
            # Update modern provider config
            if self.providers.huggingface:
                self.providers.huggingface.api_key = api_key

        # Reranking configuration
        if rerank_provider := os.getenv("CW_RERANK_PROVIDER"):
            self.provider.rerank_provider = rerank_provider.lower()
            self.embedding.rerank_provider = rerank_provider.lower()
        if rerank_model := os.getenv("CW_RERANK_MODEL"):
            self.provider.rerank_model = rerank_model
            self.embedding.rerank_model = rerank_model

        # Local model settings
        if use_local := os.getenv("CW_USE_LOCAL_MODELS"):
            value = use_local.lower() in ("true", "1", "yes")
            self.provider.use_local = value
            self.embedding.use_local = value
        if device := os.getenv("CW_MODEL_DEVICE"):
            self.provider.device = device.lower()
            self.embedding.device = device.lower()

        # New backend configuration
        if backend_provider := os.getenv("CW_VECTOR_BACKEND_PROVIDER"):
            self.backend.provider = backend_provider.lower()
        if backend_url := os.getenv("CW_VECTOR_BACKEND_URL"):
            self.backend.url = backend_url
        if backend_key := os.getenv("CW_VECTOR_BACKEND_API_KEY"):
            self.backend.api_key = backend_key

        # Collection settings
        if collection := os.getenv("CW_VECTOR_BACKEND_COLLECTION"):
            self.backend.collection_name = collection

        # Feature flags
        if hybrid := os.getenv("CW_ENABLE_HYBRID_SEARCH"):
            value = hybrid.lower() in ("true", "1", "yes")
            self.backend.enable_hybrid_search = value
        if sparse := os.getenv("CW_ENABLE_SPARSE_VECTORS"):
            value = sparse.lower() in ("true", "1", "yes")
            self.backend.enable_sparse_vectors = value

        # Legacy embedding configuration (for backward compatibility)
        if model := os.getenv("CW_VOYAGE_MODEL"):
            self.embedding.model = model
            if self.provider.embedding_provider == "voyage-ai":
                self.provider.embedding_model = model
        if provider := os.getenv("CW_EMBEDDING_PRODIVER"):
            self.embedding.provider = provider.lower()
        if base_url := os.getenv("CW_OPENAI_BASE_URL"):
            self.embedding.base_url = base_url
            self.provider.base_url = base_url

        # Server configuration
        if log_level := os.getenv("CW_LOG_LEVEL"):
            self.server.log_level = log_level.upper()

        # Sync legacy and new configurations
        self._sync_legacy_and_new_configs()

    def get_effective_embedding_provider(self) -> str:
        """Get the effective embedding provider (prefer modern config)."""
        if self.providers.embedding:
            return getattr(self.providers.embedding, "provider_name", "unknown")
        return self.embedding.provider

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
            self.provider.CW_EMBEDDING_API_KEY or self.backend.url
        )

    def to_new_format_dict(self) -> dict[str, Any]:
        """Convert configuration to new format dictionary."""
        result = {
            "backend": {
                "provider": self.get_effective_backend_provider(),
                "url": self.get_effective_backend_url(),
                "api_key": self.get_effective_backend_api_key(),
                "collection_name": self.backend.collection_name,
                "enable_hybrid_search": self.backend.enable_hybrid_search,
                "enable_sparse_vectors": self.backend.enable_sparse_vectors,
            },
            "data_sources": {
                "enabled": self.data_sources.enabled,
                "sources": self.data_sources.sources,
            },
            "_config_version": "2.0",
        }

        # Add modern providers config if available
        if self.providers.embedding or self.providers.reranking:
            result["providers"] = self.providers.model_dump(exclude_unset=True)
        else:
            # Fallback to legacy provider format
            result["provider"] = {
                "embedding_provider": self.get_effective_embedding_provider(),
                "embedding_api_key": self.provider.CW_EMBEDDING_API_KEY
                or self.embedding.api_key,
                "embedding_model": self.provider.embedding_model or self.embedding.model,
                "embedding_dimension": self.provider.embedding_dimension
                or self.embedding.dimension,
                "rerank_provider": self.provider.rerank_provider or self.embedding.rerank_provider,
                "rerank_model": self.provider.rerank_model or self.embedding.rerank_model,
            }

        return result

    def _sync_legacy_and_new_configs(self) -> None:
        """Synchronize legacy and new configuration formats."""
        # Legacy sync no longer needed since we removed legacy provider config
        # Only sync backend config with legacy Qdrant config for compatibility
        if self.backend.url and not self.qdrant.url:
            self.qdrant.url = self.backend.url
        elif self.qdrant.url and not self.backend.url:
            self.backend.url = self.qdrant.url

        if self.backend.api_key and not self.qdrant.api_key:
            self.qdrant.api_key = self.backend.api_key
        elif self.qdrant.api_key and not self.backend.api_key:
            self.backend.api_key = self.qdrant.api_key

    def _initialize_modern_providers_from_legacy(self) -> None:
        """Initialize modern provider configs from legacy config."""
        if not _EXTENDED_CONFIGS_AVAILABLE:
            return

        provider_name = self.provider.embedding_provider.lower()
        try:
            if provider_name == "voyage-ai":
                self.providers.voyage = VoyageConfig(
                    api_key=self.provider.CW_EMBEDDING_API_KEY,
                    model=self.provider.embedding_model,
                    dimension=self.provider.embedding_dimension,
                    batch_size=self.provider.embedding_batch_size,
                    normalize_embeddings=self.provider.normalize_embeddings,
                )
                self.providers.embedding = self.providers.voyage
                if self.provider.rerank_provider == "voyage-ai":
                    self.providers.reranking = self.providers.voyage
            elif provider_name == "openai":
                self.providers.openai = OpenAIConfig(
                    api_key=self.provider.CW_EMBEDDING_API_KEY,
                    model=self.provider.embedding_model,
                    dimension=self.provider.embedding_dimension,
                    batch_size=self.provider.embedding_batch_size,
                    normalize_embeddings=self.provider.normalize_embeddings,
                )
                self.providers.embedding = self.providers.openai
            elif provider_name == "cohere":
                self.providers.cohere = CohereConfig(
                    api_key=self.provider.CW_EMBEDDING_API_KEY,
                    model=self.provider.embedding_model,
                    dimension=self.provider.embedding_dimension,
                    batch_size=self.provider.embedding_batch_size,
                    normalize_embeddings=self.provider.normalize_embeddings,
                )
                self.providers.embedding = self.providers.cohere
                if self.provider.rerank_provider == "cohere":
                    self.providers.reranking = self.providers.cohere
            # Add other providers as needed
        except Exception as e:
            logger.warning("Failed to initialize modern provider config: %s", e)

    def _create_modern_provider_config(
        self, provider_name: str, embedding_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Create modern provider configuration from legacy embedding config."""
        if not _EXTENDED_CONFIGS_AVAILABLE:
            return {}

        provider_configs = {}
        api_key = embedding_config.get("api_key")
        model = embedding_config.get("model")
        dimension = embedding_config.get("dimension")
        batch_size = embedding_config.get("batch_size", 8)
        normalize = embedding_config.get("normalize_embeddings", True)

        if provider_name == "voyage-ai":
            provider_configs["voyage-ai"] = {
                "api_key": api_key,
                "model": model or "voyage-code-3",
                "dimension": dimension,
                "batch_size": batch_size,
                "normalize_embeddings": normalize,
            }
            provider_configs["embedding"] = provider_configs["voyage-ai"]
        elif provider_name == "openai":
            provider_configs["openai"] = {
                "api_key": api_key,
                "model": model or "text-embedding-3-small",
                "dimension": dimension,
                "batch_size": batch_size,
                "normalize_embeddings": normalize,
            }
            provider_configs["embedding"] = provider_configs["openai"]
        # Add other providers as needed

        return provider_configs


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
        """Load configuration from all sources with proper precedence.

        Hierarchical loading order (lowest to highest precedence):
        1. Default values (built into Pydantic models)
        2. User config file (~/.config/code-weaver/config.toml)
        3. Repository config file (.code-weaver.toml)
        4. Local workspace config file (.local.code-weaver.toml)
        5. Explicit config file (if provided to constructor)
        6. Environment variables
        7. Runtime parameters (handled elsewhere)
        """
        if self._config is not None:
            return self._config

        # Start with default configuration (step 1: defaults)
        config = self.create_default_config()
        logger.debug("Created default configuration")

        # Load from config files (steps 2-5: file hierarchy)
        config_files = list(reversed(self.CONFIG_LOCATIONS))  # Reverse for correct precedence
        if self.config_path:
            config_files.append(self.config_path)

        for config_file in config_files:
            config_path = Path(config_file) if isinstance(config_file, str) else config_file

            if config_path.exists():
                try:
                    with config_path.open("r", encoding="utf-8") as f:
                        parsed_config = tomlkit.load(f)

                    # Remove tomlkit-specific objects and convert to plain dict
                    filtered_parsed_config = self._clean_toml_dict(parsed_config)
                    config.merge_from_dict(filtered_parsed_config)
                    logger.info("Loaded configuration from: %s", config_path)
                except Exception as e:
                    logger.warning("Failed to load config from %s: %s", config_path, e)

        # Environment variables have highest precedence (step 6: env vars)
        # config.merge_from_env()  # Temporarily disabled while removing legacy provider code
        logger.debug("Environment variable overrides temporarily disabled")

        # Final validation
        self._validate_config(config)
        logger.debug("Configuration validation completed")

        self._config = config
        return config

    def _clean_toml_dict(self, toml_dict: Any) -> dict[str, Any]:
        """Clean TOML dictionary by converting tomlkit objects to plain Python objects.

        Args:
            toml_dict: Dictionary potentially containing tomlkit objects

        Returns:
            Clean dictionary with plain Python objects
        """
        if isinstance(toml_dict, dict):
            return {key: self._clean_toml_dict(value) for key, value in toml_dict.items()}
        if isinstance(toml_dict, list):
            return [self._clean_toml_dict(item) for item in toml_dict]
        if hasattr(toml_dict, "unwrap"):
            # tomlkit objects have an unwrap method
            return self._clean_toml_dict(toml_dict.unwrap())
        return toml_dict

    def _validate_config(self, config: "CodeWeaverConfig") -> None:
        """Validate the configuration and raise errors for missing required values."""
        # Get effective provider
        provider = config.get_effective_embedding_provider().lower()

        # Validate embedding provider
        if provider in ["voyage-ai", "openai", "cohere", "huggingface"]:
            # These providers require API keys (unless using local models)
            api_key = config.embedding.api_key
            use_local = config.embedding.use_local

            if (provider == "huggingface" and use_local) or provider == "sentence-transformers":
                # Local HuggingFace models don't require API key
                pass
            elif not api_key:
                provider_env_map = {
                    "voyage-ai": "CW_VOYAGE_API_KEY",
                    "openai": "CW_OPENAI_API_KEY",
                    "cohere": "CW_COHERE_API_KEY",
                    "huggingface": "CW_HUGGINGFACE_API_KEY",
                }
                env_var = provider_env_map.get(provider, f"{provider.upper()}_API_KEY")
                raise ValueError(f"{env_var} is required when using {provider} embeddings")
        elif provider != "sentence-transformers":
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
            raise ValueError(f"You must provide a backend url for provider {backend_provider}")

        # Validate backend provider
        backend_provider = config.get_effective_backend_provider()
        supported_backends = [
            "qdrant",
           # "pinecone",
           # "chroma",
           # "weaviate",
           # "pgvector",
           # "milvus",
           # "elasticsearch",
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

    def reload_config(self) -> CodeWeaverConfig:
        """Reload configuration from all sources, clearing cache.

        Returns:
            Freshly loaded configuration
        """
        self._config = None
        return self.load_config()

    def get_config_path_hierarchy(self) -> list[Path]:
        """Get the full hierarchy of config paths in precedence order.

        Returns:
            List of config paths from lowest to highest precedence
        """
        paths = []

        # Add default locations
        for location in reversed(self.CONFIG_LOCATIONS):
            path = Path(location) if isinstance(location, str) else location
            paths.append(path)

        # Add explicit path if provided
        if self.config_path:
            paths.append(Path(self.config_path))

        return paths

    def find_existing_config_files(self) -> list[Path]:
        """Find all existing configuration files in the hierarchy.

        Returns:
            List of existing config file paths
        """
        existing = []
        existing.extend(path for path in self.get_config_path_hierarchy() if path.exists())
        return existing

    def save_config(self, config: CodeWeaverConfig, config_path: str | Path | None = None) -> Path:
        """Save configuration to TOML file with preserved formatting.

        Args:
            config: Configuration to save
            config_path: Optional path to save to. If None, uses the default user config location.

        Returns:
            Path where the configuration was saved
        """
        if config_path is None:
            # Use the user config location as default for saving
            save_path = Path.home() / ".config" / "code-weaver" / "config.toml"
        else:
            save_path = Path(config_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert config to dictionary
        parsed_config = config.model_dump(
            exclude_unset=True, exclude={"config_version", "migrated_from_legacy"}
        )

        # Create TOML document with comments
        doc = tomlkit.document()

        # Add header comment
        doc.add(tomlkit.comment("Code Weaver MCP Server Configuration"))
        doc.add(tomlkit.comment(f"Generated by CodeWeaver v{config.server.server_version}"))
        doc.add(tomlkit.comment(""))
        doc.add(tomlkit.nl())

        # Add sections with comments
        if "backend" in parsed_config:
            doc.add(tomlkit.comment("Vector Database Backend Configuration"))
            doc["backend"] = parsed_config["backend"]
            doc.add(tomlkit.nl())

        if "provider" in parsed_config:
            doc.add(tomlkit.comment("Embedding and Reranking Provider Configuration"))
            doc["provider"] = parsed_config["provider"]
            doc.add(tomlkit.nl())

        if "data_sources" in parsed_config:
            doc.add(tomlkit.comment("Data Sources Configuration"))
            doc["data_sources"] = parsed_config["data_sources"]
            doc.add(tomlkit.nl())

        # Add other sections
        for key, value in parsed_config.items():
            if key not in ["backend", "provider", "data_sources"]:
                doc.add(tomlkit.comment(f"{key.replace('_', ' ').title()} Configuration"))
                doc[key] = value
                doc.add(tomlkit.nl())

        # Add metadata
        doc.add(tomlkit.comment("Configuration Metadata"))
        metadata_table = tomlkit.table()
        metadata_table["_config_version"] = "2.0"
        if config.migrated_from_legacy:
            metadata_table["_migrated_from_legacy"] = True
        doc["_metadata"] = metadata_table

        # Write to file
        with save_path.open("w", encoding="utf-8") as f:
            tomlkit.dump(doc, f)

        logger.info("Saved configuration to: %s", save_path)
        return save_path

    def load_config_from_dict(self, parsed_config: dict[str, Any]) -> CodeWeaverConfig:
        """Load configuration from a dictionary with validation.

        Args:
            parsed_config: Configuration dictionary

        Returns:
            Validated configuration object
        """
        config = CodeWeaverConfig.model_validate(parsed_config)

        # Apply environment variable overrides
        # config.merge_from_env()  # Temporarily disabled while removing legacy provider code

        # Validate the final configuration
        self._validate_config(config)

        return config

    def create_default_config(self) -> CodeWeaverConfig:
        """Create a default configuration with sensible defaults.

        Returns:
            Default configuration object
        """
        config = CodeWeaverConfig()

        # Set up a basic filesystem data source
        config.data_sources.sources = [
            {
                "type": "filesystem",
                "enabled": True,
                "priority": 1,
                "source_id": "default_filesystem",
                "config": {
                    "root_path": ".",
                    "use_gitignore": True,
                    "additional_ignore_patterns": config.indexing.additional_ignore_patterns,
                    "max_file_size_mb": config.chunking.max_file_size_mb,
                },
            }
        ]

        return config

    def get_example_config(
        self, config_format: Literal["new", "legacy", "migration"] = "new"
    ) -> str:
        """Get an example TOML configuration file."""
        if config_format == "new":
            return self._get_new_format_example()
        if config_format == "legacy":
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
url = "YOUR_BACKEND_URL"  # Can also use CW_VECTOR_BACKEND_URL env var
api_key = "YOUR_BACKEND_API_KEY"  # Can also use CW_VECTOR_BACKEND_API_KEY env var
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

# Modern Provider Configuration (Recommended)
[providers]
enable_caching = true
cache_ttl_seconds = 3600

# Voyage AI Configuration (Combined embedding + reranking)
[providers.voyage]
api_key = "YOUR_CW_VOYAGE_API_KEY"  # Can also use CW_VOYAGE_API_KEY env var
model = "voyage-code-3"
embedding_model = "voyage-code-3"
reranking_model = "voyage-rerank-2"
normalize_embeddings = true
batch_size = 8
timeout_seconds = 30.0
enable_embeddings = true
enable_reranking = true

# Set active providers
[providers.embedding]
api_key = "YOUR_CW_VOYAGE_API_KEY"
model = "voyage-code-3"
normalize_embeddings = true
batch_size = 8

[providers.reranking]
api_key = "YOUR_CW_VOYAGE_API_KEY"
model = "voyage-rerank-2"
top_k = 50
return_scores = true

# OpenAI Configuration (embedding only)
# [providers.openai]
# api_key = "YOUR_CW_OPENAI_API_KEY"
# model = "text-embedding-3-small"
# max_input_length = 8191
# normalize_embeddings = true
# batch_size = 8

# Cohere Configuration (combined)
# [providers.cohere]
# api_key = "YOUR_COHERE_API_KEY"
# model = "embed-english-v3.0"
# embedding_model = "embed-english-v3.0"
# reranking_model = "rerank-english-v3.0"
# input_type = "search_document"
# enable_embeddings = true
# enable_reranking = true

# Local SentenceTransformers (no API key required)
# [providers.sentence_transformers]
# model = "all-MiniLM-L6-v2"
# device = "cpu"  # cpu, cuda, mps
# normalize_embeddings = true
# cache_folder = "/path/to/model/cache"

# Legacy Provider Configuration (for backward compatibility)
[provider]
embedding_provider = "voyage-ai"  # voyage, openai, cohere, sentence-transformers, huggingface
CW_EMBEDDING_API_KEY = "YOUR_CW_EMBEDDING_API_KEY"  # Can also use CW_VOYAGE_API_KEY, CW_OPENAI_API_KEY, etc.
embedding_model = "voyage-code-3"
embedding_dimension = 1024
embedding_batch_size = 8

# Reranking configuration
rerank_provider = "voyage-ai"  # If None, uses embedding provider if available
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
provider = "voyage-ai"  # Kept for backward compatibility
api_key = "YOUR_CW_VOYAGE_API_KEY"  # Synced with provider.CW_EMBEDDING_API_KEY
model = "voyage-code-3"  # Synced with provider.embedding_model

[qdrant]
url = "YOUR_BACKEND_URL"  # Synced with backend.url
api_key = "YOUR_BACKEND_API_KEY"  # Synced with backend.api_key
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
provider = "voyage-ai"  # "voyage-ai" or "openai"
api_key = "YOUR_CW_VOYAGE_API_KEY"  # Can also use CW_VOYAGE_API_KEY env var
model = "voyage-code-3"
dimension = 1024
batch_size = 8

[qdrant]
url = "YOUR_BACKEND_URL"  # Can also use CW_VECTOR_BACKEND_URL env var
api_key = "YOUR_BACKEND_API_KEY"  # Can also use CW_VECTOR_BACKEND_API_KEY env var
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
# provider = "voyage-ai"
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

# Modern Provider Configuration
[providers.voyage]
api_key = "your-voyage-key"
model = "voyage-code-3"
embedding_model = "voyage-code-3"
reranking_model = "voyage-rerank-2"
enable_embeddings = true
enable_reranking = true

[providers.embedding]
api_key = "your-voyage-key"
model = "voyage-code-3"

# Legacy Provider Configuration (for backward compatibility)
[provider]
embedding_provider = "voyage-ai"
CW_EMBEDDING_API_KEY = "your-voyage-key"
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

    def validate_config_file(self, config_path: str | Path) -> dict[str, Any]:
        """Validate a specific configuration file.

        Args:
            config_path: Path to the configuration file to validate

        Returns:
            Validation results dictionary with errors, warnings, and summary
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
            # Try to load and parse the TOML file
            with path.open("r", encoding="utf-8") as f:
                parsed_config = tomlkit.load(f)

            # Clean the TOML dict
            filtered_parsed_config = self._clean_toml_dict(parsed_config)

            # Try to create a config object from it
            temp_config = CodeWeaverConfig()
            temp_config.merge_from_dict(filtered_parsed_config)

            # Run validation
            self._validate_config(temp_config)

            result["valid"] = True
            result["warnings"].append("Configuration file loaded and validated successfully")

        except Exception as e:
            result["errors"].append(f"Configuration validation failed: {e}")

        return result

    def export_config_template(
        self, template_type: Literal["minimal", "full", "enterprise"] = "full"
    ) -> str:
        """Export a configuration template with different levels of detail.

        Args:
            template_type: Type of template to generate

        Returns:
            TOML configuration template as string
        """
        if template_type == "minimal":
            return self._get_minimal_template()
        if template_type == "enterprise":
            return self._get_enterprise_template()
        return self._get_new_format_example()

    def _get_minimal_template(self) -> str:
        """Get minimal configuration template."""
        return """# Code Weaver MCP Server - Minimal Configuration

# Vector Database Backend
[backend]
provider = "qdrant"
url = "YOUR_BACKEND_URL"
api_key = "YOUR_BACKEND_API_KEY"

# Modern Provider Configuration
[providers.voyage]
api_key = "YOUR_CW_VOYAGE_API_KEY"
model = "voyage-code-3"

[providers.embedding]
api_key = "YOUR_CW_VOYAGE_API_KEY"
model = "voyage-code-3"

# Legacy Provider Configuration
[provider]
embedding_provider = "voyage-ai"
CW_EMBEDDING_API_KEY = "YOUR_CW_VOYAGE_API_KEY"

# Basic Data Source
[[data_sources.sources]]
type = "filesystem"
enabled = true
priority = 1

[data_sources.sources.config]
root_path = "."
"""

    def _get_enterprise_template(self) -> str:
        """Get enterprise configuration template with all options."""
        return """# Code Weaver MCP Server - Enterprise Configuration
# Full configuration with all available options and explanations

# Vector Database Backend Configuration
[backend]
provider = "qdrant"  # Options: qdrant, pinecone, chroma, weaviate, pgvector, milvus, elasticsearch
url = "YOUR_BACKEND_URL"
api_key = "YOUR_BACKEND_API_KEY"
collection_name = "code-embeddings"

# Performance and scaling
batch_size = 100
connection_timeout = 30.0
request_timeout = 60.0
max_connections = 10

# Advanced features
enable_hybrid_search = true
enable_sparse_vectors = true
enable_streaming = false
enable_failover = true
replica_urls = ["backup1.example.com", "backup2.example.com"]

# Modern Provider Configuration (Recommended for Enterprise)
[providers]
enable_caching = true
cache_ttl_seconds = 7200  # 2 hours

# Voyage AI Configuration (Primary)
[providers.voyage]
api_key = "YOUR_CW_VOYAGE_API_KEY"
model = "voyage-code-3"
embedding_model = "voyage-code-3"
reranking_model = "voyage-rerank-2"
batch_size = 16  # Higher for enterprise
timeout_seconds = 60.0
max_retries = 5
requests_per_minute = 200
enable_embeddings = true
enable_reranking = true

# Set as active providers
[providers.embedding]
api_key = "YOUR_CW_VOYAGE_API_KEY"
model = "voyage-code-3"
batch_size = 16
timeout_seconds = 60.0

[providers.reranking]
api_key = "YOUR_CW_VOYAGE_API_KEY"
model = "voyage-rerank-2"
top_k = 100
relevance_threshold = 0.1
return_scores = true

# Backup OpenAI Configuration
[providers.openai]
api_key = "YOUR_CW_OPENAI_API_KEY"
model = "text-embedding-3-large"
max_input_length = 8191
batch_size = 16
timeout_seconds = 60.0

# Legacy Provider Configuration (for backward compatibility)
[provider]
embedding_provider = "voyage-ai"  # Options: voyage, openai, cohere, sentence-transformers, huggingface
CW_EMBEDDING_API_KEY = "YOUR_CW_VOYAGE_API_KEY"
embedding_model = "voyage-code-3"
embedding_dimension = 1024
embedding_batch_size = 16

# Reranking (optional but recommended)
rerank_provider = "voyage-ai"
rerank_model = "voyage-rerank-2"

# Caching and optimization
enable_caching = true
cache_ttl_seconds = 7200

# Data Sources Configuration
[data_sources]
enabled = true
max_concurrent_sources = 10
enable_content_deduplication = true
enable_metadata_extraction = true

# Primary filesystem source
[[data_sources.sources]]
type = "filesystem"
enabled = true
priority = 1
source_id = "primary_codebase"

[data_sources.sources.config]
root_path = "."
use_gitignore = true
max_file_size_mb = 5

# Advanced Configuration
[chunking]
max_chunk_size = 2000
min_chunk_size = 100
max_file_size_mb = 5

[rate_limiting]
voyage_requests_per_minute = 200
voyage_tokens_per_minute = 2000000
max_retries = 10

[server]
log_level = "INFO"
max_search_results = 100
enable_request_logging = true
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
        "CW_VOYAGE_API_KEY": "Voyage AI",
        "CW_OPENAI_API_KEY": "OpenAI",
        "CW_COHERE_API_KEY": "Cohere",
        "CW_HUGGINGFACE_API_KEY": "HuggingFace",
    }

    for var, provider in provider_vars.items():
        if os.getenv(var):
            validation_results["detected_vars"][var] = f"{provider} API key detected"

    # Check backend configuration
    backend_vars = {
        "CW_VECTOR_BACKEND_URL": "Backend URL",
        "CW_VECTOR_BACKEND_API_KEY": "Backend API key",
        "CW_VECTOR_BACKEND_PROVIDER": "Backend provider",
        "CW_VECTOR_BACKEND_COLLECTION": "Backend collection name",
    }

    for var, description in backend_vars.items():
        if os.getenv(var):
            validation_results["detected_vars"][var] = description

    # Check for required variables
    if not os.getenv("CW_VECTOR_BACKEND_URL"):
        validation_results["errors"].append("No backend URL found. Set CW_VECTOR_BACKEND_URL")
        validation_results["valid"] = False

    # Check for embedding provider API key
    embedding_provider = os.getenv("CW_EMBEDDING_PROVIDER", "CW_VOYAGE_AI").lower()
    use_local = os.getenv("CW_USE_LOCAL_MODELS", "false").lower() in ("true", "1", "yes")

    if not use_local and embedding_provider in ["voyage-ai", "openai", "cohere", "huggingface"]:
        provider_key_map = {
            "voyage-ai": "CW_VOYAGE_API_KEY",
            "openai": "CW_OPENAI_API_KEY",
            "cohere": "CW_COHERE_API_KEY",
            "huggingface": "CW_HUGGINGFACE_API_KEY",
        }

        required_key = provider_key_map.get(embedding_provider)
        if required_key and not os.getenv(required_key):
            validation_results["errors"].append(
                f"No API key found for {embedding_provider} provider. Set {required_key}"
            )
            validation_results["valid"] = False

    # All configuration is now using the new format
    # Legacy environment variable support has been removed

    return validation_results


def get_effective_config_summary() -> dict[str, Any]:
    """Get a summary of the effective configuration."""
    try:
        config = get_config()

        return {
            "config_version": config.config_version,
            "migrated_from_legacy": config.migrated_from_legacy,
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
                "has_api_key": bool(
                    config.provider.CW_EMBEDDING_API_KEY or config.embedding.api_key
                ),
                "use_local": config.provider.use_local or config.embedding.use_local,
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


# Utility functions for configuration management
def is_configuration_migrated() -> bool:
    """Check if configuration has been migrated from legacy format."""
    try:
        config = get_config()
    except Exception:
        return False
    return config.migrated_from_legacy


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
        if provider in ["voyage-ai", "cohere"] and not rerank_provider:
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
