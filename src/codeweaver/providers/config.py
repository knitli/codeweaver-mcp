# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Provider configuration models using Pydantic v2.

This module defines configuration models for all provider types with comprehensive
validation, type safety, and extensibility support.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from codeweaver.cw_types import DimensionSize


class ProviderConfig(BaseModel):
    """Base configuration for all providers.

    This provides common settings that all providers inherit, including
    API authentication, rate limiting, and performance tuning.
    """

    model_config = ConfigDict(
        extra="allow",  # Allow provider-specific extensions
        validate_assignment=True,  # Validate on attribute assignment
        frozen=False,  # Allow mutation for runtime updates
    )

    # Core settings
    enabled: Annotated[
        bool, Field(default=True, description="Whether this provider is enabled for use")
    ]

    api_key: Annotated[
        str | None,
        Field(
            default=None,
            description="API key for provider authentication. Can be None for local providers.",
        ),
    ]

    # Performance settings
    batch_size: Annotated[
        int,
        Field(default=8, ge=1, le=100, description="Number of items to process in a single batch"),
    ]

    timeout_seconds: Annotated[
        float, Field(default=30.0, gt=0, le=300, description="Timeout in seconds for API requests")
    ]

    max_retries: Annotated[
        int,
        Field(
            default=3,
            ge=0,
            le=10,
            description="Maximum number of retry attempts for failed requests",
        ),
    ]

    retry_delay_seconds: Annotated[
        float,
        Field(
            default=1.0,
            ge=0.1,
            le=60,
            description="Initial delay between retry attempts (with exponential backoff)",
        ),
    ]

    # Rate limiting
    rate_limiter: Annotated[
        Any | None,
        Field(default=None, description="Rate limiter instance for API request throttling"),
    ]

    requests_per_minute: Annotated[
        int | None,
        Field(
            default=None,
            gt=0,
            le=10000,
            description="Maximum requests per minute (for built-in rate limiting)",
        ),
    ]

    # Provider metadata
    provider_name: Annotated[
        str | None,
        Field(default=None, description="Human-readable provider name for logging and debugging"),
    ]

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str | None) -> str | None:
        """Validate API key format if provided."""
        if v is not None and len(v.strip()) == 0:
            raise ValueError("API key cannot be empty if provided")
        return v.strip() if v else None


class EmbeddingProviderConfig(ProviderConfig):
    """Configuration for embedding providers.

    Extends base config with embedding-specific settings like model selection
    and dimension configuration.
    """

    model: Annotated[
        str,
        Field(
            description="Model identifier for embeddings (e.g., 'voyage-code-3', 'text-embedding-3-small')"
        ),
    ]

    dimension: Annotated[
        DimensionSize | None,
        Field(default=None, description="Embedding dimension size. None uses model default."),
    ]

    # Embedding-specific settings
    normalize_embeddings: Annotated[
        bool, Field(default=True, description="Whether to normalize embeddings to unit length")
    ]

    truncate_input: Annotated[
        bool, Field(default=True, description="Whether to truncate inputs that exceed model limits")
    ]

    max_input_length: Annotated[
        int | None,
        Field(
            default=None,
            gt=0,
            description="Maximum input length in tokens. None uses model default.",
        ),
    ]

    # Model type hints for better IDE support
    model_type: Annotated[
        Literal["code", "text", "multimodal"] | None,
        Field(default=None, description="Type of content this model is optimized for"),
    ]

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate model identifier format."""
        if not v or not v.strip():
            raise ValueError("Model identifier cannot be empty")
        return v.strip()

    @field_validator("dimension")
    @classmethod
    def validate_dimension(cls, v: DimensionSize | None) -> DimensionSize | None:
        """Validate embedding dimensions are reasonable."""
        if v is not None:
            if v < 64:
                raise ValueError("Embedding dimension must be at least 64")
            if v > 4096:
                raise ValueError("Embedding dimension cannot exceed 4096")
            if v % 64 != 0:
                # Many models require dimensions divisible by 64
                raise ValueError(
                    "Embedding dimension should be divisible by 64 for optimal performance"
                )
        return v


class RerankingProviderConfig(ProviderConfig):
    """Configuration for reranking providers.

    Extends base config with reranking-specific settings like result limits
    and relevance thresholds.
    """

    model: Annotated[
        str,
        Field(
            description="Model identifier for reranking (e.g., 'voyage-rerank-2', 'rerank-english-v3.0')"
        ),
    ]

    top_k: Annotated[
        int | None,
        Field(
            default=None,
            gt=0,
            le=1000,
            description="Maximum number of results to return. None returns all.",
        ),
    ]

    # Reranking-specific settings
    relevance_threshold: Annotated[
        float | None,
        Field(
            default=None,
            ge=0.0,
            le=1.0,
            description="Minimum relevance score to include results. None includes all.",
        ),
    ]

    return_scores: Annotated[
        bool, Field(default=True, description="Whether to return relevance scores with results")
    ]

    query_max_length: Annotated[
        int | None,
        Field(
            default=None,
            gt=0,
            description="Maximum query length in tokens. None uses model default.",
        ),
    ]

    document_max_length: Annotated[
        int | None,
        Field(
            default=None,
            gt=0,
            description="Maximum document length in tokens. None uses model default.",
        ),
    ]

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate model identifier format."""
        if not v or not v.strip():
            raise ValueError("Model identifier cannot be empty")
        return v.strip()


class CombinedProviderConfig(EmbeddingProviderConfig, RerankingProviderConfig):
    """Configuration for providers that support both embedding and reranking.

    This allows providers like Cohere to use different models for each task
    while sharing common configuration.
    """

    # Override model to be optional since we have separate fields
    model: Annotated[
        str,
        Field(description="Default model identifier (can be overridden by task-specific models)"),
    ]

    # Task-specific model overrides
    embedding_model: Annotated[
        str | None,
        Field(
            default=None, description="Model specifically for embeddings. None uses 'model' field."
        ),
    ]

    reranking_model: Annotated[
        str | None,
        Field(
            default=None, description="Model specifically for reranking. None uses 'model' field."
        ),
    ]

    # Control which capabilities are enabled
    enable_embeddings: Annotated[
        bool, Field(default=True, description="Whether to use this provider for embeddings")
    ]

    enable_reranking: Annotated[
        bool, Field(default=True, description="Whether to use this provider for reranking")
    ]

    @field_validator("embedding_model", "reranking_model")
    @classmethod
    def validate_task_models(cls, v: str | None) -> str | None:
        """Validate task-specific model identifiers."""
        if v is not None and len(v.strip()) == 0:
            raise ValueError("Model identifier cannot be empty if provided")
        return v.strip() if v else None

    def get_embedding_model(self) -> str:
        """Get the model to use for embeddings."""
        return self.embedding_model or self.model

    def get_reranking_model(self) -> str:
        """Get the model to use for reranking."""
        return self.reranking_model or self.model


# Convenience type aliases for better code readability
AnyProviderConfig = (
    ProviderConfig | EmbeddingProviderConfig | RerankingProviderConfig | CombinedProviderConfig
)


# Provider-specific configuration examples that can be extended
class VoyageConfig(CombinedProviderConfig):
    """Voyage AI specific configuration with sensible defaults."""

    model: str = "voyage-code-3"
    embedding_model: str | None = "voyage-code-3"
    reranking_model: str | None = "voyage-rerank-2"
    model_type: Literal["code"] = "code"
    provider_name: str = "Voyage AI"


class OpenAIConfig(EmbeddingProviderConfig):
    """OpenAI specific configuration with sensible defaults."""

    model: str = "text-embedding-3-small"
    model_type: Literal["text"] = "text"
    provider_name: str = "OpenAI"
    max_input_length: int = 8191  # OpenAI's token limit


class OpenAICompatibleConfig(EmbeddingProviderConfig):
    """OpenAI-compatible API configuration with customizable endpoints."""

    model: str = "text-embedding-3-small"
    model_type: Literal["text"] = "text"
    provider_name: str = "OpenAI Compatible"
    max_input_length: int = 8191  # Default OpenAI token limit

    # Custom endpoint settings
    base_url: Annotated[
        str | None,
        Field(default=None, description="Custom API base URL for OpenAI-compatible endpoints"),
    ]

    api_version: Annotated[
        str | None,
        Field(default=None, description="API version for Azure OpenAI and similar services"),
    ]


class CohereConfig(CombinedProviderConfig):
    """Cohere specific configuration with sensible defaults."""

    model: str = "embed-english-v3.0"
    embedding_model: str | None = "embed-english-v3.0"
    reranking_model: str | None = "rerank-english-v3.0"
    model_type: Literal["text"] = "text"
    provider_name: str = "Cohere"

    # Cohere-specific settings
    input_type: Annotated[
        Literal["search_document", "search_query", "classification", "clustering"] | None,
        Field(default=None, description="Cohere input type for embeddings"),
    ]


class HuggingFaceConfig(EmbeddingProviderConfig):
    """HuggingFace specific configuration with sensible defaults."""

    model: str = "sentence-transformers/all-mpnet-base-v2"
    model_type: Literal["text"] = "text"
    provider_name: str = "HuggingFace"


class SentenceTransformersConfig(EmbeddingProviderConfig):
    """SentenceTransformers specific configuration with sensible defaults."""

    model: str = "all-MiniLM-L6-v2"
    model_type: Literal["text"] = "text"
    provider_name: str = "SentenceTransformers"

    # Local model specific settings
    device: Annotated[
        Literal["cpu", "cuda", "mps"] | None,
        Field(default="cpu", description="Device to run the model on"),
    ]

    cache_folder: Annotated[
        str | None, Field(default=None, description="Directory to cache downloaded models")
    ]


class SpaCyProviderConfig(ProviderConfig):
    """spaCy NLP provider configuration with intent classification and entity recognition."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    # Provider identification
    provider_name: str = "spaCy"

    # spaCy-specific configuration
    model: Annotated[str, Field(default="en_core_web_sm", description="spaCy model name")]
    use_transformers: Annotated[
        bool, Field(default=False, description="Use transformer-based models")
    ]
    enable_intent_classification: Annotated[
        bool, Field(default=True, description="Enable intent classification")
    ]
    intent_labels: Annotated[
        list[str],
        Field(
            default_factory=lambda: ["SEARCH", "DOCUMENTATION", "ANALYSIS"],
            description="Intent classification labels",
        ),
    ]
    confidence_threshold: Annotated[
        float, Field(default=0.7, ge=0.0, le=1.0, description="Intent confidence threshold")
    ]

    # Performance settings
    max_length: Annotated[
        int, Field(default=1_000_000, ge=1000, description="Maximum text length to process")
    ]

    # Domain patterns - simplified for config
    enable_domain_patterns: Annotated[
        bool, Field(default=True, description="Enable domain-specific entity patterns")
    ]
    custom_patterns_file: Annotated[
        str | None, Field(default=None, description="Path to custom patterns TOML file")
    ]
