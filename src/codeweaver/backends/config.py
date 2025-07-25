# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Backend configuration extensions for CodeWeaver.

Extends the existing configuration system to support multiple vector
database backends while maintaining backward compatibility.
"""

import os

from typing import Annotated, Any
from uuid import uuid4

from pydantic import ConfigDict, Field, field_validator

from codeweaver._types.provider_enums import BackendProvider, HybridFusionStrategy, SparseIndexType
from codeweaver._types.provider_enums import ProviderKind
from codeweaver.backends.factory import BackendConfig


class BackendConfigExtended(BackendConfig):
    """
    Extended backend configuration with comprehensive options.

    Extends the base BackendConfig with additional settings for
    enterprise deployments and advanced use cases.
    """

    model_config = ConfigDict(
        extra="allow",
        str_strip_whitespace=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    # Override required fields with defaults
    provider: Annotated[str, Field(default="qdrant", description="Backend provider type")]
    kind: Annotated[
        ProviderKind,
        Field(default=ProviderKind.COMBINED, description="Provider kind classification"),
    ]

    # Collection management
    collection_name: Annotated[
        str,
        Field(
            default_factory=lambda: f"codeweaver-{uuid4()}",
            description="Name of the vector collection",
            min_length=1,
            max_length=255,
        ),
    ]
    auto_create_collection: Annotated[
        bool, Field(default=True, description="Automatically create collection if it doesn't exist")
    ]
    collection_dimension: Annotated[
        int,
        Field(
            default=1024, ge=1, le=4096, description="Vector dimension for the collection (1-4096)"
        ),
    ]

    # Advanced hybrid search settings
    sparse_index_fields: Annotated[
        list[str],
        Field(
            default_factory=lambda: ["content", "chunk_type"],
            description="Fields to index for sparse vector search",
        ),
    ]
    sparse_index_type: Annotated[
        SparseIndexType,
        Field(default=SparseIndexType.BM25, description="Type of sparse index to use"),
    ]
    hybrid_fusion_strategy: Annotated[
        HybridFusionStrategy,
        Field(
            default=HybridFusionStrategy.RRF,
            description="Strategy for fusing dense and sparse results",
        ),
    ]
    hybrid_alpha: Annotated[
        float,
        Field(
            default=0.5,
            ge=0.0,
            le=1.0,
            description="Balance between dense (1.0) and sparse (0.0) search",
        ),
    ]

    # Performance and scaling
    batch_size: Annotated[
        int, Field(default=100, ge=1, le=1000, description="Batch size for operations (1-1000)")
    ]
    max_batch_size: Annotated[
        int, Field(default=1000, ge=1, le=10000, description="Maximum batch size limit (1-10000)")
    ]
    connection_pool_size: Annotated[
        int, Field(default=10, ge=1, le=100, description="Connection pool size (1-100)")
    ]
    enable_connection_pooling: Annotated[
        bool, Field(default=True, description="Enable connection pooling")
    ]
    enable_request_compression: Annotated[
        bool, Field(default=True, description="Enable request compression")
    ]

    # Caching and optimization
    enable_result_caching: Annotated[
        bool, Field(default=False, description="Enable result caching")
    ]
    cache_ttl_seconds: Annotated[
        int, Field(default=300, ge=1, le=86400, description="Cache TTL in seconds (1-86400)")
    ]
    enable_query_optimization: Annotated[
        bool, Field(default=True, description="Enable query optimization")
    ]

    # High availability and failover
    replica_urls: Annotated[
        list[str], Field(default_factory=list, description="List of replica URLs for failover")
    ]
    enable_failover: Annotated[bool, Field(default=False, description="Enable automatic failover")]
    health_check_interval: Annotated[
        int,
        Field(default=30, ge=1, le=3600, description="Health check interval in seconds (1-3600)"),
    ]

    # Security and compliance
    enable_tls: Annotated[bool, Field(default=True, description="Enable TLS encryption")]
    verify_ssl: Annotated[bool, Field(default=True, description="Verify SSL certificates")]
    require_encryption: Annotated[
        bool, Field(default=True, description="Require encryption for all connections")
    ]
    enable_grpc: Annotated[bool, Field(default=False, description="Enable gRPC protocol")]
    prefer_http: Annotated[bool, Field(default=False, description="Prefer HTTP over gRPC")]
    prefer_grpc: Annotated[bool, Field(default=True, description="Prefer gRPC over HTTP")]
    client_cert_path: Annotated[
        str | None, Field(default=None, description="Path to client certificate file")
    ]
    client_key_path: Annotated[
        str | None, Field(default=None, description="Path to client private key file")
    ]

    # Monitoring and observability
    enable_metrics: Annotated[bool, Field(default=False, description="Enable metrics collection")]
    metrics_endpoint: Annotated[str | None, Field(default=None, description="Metrics endpoint URL")]
    enable_tracing: Annotated[bool, Field(default=False, description="Enable distributed tracing")]
    trace_sample_rate: Annotated[
        float, Field(default=0.1, ge=0.0, le=1.0, description="Trace sample rate (0.0-1.0)")
    ]

    @field_validator("replica_urls", mode="before")
    @classmethod
    def validate_replica_urls(cls, replica_urls: list[str]) -> list[str]:
        """Validate replica URLs."""
        if not replica_urls:
            return replica_urls
        validated_urls = []
        for url in replica_urls:
            url = url.strip()
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"Replica URL must start with http:// or https://, got: {url}")
            validated_urls.append(url)
        return validated_urls

    @field_validator("client_cert_path", "client_key_path", mode="before")
    @classmethod
    def validate_file_paths(cls, file_path: str | None) -> str | None:
        """Validate file paths exist if provided."""
        if file_path is not None:
            file_path = file_path.strip()
            if len(file_path) == 0:
                return None
            # Note: We don't check file existence here as it may not be available at config time
        return file_path

    def model_post_init(self, __context: Any, /) -> None:
        """Post-initialization to ensure configuration consistency."""
        # Ensure gRPC preferences are consistent
        if not self.enable_grpc:
            self.prefer_grpc = False
        if self.url:
            self.prefer_http = True


def create_backend_config_from_env() -> BackendConfigExtended:
    """
    Create backend configuration from environment variables.

    Uses new backend-agnostic environment variables:
    - CW_VECTOR_BACKEND_PROVIDER, CW_VECTOR_BACKEND_URL, CW_VECTOR_BACKEND_API_KEY
    - CW_VECTOR_BACKEND_COLLECTION

    Returns:
        Backend configuration from environment
    """
    # Check for new environment variables
    provider_env_var = os.getenv("CW_VECTOR_BACKEND_PROVIDER", "qdrant").lower()

    # Convert to enum if possible, otherwise use string
    try:
        provider = BackendProvider.from_string(provider_env_var).value
    except ValueError:
        provider = provider_env_var  # Use string if not in enum

    url = os.getenv("CW_VECTOR_BACKEND_URL")
    api_key = os.getenv("CW_VECTOR_BACKEND_API_KEY")

    # Collection settings
    collection_name = os.getenv("CW_VECTOR_BACKEND_COLLECTION", "code-embeddings")

    # Feature flags
    enable_hybrid = os.getenv("CW_ENABLE_HYBRID_SEARCH", "false").lower() == "true"
    enable_sparse = os.getenv("ENABLE_SPARSE_VECTORS", "false").lower() == "true"

    # Performance settings
    batch_size = int(os.getenv("BACKEND_BATCH_SIZE", "100"))
    connection_timeout = float(os.getenv("BACKEND_CONNECTION_TIMEOUT", "30.0"))
    request_timeout = float(os.getenv("BACKEND_REQUEST_TIMEOUT", "60.0"))

    return BackendConfigExtended(
        provider=provider,
        kind="combined",  # Default kind for env configs
        url=url,
        api_key=api_key,
        collection_name=collection_name,
        enable_hybrid_search=enable_hybrid,
        enable_sparse_vectors=enable_sparse,
        batch_size=batch_size,
        connection_timeout=connection_timeout,
        request_timeout=request_timeout,
    )


def get_provider_specific_config(
    provider: str, base_config: BackendConfigExtended
) -> dict[str, Any]:
    """
    Get provider-specific configuration options.

    Args:
        provider: Backend provider name
        base_config: Base configuration

    Returns:
        Provider-specific options dictionary
    """
    if provider == "qdrant":
        return {
            "enable_sparse_vectors": base_config.enable_sparse_vectors,
            "sparse_on_disk": base_config.prefer_disk,
            "timeout": base_config.request_timeout,
            "prefer_grpc": base_config.enable_request_compression,
        }

    if provider == "pinecone":
        return {
            "environment": base_config.provider_options.get("environment", "us-west1-gcp"),
            "api_version": base_config.provider_options.get("api_version", "2024-07"),
            "pool_threads": base_config.connection_pool_size,
        }

    if provider == "chroma":
        return {
            "host": base_config.url,
            "port": base_config.provider_options.get("port", 8000),
            "ssl": base_config.enable_tls,
            "headers": base_config.provider_options.get("headers", {}),
        }

    if provider == "weaviate":
        return {
            "url": base_config.url,
            "auth_config": {"api_key": base_config.api_key} if base_config.api_key else None,
            "timeout_config": (base_config.connection_timeout, base_config.request_timeout),
            "connection_config": {"session_pool_connections": base_config.connection_pool_size},
        }

    if provider == "pgvector":
        return {
            "dsn": base_config.url,
            "pool_size": base_config.connection_pool_size,
            "pool_timeout": base_config.connection_timeout,
            "command_timeout": base_config.request_timeout,
        }

    # Default configuration for unknown providers
    return {
        "url": base_config.url,
        "api_key": base_config.api_key,
        "timeout": base_config.request_timeout,
    }


# Example configurations for different backends
EXAMPLE_CONFIGS = {
    "qdrant": {
        "provider": BackendProvider.QDRANT.value,
        "kind": "combined",
        "url": "https://your-cluster.qdrant.io",
        "api_key": "your-qdrant-api-key",
        "enable_hybrid_search": True,
        "enable_sparse_vectors": True,
    },
    "pinecone": {
        "provider": BackendProvider.PINECONE.value,
        "kind": "combined",
        "api_key": "your-pinecone-api-key",
        "provider_options": {"environment": "us-west1-gcp", "index_name": "code-embeddings"},
    },
    "chroma": {
        "provider": BackendProvider.CHROMA.value,
        "kind": "combined",
        "url": "http://localhost:8000",
        "enable_tls": False,
        "provider_options": {"port": 8000, "headers": {}},
    },
    "weaviate": {
        "provider": BackendProvider.WEAVIATE.value,
        "kind": "combined",
        "url": "https://your-cluster.weaviate.network",
        "api_key": "your-weaviate-api-key",
        "enable_hybrid_search": True,
    },
    "pgvector": {
        "provider": BackendProvider.PGVECTOR.value,
        "kind": "combined",
        "url": "postgresql://user:password@localhost:5432/database",
        "provider_options": {"table_name": "code_embeddings", "dimension": 1024},
    },
}
