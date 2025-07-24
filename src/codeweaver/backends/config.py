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

from dataclasses import dataclass, field
from typing import Any, Literal

from codeweaver.backends.factory import BackendConfig


@dataclass
class BackendConfigExtended(BackendConfig):
    """
    Extended backend configuration with comprehensive options.

    Extends the base BackendConfig with additional settings for
    enterprise deployments and advanced use cases.
    """

    # Collection management
    collection_name: str = "code-embeddings"
    auto_create_collection: bool = True
    collection_dimension: int = 1024

    # Advanced hybrid search settings
    sparse_index_fields: list[str] = field(default_factory=lambda: ["content", "chunk_type"])
    sparse_index_type: Literal["keyword", "text", "bm25"] = "bm25"
    hybrid_fusion_strategy: Literal["rrf", "dbsf", "linear", "convex"] = "rrf"
    hybrid_alpha: float = 0.5  # Balance between dense (1.0) and sparse (0.0)

    # Performance and scaling
    batch_size: int = 100
    max_batch_size: int = 1000
    connection_pool_size: int = 10
    enable_connection_pooling: bool = True
    enable_request_compression: bool = True

    # Caching and optimization
    enable_result_caching: bool = False
    cache_ttl_seconds: int = 300
    enable_query_optimization: bool = True

    # High availability and failover
    replica_urls: list[str] = field(default_factory=list)
    enable_failover: bool = False
    health_check_interval: int = 30

    # Security and compliance
    enable_tls: bool = True
    verify_ssl: bool = True
    client_cert_path: str | None = None
    client_key_path: str | None = None

    # Monitoring and observability
    enable_metrics: bool = False
    metrics_endpoint: str | None = None
    enable_tracing: bool = False
    trace_sample_rate: float = 0.1


def create_backend_config_from_legacy(
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    collection_name: str = "code-embeddings",
    enable_sparse_vectors: bool = False,
    **kwargs: Any,
) -> BackendConfigExtended:
    """
    Create backend configuration from legacy Qdrant-specific settings.

    Provides backward compatibility for existing CodeWeaver configurations
    while enabling migration to the new backend abstraction.

    Args:
        qdrant_url: Legacy Qdrant URL
        qdrant_api_key: Legacy Qdrant API key
        collection_name: Collection name
        enable_sparse_vectors: Enable sparse vector support
        **kwargs: Additional configuration options

    Returns:
        Backend configuration for Qdrant
    """
    return BackendConfigExtended(
        provider="qdrant",
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
        enable_sparse_vectors=enable_sparse_vectors,
        enable_hybrid_search=enable_sparse_vectors,
        **kwargs,
    )


def create_backend_config_from_env() -> BackendConfigExtended:
    """
    Create backend configuration from environment variables.

    Supports both legacy environment variables and new backend-agnostic ones:

    Legacy (Qdrant-specific):
    - QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME

    New (Backend-agnostic):
    - VECTOR_BACKEND_PROVIDER, VECTOR_BACKEND_URL, VECTOR_BACKEND_API_KEY

    Returns:
        Backend configuration from environment
    """
    # Check for new environment variables first
    provider = os.getenv("VECTOR_BACKEND_PROVIDER", "qdrant").lower()
    url = os.getenv("VECTOR_BACKEND_URL") or os.getenv("QDRANT_URL")
    api_key = os.getenv("VECTOR_BACKEND_API_KEY") or os.getenv("QDRANT_API_KEY")

    # Collection settings
    collection_name = os.getenv("COLLECTION_NAME", "code-embeddings")

    # Feature flags
    enable_hybrid = os.getenv("ENABLE_HYBRID_SEARCH", "false").lower() == "true"
    enable_sparse = os.getenv("ENABLE_SPARSE_VECTORS", "false").lower() == "true"

    # Performance settings
    batch_size = int(os.getenv("BACKEND_BATCH_SIZE", "100"))
    connection_timeout = float(os.getenv("BACKEND_CONNECTION_TIMEOUT", "30.0"))
    request_timeout = float(os.getenv("BACKEND_REQUEST_TIMEOUT", "60.0"))

    return BackendConfigExtended(
        provider=provider,
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


def migrate_config_to_toml(config: BackendConfigExtended) -> str:
    """
    Generate TOML configuration for the new backend system.

    Args:
        config: Backend configuration

    Returns:
        TOML configuration string
    """
    return f"""# Vector Database Backend Configuration
[backend]
provider = "{config.provider}"
url = "{config.url or "your-backend-url"}"
api_key = "{config.api_key or "your-api-key"}"
collection_name = "{config.collection_name}"

# Feature capabilities
enable_hybrid_search = {str(config.enable_hybrid_search).lower()}
enable_sparse_vectors = {str(config.enable_sparse_vectors).lower()}
enable_streaming = {str(config.enable_streaming).lower()}

# Performance settings
batch_size = {config.batch_size}
connection_timeout = {config.connection_timeout}
request_timeout = {config.request_timeout}
max_connections = {config.max_connections}

# Hybrid search configuration
[backend.hybrid_search]
sparse_index_fields = {config.sparse_index_fields}
sparse_index_type = "{config.sparse_index_type}"
fusion_strategy = "{config.hybrid_fusion_strategy}"
alpha = {config.hybrid_alpha}

# High availability
[backend.ha]
enable_failover = {str(config.enable_failover).lower()}
replica_urls = {config.replica_urls}
health_check_interval = {config.health_check_interval}

# Security
[backend.security]
enable_tls = {str(config.enable_tls).lower()}
verify_ssl = {str(config.verify_ssl).lower()}
# client_cert_path = "path/to/cert.pem"
# client_key_path = "path/to/key.pem"

# Monitoring
[backend.monitoring]
enable_metrics = {str(config.enable_metrics).lower()}
enable_tracing = {str(config.enable_tracing).lower()}
trace_sample_rate = {config.trace_sample_rate}
"""


# Example configurations for different backends
EXAMPLE_CONFIGS = {
    "qdrant": {
        "provider": "qdrant",
        "url": "https://your-cluster.qdrant.io",
        "api_key": "your-qdrant-api-key",
        "enable_hybrid_search": True,
        "enable_sparse_vectors": True,
    },
    "pinecone": {
        "provider": "pinecone",
        "api_key": "your-pinecone-api-key",
        "provider_options": {"environment": "us-west1-gcp", "index_name": "code-embeddings"},
    },
    "chroma": {
        "provider": "chroma",
        "url": "http://localhost:8000",
        "enable_tls": False,
        "provider_options": {"port": 8000, "headers": {}},
    },
    "weaviate": {
        "provider": "weaviate",
        "url": "https://your-cluster.weaviate.network",
        "api_key": "your-weaviate-api-key",
        "enable_hybrid_search": True,
    },
    "pgvector": {
        "provider": "pgvector",
        "url": "postgresql://user:password@localhost:5432/database",
        "provider_options": {"table_name": "code_embeddings", "dimension": 1024},
    },
}
