# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Factory for creating vector database backends.

Provides dynamic backend instantiation supporting 15+ vector databases
with automatic capability detection and fallback mechanisms.
"""

import logging

from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from codeweaver.backends.base import HybridSearchBackend, VectorBackend
from codeweaver.backends.qdrant import QdrantBackend, QdrantHybridBackend


logger = logging.getLogger(__name__)

# Type variables for factory returns
BackendT = TypeVar("BackendT", bound=VectorBackend)


@dataclass
class BackendConfig:
    """Configuration for vector database backends."""

    # Core connection settings
    provider: Literal[
        "qdrant",
        "pinecone",
        "chroma",
        "weaviate",
        "pgvector",
        "redis",
        "elasticsearch",
        "opensearch",
        "milvus",
        "vespa",
        "faiss",
        "annoy",
        "scann",
        "lancedb",
        "marqo",
    ]
    url: str | None = None
    api_key: str | None = None

    # Advanced capabilities
    enable_hybrid_search: bool = False
    enable_sparse_vectors: bool = False
    enable_streaming: bool = False
    enable_transactions: bool = False

    # Performance settings
    connection_timeout: float = 30.0
    request_timeout: float = 60.0
    max_connections: int = 10
    retry_count: int = 3

    # Storage preferences
    prefer_memory: bool = False
    prefer_disk: bool = False

    # Provider-specific settings
    provider_options: dict[str, Any] | None = None


class BackendFactory:
    """
    Factory for creating vector database backends.

    Supports dynamic instantiation of 15+ vector databases with
    automatic capability detection and intelligent fallbacks.
    """

    # Registry of available backends
    _backends: dict[str, tuple[type[VectorBackend], bool]] = {
        # (backend_class, supports_hybrid)
        "qdrant": (QdrantBackend, True)
        # Future backends will be registered here
        # "pinecone": (PineconeBackend, False),
        # "chroma": (ChromaBackend, False),
        # "weaviate": (WeaviateBackend, True),
        # "pgvector": (PgVectorBackend, False),
        # "redis": (RedisBackend, False),
        # "elasticsearch": (ElasticsearchBackend, True),
        # "opensearch": (OpenSearchBackend, True),
        # "milvus": (MilvusBackend, False),
        # "vespa": (VespaBackend, True),
        # "faiss": (FAISSBackend, False),
        # "annoy": (AnnoyBackend, False),
        # "scann": (ScaNNBackend, False),
        # "lancedb": (LanceDBBackend, False),
        # "marqo": (MarqoBackend, True),
    }

    @classmethod
    def create_backend(cls, config: BackendConfig) -> VectorBackend:
        """
        Create a vector database backend based on configuration.

        Args:
            config: Backend configuration

        Returns:
            Configured backend instance

        Raises:
            ValueError: If provider is not supported
            ConnectionError: If backend connection fails
        """
        provider = config.provider.lower()

        if provider not in cls._backends:
            available = ", ".join(cls._backends.keys())
            raise ValueError(
                f"Unsupported backend provider: {provider}. Available providers: {available}"
            )

        backend_class, supports_hybrid = cls._backends[provider]

        # Choose hybrid version if requested and supported
        if config.enable_hybrid_search and supports_hybrid:
            backend_class = cls._get_hybrid_backend_class(provider)

        # Build backend-specific arguments
        backend_args = cls._build_backend_args(config)

        try:
            logger.info("Creating %s backend with hybrid=%s", provider, config.enable_hybrid_search)
            return backend_class(**backend_args)

        except Exception as e:
            logger.exception("Failed to create %s backend", provider)
            raise ConnectionError(f"Failed to create {provider} backend") from e

    @classmethod
    def _get_hybrid_backend_class(cls, provider: str) -> type[HybridSearchBackend]:
        """Get the hybrid search version of a backend."""
        if provider == "qdrant":
            return QdrantHybridBackend
        # Add other hybrid backends here
        # elif provider == "weaviate":
        #     return WeaviateHybridBackend
        # elif provider == "vespa":
        #     return VespaHybridBackend

        # Fallback to basic backend
        backend_class, _ = cls._backends[provider]
        return backend_class

    @classmethod
    def _build_backend_args(cls, config: BackendConfig) -> dict[str, Any]:
        """Build backend-specific arguments from configuration."""
        args = {"url": config.url, "api_key": config.api_key}

        # Add provider-specific settings
        if config.provider == "qdrant":
            args.update({
                "enable_sparse_vectors": config.enable_sparse_vectors,
                "sparse_on_disk": config.prefer_disk,
                "timeout": config.request_timeout,
            })
        # Add other provider-specific configurations here

        # Add any custom provider options
        if config.provider_options:
            args.update(config.provider_options)

        # Remove None values
        return {k: v for k, v in args.items() if v is not None}

    @classmethod
    def list_supported_providers(cls) -> dict[str, dict[str, bool]]:
        """
        List all supported providers and their capabilities.

        Returns:
            Dictionary mapping provider names to their capabilities
        """
        providers = {}
        for provider, (backend_class, supports_hybrid) in cls._backends.items():
            providers[provider] = {
                "available": True,
                "supports_hybrid_search": supports_hybrid,
                "supports_streaming": hasattr(backend_class, "stream_upsert"),
                "supports_transactions": hasattr(backend_class, "begin_transaction"),
            }

        # Add planned providers (not yet implemented)
        planned_providers = {
            "pinecone": {"available": False, "supports_hybrid_search": False},
            "chroma": {"available": False, "supports_hybrid_search": False},
            "weaviate": {"available": False, "supports_hybrid_search": True},
            "pgvector": {"available": False, "supports_hybrid_search": False},
            "redis": {"available": False, "supports_hybrid_search": False},
            "elasticsearch": {"available": False, "supports_hybrid_search": True},
            "opensearch": {"available": False, "supports_hybrid_search": True},
            "milvus": {"available": False, "supports_hybrid_search": False},
            "vespa": {"available": False, "supports_hybrid_search": True},
            "faiss": {"available": False, "supports_hybrid_search": False},
            "annoy": {"available": False, "supports_hybrid_search": False},
            "scann": {"available": False, "supports_hybrid_search": False},
            "lancedb": {"available": False, "supports_hybrid_search": False},
            "marqo": {"available": False, "supports_hybrid_search": True},
        }

        providers.update(planned_providers)
        return providers

    @classmethod
    def register_backend(
        cls, provider: str, backend_class: type[VectorBackend], supports_hybrid: bool = False
    ) -> None:
        """
        Register a new backend provider.

        Args:
            provider: Provider name
            backend_class: Backend implementation class
            supports_hybrid: Whether the backend supports hybrid search
        """
        cls._backends[provider.lower()] = (backend_class, supports_hybrid)
        logger.info("Registered backend provider: %s", provider)

    @classmethod
    def create_from_url(cls, url: str, **kwargs: Any) -> VectorBackend:
        """
        Create a backend from a connection URL.

        Supports URLs like:
        - qdrant://api-key@cluster-url:6333/collection-name
        - pinecone://api-key@environment.pinecone.io/index-name
        - postgres://user:pass@host:5432/db (for pgvector)

        Args:
            url: Connection URL
            **kwargs: Additional configuration options

        Returns:
            Configured backend instance
        """
        # Parse URL to determine provider and connection details
        if url.startswith("qdrant://"):
            provider = "qdrant"
            # Parse Qdrant URL format
            # This is a simplified parser - in production, use proper URL parsing
            base_url = url.replace("qdrant://", "https://")
        elif url.startswith(("postgres://", "postgresql://")):
            provider = "pgvector"
            base_url = url
        else:
            raise ValueError(f"Unsupported URL scheme: {url}")

        config = BackendConfig(provider=provider, url=base_url, **kwargs)

        return cls.create_backend(config)


# Convenience function for backward compatibility
def create_backend(config: BackendConfig) -> VectorBackend:
    """Create a vector database backend from configuration."""
    return BackendFactory.create_backend(config)
