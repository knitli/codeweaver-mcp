# sourcery skip: lambdas-should-be-short
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

from typing import Annotated, Any, ClassVar, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from codeweaver.backends.base import BackendConnectionError, HybridSearchBackend, VectorBackend
from codeweaver.backends.qdrant import QdrantHybridBackend
from codeweaver.providers.base import (
    CombinedProvider,
    EmbeddingProvider,
    LocalEmbeddingProvider,
    ProviderKind,
    RerankProvider,
)


logger = logging.getLogger(__name__)

# Type variables for factory returns
BackendT = TypeVar("BackendT", bound=VectorBackend)

# Custom backend capabilities type
CustomBackendCapabilities = dict[str, Any]


class BackendConfig(BaseModel):
    """Configuration for vector database backends."""

    model_config = ConfigDict(
        extra="allow",
        str_strip_whitespace=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    # Core connection settings
    provider: Annotated[
        type[CombinedProvider] | type[EmbeddingProvider] | type[LocalEmbeddingProvider] | type[RerankProvider] | str,
        Field(description="Backend provider type or string identifier")
    ]
    kind: Annotated[ProviderKind, Field(description="Provider kind classification")]
    name: Annotated[
        str | None,
        Field(default=None, description="Custom name for the backend instance")
    ]
    url: Annotated[
        str | None,
        Field(default=None, description="Backend connection URL")
    ]
    api_key: Annotated[
        str | None,
        Field(default=None, description="API key for authentication")
    ]
    capabilities: Annotated[
        CustomBackendCapabilities | None,
        Field(
            default=None,
            description="Capabilities of the backend, required for custom backends to define their features"
        )
    ]

    # Advanced capabilities
    enable_hybrid_search: Annotated[
        bool,
        Field(default=False, description="Enable hybrid dense/sparse search")
    ]
    enable_sparse_vectors: Annotated[
        bool,
        Field(default=False, description="Enable sparse vector support")
    ]
    enable_streaming: Annotated[
        bool,
        Field(default=False, description="Enable streaming operations")
    ]
    enable_transactions: Annotated[
        bool,
        Field(default=False, description="Enable transaction support")
    ]

    # Performance settings
    connection_timeout: Annotated[
        float,
        Field(
            default=30.0,
            ge=0.1,
            le=300.0,
            description="Connection timeout in seconds (0.1-300)"
        )
    ]
    request_timeout: Annotated[
        float,
        Field(
            default=60.0,
            ge=0.1,
            le=300.0,
            description="Request timeout in seconds (0.1-300)"
        )
    ]
    max_connections: Annotated[
        int,
        Field(
            default=10,
            ge=1,
            le=1000,
            description="Maximum number of connections (1-1000)"
        )
    ]
    retry_count: Annotated[
        int,
        Field(
            default=3,
            ge=0,
            le=10,
            description="Number of retry attempts (0-10)"
        )
    ]

    # Storage preferences
    prefer_memory: Annotated[
        bool,
        Field(default=False, description="Prefer memory-based storage")
    ]
    prefer_disk: Annotated[
        bool,
        Field(default=False, description="Prefer disk-based storage")
    ]

    # Provider-specific settings
    provider_options: Annotated[
        dict[str, Any] | None,
        Field(
            default=None,
            description="Additional options specific to the backend provider. Providers should validate these options themselves."
        )
    ]

    @field_validator("api_key", mode="before")
    @classmethod
    def validate_api_key(cls, api_key: str | None) -> str | None:
        """Validate API key format."""
        return None if api_key is not None and len(api_key.strip()) == 0 else api_key

    @field_validator("url", mode="before")
    @classmethod
    def validate_url(cls, url: str | None) -> str | None:
        """Validate URL format."""
        if url is not None:
            url = url.strip()
            if len(url) == 0:
                return None
            # Basic URL validation - must start with http/https for web services
            if not (url.startswith(("http://", "https://", "postgresql://", "postgres://", "mongodb://"))):
                raise ValueError(f"URL must be a valid connection string, got: {url}")
        return url


class BackendFactory:
    """
    Factory for creating vector database backends.

    Supports dynamic instantiation of 15+ vector databases with
    automatic capability detection and intelligent fallbacks.
    """

    # Registry of available backends
    _backends: ClassVar[
        dict[
            Literal[  # We're being explicit about the expected keys
                "annoy",  # noqa: PYI051
                "chroma",  # noqa: PYI051
                "custom",  # noqa: PYI051
                "faiss",  # noqa: PYI051
                "lancedb",  # noqa: PYI051
                "marqo",  # noqa: PYI051
                "milvus",  # noqa: PYI051
                "opensearch",  # noqa: PYI051
                "pgvector",  # noqa: PYI051
                "pinecone",  # noqa: PYI051
                "qdrant",  # noqa: PYI051
                "redis",  # noqa: PYI051
                "scann",  # noqa: PYI051
                "vespa",  # noqa: PYI051
                "weaviate",  # noqa: PYI051
            ]
            | str,
            ProviderKind,
        ]
    ] = {}

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
            BackendConnectionError: If backend connection fails
        """
        # Handle both string and enum provider values
        if hasattr(config.provider, 'value'):
            provider = config.provider.value.lower()
        else:
            provider = str(config.provider).lower()

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
            raise BackendConnectionError(f"Failed to create {provider} backend", backend_type=provider) from e

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

        # Handle both string and enum provider values
        if hasattr(config.provider, 'value'):
            provider = config.provider.value.lower()
        else:
            provider = str(config.provider).lower()

        # Add provider-specific settings
        if provider == "qdrant":
            args |= {
                "enable_sparse_vectors": config.enable_sparse_vectors,
                "sparse_on_disk": config.prefer_disk,
                "timeout": config.request_timeout,
            }
        # Add other provider-specific configurations here

        # Add any custom provider options
        if config.provider_options:
            args |= config.provider_options

        # Remove None values
        return {k: v for k, v in args.items() if v is not None}

    @classmethod
    def list_supported_providers(cls) -> dict[str, dict[str, bool]]:
        """
        List all supported providers and their capabilities.

        Returns:
            Dictionary mapping provider names to their capabilities
        """
        providers = {
            provider: {
                "available": True,
                "supports_hybrid_search": supports_hybrid,
                "supports_streaming": hasattr(backend_class, "stream_upsert"),
                "supports_transactions": hasattr(
                    backend_class, "begin_transaction"
                ),
            }
            for provider, (backend_class, supports_hybrid) in cls._backends.items()
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

        providers |= planned_providers
        return providers

    @classmethod
    def register_backend(
        cls, provider: str, backend_class: type[VectorBackend], *, supports_hybrid: bool = False
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

        # Ensure we have a valid ProviderKind for the config
        from codeweaver.providers.base import ProviderKind
        kind = kwargs.pop('kind', ProviderKind.COMBINED)

        config = BackendConfig(provider=provider, kind=kind, url=base_url, **kwargs)

        return cls.create_backend(config)


# Register default backends
def _register_default_backends() -> None:
    """Register the default backend implementations."""
    try:
        from codeweaver.backends.qdrant import QdrantHybridBackend

        # Register Qdrant as both basic and hybrid backend
        BackendFactory.register_backend("qdrant", QdrantHybridBackend, supports_hybrid=True)
        logger.info("Registered Qdrant backend")

    except ImportError as e:
        logger.warning("Failed to register Qdrant backend: %s", e)


# Auto-register backends on module import
_register_default_backends()
