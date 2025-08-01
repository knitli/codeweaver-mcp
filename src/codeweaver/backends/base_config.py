# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Base configuration classes for vector database backends."""

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from codeweaver.cw_types import ProviderKind
from codeweaver.providers.base import (
    CombinedProvider,
    EmbeddingProvider,
    LocalEmbeddingProvider,
    RerankProvider,
)


CustomBackendCapabilities = dict[str, Any]


class BackendConfig(BaseModel):
    """Configuration for vector database backends."""

    model_config = ConfigDict(
        extra="allow",
        str_strip_whitespace=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
    provider: Annotated[
        type[CombinedProvider]
        | type[EmbeddingProvider]
        | type[LocalEmbeddingProvider]
        | type[RerankProvider]
        | str,
        Field(description="Backend provider type or string identifier"),
    ]
    kind: Annotated[ProviderKind, Field(description="Provider kind classification")]
    name: Annotated[
        str | None, Field(default=None, description="Custom name for the backend instance")
    ]
    url: Annotated[str | None, Field(default=None, description="Backend connection URL")]
    api_key: Annotated[str | None, Field(default=None, description="API key for authentication")]
    capabilities: Annotated[
        CustomBackendCapabilities | None,
        Field(
            default=None,
            description="Capabilities of the backend, required for custom backends to define their features",
        ),
    ]
    enable_hybrid_search: Annotated[
        bool, Field(default=False, description="Enable hybrid dense/sparse search")
    ]
    enable_sparse_vectors: Annotated[
        bool, Field(default=False, description="Enable sparse vector support")
    ]
    enable_streaming: Annotated[
        bool, Field(default=False, description="Enable streaming operations")
    ]
    enable_transactions: Annotated[
        bool, Field(default=False, description="Enable transaction support")
    ]
    connection_timeout: Annotated[
        float,
        Field(
            default=30.0, ge=0.1, le=300.0, description="Connection timeout in seconds (0.1-300)"
        ),
    ]
    request_timeout: Annotated[
        float,
        Field(default=60.0, ge=0.1, le=300.0, description="Request timeout in seconds (0.1-300)"),
    ]
    max_connections: Annotated[
        int, Field(default=10, ge=1, le=1000, description="Maximum number of connections (1-1000)")
    ]
    retry_count: Annotated[
        int, Field(default=3, ge=0, le=10, description="Number of retry attempts (0-10)")
    ]
    prefer_memory: Annotated[bool, Field(default=False, description="Prefer memory-based storage")]
    prefer_disk: Annotated[bool, Field(default=False, description="Prefer disk-based storage")]
    provider_options: Annotated[
        dict[str, Any] | None,
        Field(
            default=None,
            description="Additional options specific to the backend provider. Providers should validate these options themselves.",
        ),
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
            if not url.startswith((
                "http://",
                "https://",
                "postgresql://",
                "postgres://",
                "mongodb://",
            )):
                raise ValueError(f"URL must be a valid connection string, got: {url}")
        return url
