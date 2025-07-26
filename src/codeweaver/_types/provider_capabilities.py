# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Provider capabilities model using Pydantic v2.

Centralized provider capability definitions with validation and serialization support.
Replaces scattered hardcoded capabilities across provider implementations.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class ProviderCapabilities(BaseModel):
    """Centralized provider capability definitions.

    This Pydantic model consolidates all provider capabilities in one place,
    replacing hardcoded attributes scattered across provider implementations.
    """

    # Core capabilities
    supports_embedding: bool = Field(False, description="Supports document embedding")
    supports_reranking: bool = Field(False, description="Supports document reranking")
    supports_batch_processing: bool = Field(False, description="Efficient batch operations")
    supports_streaming: bool = Field(False, description="Streaming responses")
    supports_rate_limiting: bool = Field(False, description="Built-in rate limiting")

    # Model capabilities
    supports_custom_dimensions: bool = Field(
        False, description="Supports custom embedding dimensions"
    )
    supports_multiple_models: bool = Field(False, description="Multiple model support")
    supports_model_switching: bool = Field(False, description="Runtime model switching")
    supports_local_inference: bool = Field(False, description="Local inference without API calls")

    # Performance characteristics
    max_batch_size: Annotated[int | None, Field(None, ge=1, description="Maximum batch size")]
    max_input_length: Annotated[int | None, Field(None, ge=1, description="Maximum input length")]
    max_concurrent_requests: Annotated[
        int, Field(10, ge=1, le=100, description="Max concurrent requests")
    ]

    # Rate limiting
    requests_per_minute: Annotated[int | None, Field(None, ge=1, description="Request rate limit")]
    tokens_per_minute: Annotated[int | None, Field(None, ge=1, description="Token rate limit")]

    # Dependencies
    requires_api_key: bool = Field(True, description="Requires API key")
    required_dependencies: list[str] = Field(default_factory=list, description="Required packages")
    optional_dependencies: list[str] = Field(default_factory=list, description="Optional packages")

    # Model configuration
    default_embedding_model: str | None = Field(None, description="Default embedding model")
    default_reranking_model: str | None = Field(None, description="Default reranking model")
    supported_embedding_models: list[str] = Field(
        default_factory=list, description="Supported embedding models"
    )
    supported_reranking_models: list[str] = Field(
        default_factory=list, description="Supported reranking models"
    )

    # Native dimensions mapping
    native_dimensions: dict[str, int] = Field(
        default_factory=dict, description="Model -> native dimensions mapping"
    )
