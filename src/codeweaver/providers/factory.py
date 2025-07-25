# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Provider factory and registry system for dynamic provider management.

Enables registration, discovery, and creation of embedding and reranking providers
with support for capability detection and configuration validation.
"""

import logging

from dataclasses import dataclass
from typing import ClassVar

from codeweaver._types import EmbeddingProviderInfo, ProviderCapability, ProviderType
from codeweaver.providers.base import (
    CombinedProvider,
    EmbeddingProvider,
    EmbeddingProviderBase,
    RerankProvider,
    RerankProviderBase,
)
from codeweaver.providers.config import EmbeddingProviderConfig, RerankingProviderConfig


logger = logging.getLogger(__name__)


@dataclass
class ProviderRegistration:
    """Registration information for a provider."""

    provider_class: type[EmbeddingProviderBase | RerankProviderBase | CombinedProvider]
    capabilities: list[ProviderCapability]
    provider_info: EmbeddingProviderInfo
    is_available: bool = True
    unavailable_reason: str | None = None


class ProviderRegistry:
    """Registry for managing available embedding and reranking providers."""

    _embedding_providers: ClassVar[dict[str, ProviderRegistration]] = {}
    _reranking_providers: ClassVar[dict[str, ProviderRegistration]] = {}
    _initialized: ClassVar[bool] = False

    @classmethod
    def register_embedding_provider(
        cls,
        name: str,
        provider_class: type[EmbeddingProviderBase | CombinedProvider],
        provider_info: EmbeddingProviderInfo,
        *,
        check_availability: bool = True,
    ) -> None:
        """Register an embedding provider.

        Args:
            name: Unique provider name
            provider_class: Provider implementation class
            provider_info: Provider capability information
            check_availability: Whether to check if provider dependencies are available
        """
        is_available = True
        unavailable_reason = None

        if check_availability:
            is_available, unavailable_reason = cls._check_provider_availability(
                provider_class, ProviderCapability.EMBEDDING
            )

        registration = ProviderRegistration(
            provider_class=provider_class,
            capabilities=[ProviderCapability.EMBEDDING],
            provider_info=provider_info,
            is_available=is_available,
            unavailable_reason=unavailable_reason,
        )

        cls._embedding_providers[name] = registration

        if is_available:
            logger.info("Registered embedding provider: %s", name)
        else:
            logger.warning(
                "Registered embedding provider %s (unavailable: %s)", name, unavailable_reason
            )

    @classmethod
    def register_reranking_provider(
        cls,
        name: str,
        provider_class: type[RerankProviderBase | CombinedProvider],
        provider_info: EmbeddingProviderInfo,
        *,
        check_availability: bool = True,
    ) -> None:
        """Register a reranking provider.

        Args:
            name: Unique provider name
            provider_class: Provider implementation class
            provider_info: Provider capability information
            check_availability: Whether to check if provider dependencies are available
        """
        is_available = True
        unavailable_reason = None

        if check_availability:
            is_available, unavailable_reason = cls._check_provider_availability(
                provider_class, ProviderCapability.RERANKING
            )

        registration = ProviderRegistration(
            provider_class=provider_class,
            capabilities=[ProviderCapability.RERANKING],
            provider_info=provider_info,
            is_available=is_available,
            unavailable_reason=unavailable_reason,
        )

        cls._reranking_providers[name] = registration

        if is_available:
            logger.info("Registered reranking provider: %s", name)
        else:
            logger.warning(
                "Registered reranking provider %s (unavailable: %s)", name, unavailable_reason
            )

    @classmethod
    def register_combined_provider(
        cls,
        name: str,
        provider_class: type[CombinedProvider],
        provider_info: EmbeddingProviderInfo,
        *,
        check_availability: bool = True,
    ) -> None:
        """Register a provider that supports both embedding and reranking.

        Args:
            name: Unique provider name
            provider_class: Combined provider implementation class
            provider_info: Provider capability information
            check_availability: Whether to check if provider dependencies are available
        """
        is_available = True
        unavailable_reason = None

        if check_availability:
            # Check both capabilities
            embed_available, embed_reason = cls._check_provider_availability(
                provider_class, ProviderCapability.EMBEDDING
            )
            rerank_available, rerank_reason = cls._check_provider_availability(
                provider_class, ProviderCapability.RERANKING
            )

            is_available = embed_available and rerank_available
            if not is_available:
                reasons = []
                if not embed_available and embed_reason:
                    reasons.append(f"embedding: {embed_reason}")
                if not rerank_available and rerank_reason:
                    reasons.append(f"reranking: {rerank_reason}")
                unavailable_reason = "; ".join(reasons)

        # Register for embedding
        embed_registration = ProviderRegistration(
            provider_class=provider_class,
            capabilities=[ProviderCapability.EMBEDDING],
            provider_info=provider_info,
            is_available=is_available,
            unavailable_reason=unavailable_reason,
        )
        cls._embedding_providers[name] = embed_registration

        # Register for reranking
        rerank_registration = ProviderRegistration(
            provider_class=provider_class,
            capabilities=[ProviderCapability.RERANKING],
            provider_info=provider_info,
            is_available=is_available,
            unavailable_reason=unavailable_reason,
        )
        cls._reranking_providers[name] = rerank_registration

        if is_available:
            logger.info("Registered combined provider: %s", name)
        else:
            logger.warning(
                "Registered combined provider %s (unavailable: %s)", name, unavailable_reason
            )

    @classmethod
    def _check_provider_availability(
        cls,
        provider_class: type[EmbeddingProviderBase | RerankProviderBase | CombinedProvider],
        capability: ProviderCapability,
    ) -> tuple[bool, str | None]:
        """Check if a provider is available by testing its dependencies.

        Args:
            provider_class: Provider class to check
            capability: The capability to check availability for

        Returns:
            Tuple of (is_available, reason_if_unavailable)
        """
        try:
            # Try to get provider info without instantiating
            if hasattr(provider_class, "check_availability"):
                return provider_class.check_availability(capability)  # type: ignore

        except Exception as e:
            return False, str(e)

        else:
            # Default: assume available (will fail at runtime if not)
            return True, None

    @classmethod
    def get_available_providers(
        cls, capability: ProviderCapability
    ) -> dict[str, EmbeddingProviderInfo]:
        """Get all available providers for a specific capability."""
        cls._ensure_initialized()
        if capability == ProviderCapability.EMBEDDING:
            return cls.get_available_embedding_providers()
        if capability == ProviderCapability.RERANKING:
            return cls.get_available_reranking_providers()
        return {}

    @classmethod
    def get_provider_info(cls, provider_type: ProviderType) -> EmbeddingProviderInfo | None:
        """Get provider info by provider type."""
        cls._ensure_initialized()
        # Map provider type to name
        provider_name = provider_type.value

        # Check embedding providers first
        embedding_providers = cls.get_available_embedding_providers()
        if provider_name in embedding_providers:
            return embedding_providers[provider_name]

        # Check reranking providers
        reranking_providers = cls.get_available_reranking_providers()
        if provider_name in reranking_providers:
            return reranking_providers[provider_name]

        return None

    @classmethod
    def get_available_embedding_providers(cls) -> dict[str, EmbeddingProviderInfo]:
        """Get all available embedding providers."""
        cls._ensure_initialized()
        return {
            name: reg.provider_info
            for name, reg in cls._embedding_providers.items()
            if reg.is_available
        }

    @classmethod
    def get_available_reranking_providers(cls) -> dict[str, EmbeddingProviderInfo]:
        """Get all available reranking providers."""
        cls._ensure_initialized()
        return {
            name: reg.provider_info
            for name, reg in cls._reranking_providers.items()
            if reg.is_available
        }

    @classmethod
    def get_all_embedding_providers(cls) -> dict[str, ProviderRegistration]:
        """Get all embedding providers (including unavailable ones)."""
        cls._ensure_initialized()
        return cls._embedding_providers.copy()

    @classmethod
    def get_all_reranking_providers(cls) -> dict[str, ProviderRegistration]:
        """Get all reranking providers (including unavailable ones)."""
        cls._ensure_initialized()
        return cls._reranking_providers.copy()

    @classmethod
    def is_embedding_provider_available(cls, name: str) -> bool:
        """Check if an embedding provider is available."""
        cls._ensure_initialized()
        registration = cls._embedding_providers.get(name)
        return registration is not None and registration.is_available

    @classmethod
    def is_reranking_provider_available(cls, name: str) -> bool:
        """Check if a reranking provider is available."""
        cls._ensure_initialized()
        registration = cls._reranking_providers.get(name)
        return registration is not None and registration.is_available

    @classmethod
    def get_embedding_provider_registration(cls, name: str) -> ProviderRegistration | None:
        """Get embedding provider registration by name."""
        cls._ensure_initialized()
        return cls._embedding_providers.get(name)

    @classmethod
    def get_reranking_provider_registration(cls, name: str) -> ProviderRegistration | None:
        """Get reranking provider registration by name."""
        cls._ensure_initialized()
        return cls._reranking_providers.get(name)

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Ensure the registry is initialized with built-in providers."""
        if not cls._initialized:
            cls._register_builtin_providers()
            cls._initialized = True

    @classmethod
    def _register_builtin_providers(cls) -> None:
        """Register built-in providers."""
        # This will be called during module import to register providers
        # Import here to avoid circular dependencies
        try:
            from codeweaver.providers.voyage import VoyageAIProvider

            cls.register_combined_provider(
                "voyage-ai", VoyageAIProvider, VoyageAIProvider.get_static_provider_info()
            )
        except ImportError:
            logger.debug("VoyageAI provider not available")

        try:
            from codeweaver.providers.openai import OpenAICompatibleProvider, OpenAIProvider

            # Register legacy OpenAI provider
            cls.register_embedding_provider(
                "openai", OpenAIProvider, OpenAIProvider.get_static_provider_info()
            )

            # Register flexible OpenAI-compatible provider
            cls.register_embedding_provider(
                "openai-compatible",
                OpenAICompatibleProvider,
                OpenAICompatibleProvider.get_static_provider_info(),
            )
        except ImportError:
            logger.debug("OpenAI providers not available")

        try:
            from codeweaver.providers.cohere import CohereProvider

            cls.register_combined_provider(
                "cohere", CohereProvider, CohereProvider.get_static_provider_info()
            )
        except ImportError:
            logger.debug("Cohere provider not available")

        try:
            from codeweaver.providers.sentence_transformers import SentenceTransformersProvider

            cls.register_embedding_provider(
                "sentence-transformers",
                SentenceTransformersProvider,
                SentenceTransformersProvider.get_static_provider_info(),
            )
        except ImportError:
            logger.debug("SentenceTransformers provider not available")

        try:
            from codeweaver.providers.huggingface import HuggingFaceProvider

            cls.register_embedding_provider(
                "huggingface", HuggingFaceProvider, HuggingFaceProvider.get_static_provider_info()
            )
        except ImportError:
            logger.debug("HuggingFace provider not available")


class ProviderFactory:
    """Factory for creating embedding and reranking provider instances."""

    def __init__(self, registry: ProviderRegistry | None = None):
        """Initialize the factory.

        Args:
            registry: Provider registry to use (uses global registry if None)
        """
        self.registry = registry or ProviderRegistry

    def create_embedding_provider(self, config: EmbeddingProviderConfig) -> EmbeddingProvider:
        """Create an embedding provider based on configuration.

        Args:
            config: Embedding provider configuration (Pydantic model)

        Returns:
            Configured embedding provider instance

        Raises:
            ValueError: If provider is unknown or unavailable
            RuntimeError: If provider cannot be instantiated
        """

        def raise_type_error(provider_name: str) -> None:
            """Raise a TypeError for invalid provider."""
            raise TypeError(
                f"Provider {provider_name} does not implement EmbeddingProvider protocol"
            )

        provider_name = config.provider_type.value.lower()
        registration = self.registry.get_embedding_provider_registration(provider_name)

        if registration is None:
            available = list(self.registry.get_available_embedding_providers().keys())
            raise ValueError(
                f"Unknown embedding provider: {provider_name}. "
                f"Available providers: {', '.join(available)}"
            )

        if not registration.is_available:
            raise ValueError(
                f"Embedding provider '{provider_name}' is not available: "
                f"{registration.unavailable_reason}"
            )

        try:
            # Pass config directly to provider - providers handle their own configuration
            provider = registration.provider_class(config)

            # Ensure it implements the correct protocol
            if not isinstance(provider, EmbeddingProvider | EmbeddingProviderBase):
                raise_type_error(provider_name)

        except Exception as e:
            raise RuntimeError(f"Failed to create embedding provider {provider_name}: {e}") from e

        else:
            return provider

    def create_reranking_provider(self, config: RerankingProviderConfig) -> RerankProvider:
        """Create a reranking provider.

        Args:
            config: Reranking provider configuration (Pydantic model)

        Returns:
            Configured reranking provider instance

        Raises:
            ValueError: If provider is unknown or unavailable
            RuntimeError: If provider cannot be instantiated
        """

        def raise_type_error(provider_name: str) -> None:
            """Raise a TypeError for invalid provider."""
            raise TypeError(f"Provider {provider_name} does not implement RerankProvider protocol")

        provider_name = config.provider_type.value.lower()
        registration = self.registry.get_reranking_provider_registration(provider_name)

        if registration is None:
            available = list(self.registry.get_available_reranking_providers().keys())
            raise ValueError(
                f"Unknown reranking provider: {provider_name}. "
                f"Available providers: {', '.join(available)}"
            )

        if not registration.is_available:
            raise ValueError(
                f"Reranking provider '{provider_name}' is not available: "
                f"{registration.unavailable_reason}"
            )

        try:
            # Pass config directly to provider - providers handle their own configuration
            provider = registration.provider_class(config)

            # Ensure it implements the correct protocol
            if not isinstance(provider, RerankProvider | RerankProviderBase):
                raise_type_error(provider_name)

        except Exception as e:
            raise RuntimeError(f"Failed to create reranking provider '{provider_name}': {e}") from e

        else:
            return provider

    def get_default_reranking_provider(
        self, embedding_provider_name: str, config: RerankingProviderConfig
    ) -> RerankProvider | None:
        """Get the default reranking provider for an embedding provider.

        Args:
            embedding_provider_name: Name of the embedding provider
            config: Reranking provider configuration

        Returns:
            Default reranking provider or None if none available
        """
        # If the embedding provider also supports reranking, use it
        if self.registry.is_reranking_provider_available(embedding_provider_name):
            try:
                return self.create_reranking_provider(config)
            except Exception:
                logger.warning(
                    "Failed to create reranking provider for %s", embedding_provider_name
                )

        # Fallback to VoyageAI if available (best reranking quality)
        if self.registry.is_reranking_provider_available("voyage-ai"):
            try:
                # Create VoyageAI config for fallback
                from codeweaver.providers.config import VoyageConfig

                voyage_config = VoyageConfig(api_key=config.api_key, rerank_model=config.model)
                return self.create_reranking_provider(voyage_config)
            except Exception:
                logger.warning("Failed to create fallback VoyageAI reranking provider")

        return None


# Global factory instance
_provider_factory: ProviderFactory | None = None


def get_provider_factory() -> ProviderFactory:
    """Get the global provider factory instance."""
    global _provider_factory
    if _provider_factory is None:
        _provider_factory = ProviderFactory()
    return _provider_factory
