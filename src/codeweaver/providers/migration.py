# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Migration utilities for transitioning to the new provider system.

Provides tools to migrate existing configurations, validate provider availability,
and assist users in upgrading their CodeWeaver installations.
"""

import logging

from dataclasses import asdict
from typing import Any

from codeweaver.config import CodeWeaverConfig
from codeweaver.providers import ProviderRegistry, get_provider_factory


logger = logging.getLogger(__name__)


class ProviderMigrationHelper:
    """Helper class for migrating to the new provider system."""

    def __init__(self):
        """Initialize migration helper."""
        self.factory = get_provider_factory()
        self.registry = self.factory.registry

    def validate_current_config(self, config: CodeWeaverConfig) -> dict[str, Any]:
        """Validate current configuration against new provider system.

        Args:
            config: Current configuration to validate

        Returns:
            Validation result with status, warnings, and recommendations
        """
        result = {
            "status": "valid",
            "warnings": [],
            "recommendations": [],
            "provider_info": {},
            "migration_needed": False,
        }

        # Check embedding provider
        embedding_provider = config.embedding.provider.lower()
        if self.registry.is_embedding_provider_available(embedding_provider):
            registration = self.registry.get_embedding_provider_registration(embedding_provider)
            result["provider_info"]["embedding"] = {
                "provider": embedding_provider,
                "available": True,
                "info": asdict(registration.provider_info) if registration else None,
            }
        else:
            result["status"] = "invalid"
            result["warnings"].append(f"Embedding provider '{embedding_provider}' not available")

            # Suggest alternatives
            available = list(self.registry.get_available_embedding_providers().keys())
            if available:
                result["recommendations"].append(
                    f"Available embedding providers: {', '.join(available)}"
                )

        # Check reranking provider
        rerank_provider = config.embedding.rerank_provider or embedding_provider
        if self.registry.is_reranking_provider_available(rerank_provider):
            registration = self.registry.get_reranking_provider_registration(rerank_provider)
            result["provider_info"]["reranking"] = {
                "provider": rerank_provider,
                "available": True,
                "info": asdict(registration.provider_info) if registration else None,
            }
        else:
            result["warnings"].append(f"Reranking provider '{rerank_provider}' not available")

            # Suggest alternatives
            available = list(self.registry.get_available_reranking_providers().keys())
            if available:
                result["recommendations"].append(
                    f"Available reranking providers: {', '.join(available)}"
                )

        # Check for legacy configuration patterns
        if self._has_legacy_patterns(config):
            result["migration_needed"] = True
            result["recommendations"].extend(self._get_migration_recommendations(config))

        return result

    def _has_legacy_patterns(self, config: CodeWeaverConfig) -> bool:
        """Check if configuration has legacy patterns that should be migrated."""
        # Check for old model names that should be updated
        if config.embedding.model == "voyage-code-3" and config.embedding.provider == "openai":
            return True

        # Check for missing new configuration options
        return bool(config.embedding.rerank_provider is None and config.embedding.provider in ["voyage", "cohere"])

    def _get_migration_recommendations(self, config: CodeWeaverConfig) -> list[str]:
        """Get specific migration recommendations for the configuration."""
        recommendations = []

        # Model name migrations
        if config.embedding.model == "voyage-code-3" and config.embedding.provider == "openai":
            recommendations.append(
                "Consider updating model from 'voyage-code-3' to 'text-embedding-3-small' "
                "for OpenAI provider"
            )

        # Reranking configuration
        if config.embedding.rerank_provider is None:
            if config.embedding.provider in ["voyage", "cohere"]:
                recommendations.append(
                    f"Consider explicitly setting rerank_provider to '{config.embedding.provider}' "
                    f"to use the same provider for reranking"
                )

        return recommendations

    def get_provider_comparison(self) -> dict[str, Any]:
        """Get a comparison of available providers and their capabilities."""
        embedding_providers = self.registry.get_available_embedding_providers()
        reranking_providers = self.registry.get_available_reranking_providers()

        comparison = {
            "embedding_providers": {},
            "reranking_providers": {},
            "recommendations": {
                "best_for_code": "voyage",
                "most_cost_effective": "sentence-transformers",
                "most_versatile": "openai",
                "local_deployment": "sentence-transformers",
            },
        }

        # Embedding provider details
        for name, info in embedding_providers.items():
            comparison["embedding_providers"][name] = {
                "display_name": info.display_name,
                "description": info.description,
                "requires_api_key": info.requires_api_key,
                "supports_batch": info.supports_batch_processing,
                "max_batch_size": info.max_batch_size,
                "default_model": info.default_models.get("embedding"),
                "capabilities": [cap.value for cap in info.supported_capabilities],
            }

        # Reranking provider details
        for name, info in reranking_providers.items():
            comparison["reranking_providers"][name] = {
                "display_name": info.display_name,
                "description": info.description,
                "requires_api_key": info.requires_api_key,
                "default_model": info.default_models.get("reranking"),
                "max_documents": getattr(info, "max_documents", None),
            }

        return comparison

    def create_migration_config(self, current_config: CodeWeaverConfig) -> dict[str, Any]:
        """Create a migrated configuration with recommendations applied.

        Args:
            current_config: Current configuration to migrate

        Returns:
            Dictionary with migrated configuration
        """
        # Start with current configuration
        migrated = asdict(current_config)

        # Apply automatic migrations
        embedding_config = migrated["embedding"]

        # Model name migrations
        if (
            embedding_config["model"] == "voyage-code-3"
            and embedding_config["provider"] == "openai"
        ):
            embedding_config["model"] = "text-embedding-3-small"
            logger.info("Migrated model from voyage-code-3 to text-embedding-3-small for OpenAI")

        # Set explicit reranking provider
        if embedding_config["rerank_provider"] is None:
            provider = embedding_config["provider"]
            if self.registry.is_reranking_provider_available(provider):
                embedding_config["rerank_provider"] = provider
                logger.info("Set rerank_provider to %s to match embedding provider", provider)

        # Add new configuration sections with examples
        migrated["_migration_notes"] = [
            "This configuration has been migrated to the new provider system",
            "New features available:",
            "- Multiple embedding providers (voyage, openai, cohere, sentence-transformers, huggingface)",
            "- Separate reranking provider configuration",
            "- Local model support for sentence-transformers and huggingface",
            "- Custom dimensions and batch sizes per provider",
        ]

        return migrated

    def generate_example_configs(self) -> dict[str, dict[str, Any]]:
        """Generate example configurations for different use cases."""
        examples = {}

        # High-quality code search (VoyageAI)
        examples["high_quality_code"] = {
            "description": "Best quality for code search using VoyageAI",
            "config": {
                "embedding": {
                    "provider": "voyage",
                    "model": "voyage-code-3",
                    "dimension": 1024,
                    "rerank_provider": "voyage",
                    "rerank_model": "voyage-rerank-2",
                },
                "notes": ["Requires VOYAGE_API_KEY", "Best for code-specific embeddings"],
            },
        }

        # Cost-effective local deployment
        examples["local_deployment"] = {
            "description": "Local deployment with no API costs",
            "config": {
                "embedding": {
                    "provider": "sentence-transformers",
                    "model": "all-MiniLM-L6-v2",
                    "dimension": 384,
                    "use_local": True,
                    "device": "auto",
                    "normalize_embeddings": True,
                },
                "notes": ["No API key required", "Runs locally", "Good performance/cost ratio"],
            },
        }

        # Versatile OpenAI deployment
        examples["openai_versatile"] = {
            "description": "Versatile deployment using OpenAI embeddings",
            "config": {
                "embedding": {
                    "provider": "openai",
                    "model": "text-embedding-3-small",
                    "dimension": 1536,
                    "rerank_provider": "voyage",  # If available
                    "rerank_model": "voyage-rerank-2",
                },
                "notes": ["Requires OPENAI_API_KEY", "Good general-purpose embeddings"],
            },
        }

        # Multilingual with Cohere
        examples["multilingual"] = {
            "description": "Multilingual support using Cohere",
            "config": {
                "embedding": {
                    "provider": "cohere",
                    "model": "embed-multilingual-v3.0",
                    "dimension": 1024,
                    "rerank_provider": "cohere",
                    "rerank_model": "rerank-multilingual-v3.0",
                },
                "notes": ["Requires COHERE_API_KEY", "Best for multilingual codebases"],
            },
        }

        return examples


def validate_provider_availability() -> dict[str, Any]:
    """Validate which providers are currently available on the system.

    Returns:
        Dictionary with availability status for all providers
    """
    registry = ProviderRegistry()

    all_embedding = registry.get_all_embedding_providers()
    all_reranking = registry.get_all_reranking_providers()

    result = {
        "embedding_providers": {},
        "reranking_providers": {},
        "summary": {
            "total_embedding": len(all_embedding),
            "available_embedding": len([p for p in all_embedding.values() if p.is_available]),
            "total_reranking": len(all_reranking),
            "available_reranking": len([p for p in all_reranking.values() if p.is_available]),
        },
    }

    # Embedding provider status
    for name, registration in all_embedding.items():
        result["embedding_providers"][name] = {
            "available": registration.is_available,
            "reason": registration.unavailable_reason,
            "capabilities": [cap.value for cap in registration.capabilities],
        }

    # Reranking provider status
    for name, registration in all_reranking.items():
        result["reranking_providers"][name] = {
            "available": registration.is_available,
            "reason": registration.unavailable_reason,
            "capabilities": [cap.value for cap in registration.capabilities],
        }

    return result
