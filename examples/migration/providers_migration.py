# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Example demonstrating how to migrate to the new provider system
supporting multiple embedding and reranking providers.
"""

import logging

from dataclasses import asdict

from codeweaver.config import CodeWeaverConfig
from codeweaver.providers import ProviderRegistry, get_provider_factory


logger = logging.getLogger(__name__)


def check_provider_availability() -> None:
    """Example: Check which providers are available on the system."""
    registry = ProviderRegistry()

    print("Checking provider availability...")
    print("-" * 50)

    # Check embedding providers
    embedding_providers = registry.get_all_embedding_providers()
    print("\nEmbedding Providers:")
    for name, registration in embedding_providers.items():
        status = "✓ Available" if registration.is_available else "✗ Unavailable"
        print(f"  {name}: {status}")
        if not registration.is_available:
            print(f"    Reason: {registration.unavailable_reason}")

    # Check reranking providers
    reranking_providers = registry.get_all_reranking_providers()
    print("\nReranking Providers:")
    for name, registration in reranking_providers.items():
        status = "✓ Available" if registration.is_available else "✗ Unavailable"
        print(f"  {name}: {status}")
        if not registration.is_available:
            print(f"    Reason: {registration.unavailable_reason}")


def validate_current_configuration():
    """Example: Validate existing configuration against new provider system."""
    # Create a sample configuration
    config = CodeWeaverConfig()

    # Validate with migration helper
    helper = ProviderMigrationHelper()
    validation_result = helper.validate_current_config(config)

    print("\nConfiguration Validation:")
    print("-" * 50)
    print(f"Status: {validation_result['status']}")

    if validation_result['warnings']:
        print("\nWarnings:")
        for warning in validation_result['warnings']:
            print(f"  ⚠ {warning}")

    if validation_result['recommendations']:
        print("\nRecommendations:")
        for rec in validation_result['recommendations']:
            print(f"  → {rec}")

    return validation_result


def compare_providers() -> None:
    """Example: Compare available providers and their capabilities."""
    helper = ProviderMigrationHelper()
    comparison = helper.get_provider_comparison()

    print("\nProvider Comparison:")
    print("-" * 50)

    print("\nEmbedding Providers:")
    for name, info in comparison['embedding_providers'].items():
        print(f"\n  {name}:")
        print(f"    Name: {info['display_name']}")
        print(f"    Description: {info['description']}")
        print(f"    API Key Required: {info['requires_api_key']}")
        print(f"    Batch Support: {info['supports_batch']}")
        print(f"    Default Model: {info['default_model']}")

    print("\n\nRecommendations:")
    for use_case, provider in comparison['recommendations'].items():
        print(f"  {use_case.replace('_', ' ').title()}: {provider}")


def generate_example_configurations() -> None:
    """Example: Generate configurations for different use cases."""
    helper = ProviderMigrationHelper()
    examples = helper.generate_example_configs()

    print("\nExample Configurations:")
    print("-" * 50)

    for example in examples.values():
        print(f"\n{example['description']}:")
        print(f"Config: {example['config']}")
        for note in example['config']['notes']:
            print(f"  - {note}")


def migrate_legacy_configuration() -> None:
    """Example: Migrate a legacy configuration to the new system."""
    # Create a legacy configuration
    legacy_config = CodeWeaverConfig()
    legacy_config.embedding.provider = "openai"
    legacy_config.embedding.model = "voyage-code-3"  # Old model name

    print("\nMigrating Legacy Configuration:")
    print("-" * 50)
    print(f"Original: provider={legacy_config.embedding.provider}, "
          f"model={legacy_config.embedding.model}")

    # Migrate configuration
    helper = ProviderMigrationHelper()
    migrated = helper.create_migration_config(legacy_config)

    print(f"Migrated: provider={migrated['embedding']['provider']}, "
          f"model={migrated['embedding']['model']}")

    if '_migration_notes' in migrated:
        print("\nMigration Notes:")
        for note in migrated['_migration_notes']:
            print(f"  - {note}")


def use_provider_factory() -> None:
    """Example: Using the provider factory to create embedders and rerankers."""
    factory = get_provider_factory()

    print("\nUsing Provider Factory:")
    print("-" * 50)

    # List available providers
    available_embedding = factory.registry.get_available_embedding_providers()
    print(f"\nAvailable embedding providers: {list(available_embedding.keys())}")

    # Create an embedder (example with sentence-transformers if available)
    if "sentence-transformers" in available_embedding:
        config = CodeWeaverConfig()
        config.embedding.provider = "sentence-transformers"
        config.embedding.model = "all-MiniLM-L6-v2"

        try:
            embedder = factory.create_embedder(config)
            print(f"\n✓ Created embedder: {embedder.__class__.__name__}")
            print(f"  Provider: {config.embedding.provider}")
            print(f"  Model: {config.embedding.model}")
        except Exception as e:
            print(f"\n✗ Failed to create embedder: {e}")


class ProviderMigrationHelper:
    """Helper class for provider migration examples."""

    def __init__(self):
        self.factory = get_provider_factory()
        self.registry = self.factory.registry

    def validate_current_config(self, config):
        """Validate configuration (simplified for example)."""
        result = {
            "status": "valid",
            "warnings": [],
            "recommendations": [],
            "provider_info": {},
        }

        # Check if provider is available
        provider = config.embedding.provider
        if not self.registry.is_embedding_provider_available(provider):
            result["status"] = "invalid"
            result["warnings"].append(f"Provider '{provider}' not available")

            # Suggest alternatives
            available = list(self.registry.get_available_embedding_providers().keys())
            if available:
                result["recommendations"].append(
                    f"Available providers: {', '.join(available)}"
                )

        return result

    def get_provider_comparison(self):
        """Get provider comparison data."""
        return {
            "embedding_providers": {
                name: {
                    "display_name": info.display_name,
                    "description": info.description,
                    "requires_api_key": info.requires_api_key,
                    "supports_batch": info.supports_batch_processing,
                    "default_model": info.default_models.get("embedding"),
                }
                for name, info in self.registry.get_available_embedding_providers().items()
            },
            "recommendations": {
                "best_for_code": "voyage",
                "most_cost_effective": "sentence-transformers",
                "most_versatile": "openai",
                "local_deployment": "sentence-transformers",
            }
        }

    def create_migration_config(self, config):
        """Create migrated configuration."""
        migrated = asdict(config)

        # Example migration: update model names
        if (migrated["embedding"]["model"] == "voyage-code-3" and
            migrated["embedding"]["provider"] == "openai"):
            migrated["embedding"]["model"] = "text-embedding-3-small"

        # Add migration notes
        migrated["_migration_notes"] = [
            "Configuration migrated to new provider system",
            "Multiple providers now available",
            "Separate reranking configuration supported",
        ]

        return migrated

    def generate_example_configs(self):
        """Generate example configurations."""
        return {
            "high_quality_code": {
                "description": "Best quality for code search using VoyageAI",
                "config": {
                    "embedding": {
                        "provider": "voyage",
                        "model": "voyage-code-3",
                        "dimension": 1024,
                        "rerank_provider": "voyage",
                        "rerank_model": "voyage-rerank-2",
                    },
                    "notes": ["Requires VOYAGE_API_KEY", "Best for code embeddings"],
                },
            },
            "local_deployment": {
                "description": "Local deployment with no API costs",
                "config": {
                    "embedding": {
                        "provider": "sentence-transformers",
                        "model": "all-MiniLM-L6-v2",
                        "dimension": 384,
                    },
                    "notes": ["No API key required", "Runs locally"],
                },
            },
        }


def main() -> None:
    """Run all provider migration examples."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Provider System Migration Examples")
    print("=" * 60)

    # Example 1: Check availability
    check_provider_availability()

    # Example 2: Validate configuration
    validate_current_configuration()

    # Example 3: Compare providers
    compare_providers()

    # Example 4: Generate examples
    generate_example_configurations()

    # Example 5: Migrate legacy config
    migrate_legacy_configuration()

    # Example 6: Use provider factory
    use_provider_factory()

    print("\n" + "=" * 60)
    print("Provider migration examples completed!")


if __name__ == "__main__":
    main()
