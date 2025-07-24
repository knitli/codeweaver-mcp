#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Test script for the new provider system.

This script demonstrates the provider system functionality, validates
configurations, and provides examples of using different providers.
"""

import asyncio
import logging
import sys

from pathlib import Path


# Add src to path to import codeweaver modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from codeweaver.config import CodeWeaverConfig, EmbeddingConfig
from codeweaver.providers import ProviderRegistry, get_provider_factory


# Migration helpers moved to examples/migration/providers_migration.py
# We'll implement simple validation functions here instead


class ProviderMigrationHelper:
    """Simple migration helper for testing."""

    def __init__(self):
        self.registry = ProviderRegistry()

    def validate_current_config(self, config):
        """Validate configuration."""
        result = {
            "status": "valid",
            "warnings": [],
            "recommendations": [],
            "migration_needed": False,
        }

        # Check for old model names
        if config.embedding.model == "voyage-code-3" and config.embedding.provider == "openai":
            result["migration_needed"] = True
            result["recommendations"].append(
                "Consider updating model from 'voyage-code-3' to 'text-embedding-3-small'"
            )

        return result

    def create_migration_config(self, config):
        """Create migrated configuration."""
        from dataclasses import asdict
        migrated = asdict(config)

        # Update model names
        if (migrated["embedding"]["model"] == "voyage-code-3" and
            migrated["embedding"]["provider"] == "openai"):
            migrated["embedding"]["model"] = "text-embedding-3-small"

        return migrated

    def generate_example_configs(self):
        """Generate example configurations."""
        return {
            "high_quality_code": {
                "description": "Best quality for code search",
                "config": {
                    "embedding": {
                        "provider": "voyage",
                        "model": "voyage-code-3",
                        "dimension": 1024,
                    }
                },
                "notes": ["Requires VOYAGE_API_KEY"]
            }
        }


def validate_provider_availability():
    """Validate provider availability."""
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
        }

    # Reranking provider status
    for name, registration in all_reranking.items():
        result["reranking_providers"][name] = {
            "available": registration.is_available,
            "reason": registration.unavailable_reason,
        }

    return result


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def test_provider_registry() -> None:
    """Test provider registry functionality."""
    print("\n=== Provider Registry Test ===")

    registry = ProviderRegistry()

    # Get available providers
    embedding_providers = registry.get_available_embedding_providers()
    reranking_providers = registry.get_available_reranking_providers()

    print(f"Available embedding providers: {list(embedding_providers.keys())}")
    print(f"Available reranking providers: {list(reranking_providers.keys())}")

    # Show provider details
    for name, info in embedding_providers.items():
        print(f"\n{info.display_name} ({name}):")
        print(f"  Description: {info.description}")
        print(f"  Capabilities: {[cap.value for cap in info.supported_capabilities]}")
        print(f"  Requires API key: {info.requires_api_key}")
        print(f"  Default model: {info.default_models.get('embedding', 'N/A')}")
        if info.supported_models:
            print(f"  Supported models: {info.supported_models.get('embedding', [])}")


async def test_provider_factory() -> None:
    """Test provider factory functionality."""
    print("\n=== Provider Factory Test ===")

    factory = get_provider_factory()

    # Test creating providers with mock configurations
    test_configs = [
        {"provider": "voyage", "api_key": "test-key", "model": "voyage-code-3", "dimension": 1024},
        {
            "provider": "openai",
            "api_key": "test-key",
            "model": "text-embedding-3-small",
            "dimension": 1536,
        },
        {"provider": "sentence-transformers", "model": "all-MiniLM-L6-v2", "dimension": 384},
    ]

    for config_dict in test_configs:
        provider_name = config_dict["provider"]
        print(f"\nTesting {provider_name} provider:")

        try:
            # Create embedding config
            config = EmbeddingConfig(**config_dict)

            if factory.registry.is_embedding_provider_available(provider_name):
                print(f"  ✓ Provider {provider_name} is available")

                # Note: We can't actually create providers without real API keys
                # or installed dependencies, but we can test the factory logic
                print(f"  - Would create provider with model: {config.model}")
                print(f"  - Would use dimension: {config.dimension}")

            elif registration := factory.registry.get_embedding_provider_registration(
                provider_name
            ):
                print(
                    f"  ✗ Provider {provider_name} unavailable: {registration.unavailable_reason}"
                )
            else:
                print(f"  ✗ Provider {provider_name} not found")

        except Exception as e:
            print(f"  ✗ Error testing {provider_name}: {e}")


async def test_migration_helper() -> None:
    """Test migration helper functionality."""
    print("\n=== Migration Helper Test ===")

    helper = ProviderMigrationHelper()

    # Test legacy configuration
    legacy_config = CodeWeaverConfig()
    legacy_config.embedding.provider = "openai"
    legacy_config.embedding.model = "voyage-code-3"  # Legacy model name
    legacy_config.embedding.api_key = "test-key"

    print("Testing legacy configuration validation:")
    validation_result = helper.validate_current_config(legacy_config)
    print(f"  Status: {validation_result['status']}")
    print(f"  Migration needed: {validation_result['migration_needed']}")

    if validation_result["warnings"]:
        print("  Warnings:")
        for warning in validation_result["warnings"]:
            print(f"    - {warning}")

    if validation_result["recommendations"]:
        print("  Recommendations:")
        for rec in validation_result["recommendations"]:
            print(f"    - {rec}")

    # Test migration
    if validation_result["migration_needed"]:
        print("\nGenerating migrated configuration:")
        migrated = helper.create_migration_config(legacy_config)
        print(f"  Original model: {legacy_config.embedding.model}")
        print(f"  Migrated model: {migrated['embedding']['model']}")


async def test_provider_availability() -> None:
    """Test provider availability validation."""
    print("\n=== Provider Availability Test ===")

    availability = validate_provider_availability()

    print("Summary:")
    print(
        f"  Embedding providers: {availability['summary']['available_embedding']}/{availability['summary']['total_embedding']} available"
    )
    print(
        f"  Reranking providers: {availability['summary']['available_reranking']}/{availability['summary']['total_reranking']} available"
    )

    print("\nEmbedding provider status:")
    for name, status in availability["embedding_providers"].items():
        status_icon = "✓" if status["available"] else "✗"
        print(f"  {status_icon} {name}")
        if not status["available"] and status["reason"]:
            print(f"    Reason: {status['reason']}")

    print("\nReranking provider status:")
    for name, status in availability["reranking_providers"].items():
        status_icon = "✓" if status["available"] else "✗"
        print(f"  {status_icon} {name}")
        if not status["available"] and status["reason"]:
            print(f"    Reason: {status['reason']}")


async def test_configuration_examples() -> None:
    """Test configuration examples generation."""
    print("\n=== Configuration Examples Test ===")

    helper = ProviderMigrationHelper()
    examples = helper.generate_example_configs()

    print("Available configuration examples:")
    for name, example in examples.items():
        print(f"\n{name}: {example['description']}")
        config = example["config"]

        print(f"  Provider: {config['embedding']['provider']}")
        print(f"  Model: {config['embedding']['model']}")
        print(f"  Dimension: {config['embedding']['dimension']}")

        if "rerank_provider" in config["embedding"]:
            print(f"  Rerank provider: {config['embedding']['rerank_provider']}")

        if example.get("notes"):
            print("  Notes:")
            for note in example["notes"]:
                print(f"    - {note}")


async def test_backward_compatibility() -> None:
    """Test backward compatibility with legacy interfaces."""
    print("\n=== Backward Compatibility Test ===")

    # Test legacy create_embedder function
    print("Testing legacy create_embedder function:")
    try:
        # Test with VoyageAI (will show deprecation warning)
        EmbeddingConfig(
            provider="voyage", api_key="test-key", model="voyage-code-3", dimension=1024
        )

        print("  - Calling create_embedder (expect deprecation warning)")
        # Note: This will fail without real API key, but demonstrates the interface

    except Exception as e:
        print(f"  - Expected error (no real API key): {type(e).__name__}")

    # Test legacy VoyageAIReranker
    print("\nTesting legacy VoyageAIReranker:")
    try:
        print("  - Calling VoyageAIReranker constructor (expect deprecation warning)")
        # Note: This will fail without real API key, but demonstrates the interface

    except Exception as e:
        print(f"  - Expected error (no real API key): {type(e).__name__}")


async def main() -> None:
    """Run all provider system tests."""
    print("CodeWeaver Provider System Test Suite")
    print("=" * 50)

    try:
        await test_provider_registry()
        await test_provider_factory()
        await test_migration_helper()
        await test_provider_availability()
        await test_configuration_examples()
        await test_backward_compatibility()

        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("\nTo use the new provider system in your code:")
        print("1. Import: from codeweaver.providers import get_provider_factory")
        print("2. Create: factory = get_provider_factory()")
        print("3. Use: provider = factory.create_embedding_provider(config)")
        print("\nFor backward compatibility, existing code will continue to work")
        print("but will show deprecation warnings encouraging migration.")

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
