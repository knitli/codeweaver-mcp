# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Factory pattern validation utilities for CodeWeaver.

Validates that factory patterns work correctly with configuration loading,
component instantiation, and error handling across all protocols.
"""

import contextlib
import logging

from typing import Any

from pydantic import Field
from pydantic.dataclasses import dataclass

from codeweaver.backends import BackendConfig, BackendFactory, VectorBackend
from codeweaver.config import CodeWeaverConfig
from codeweaver.cw_types import ProviderType
from codeweaver.providers import (
    EmbeddingProvider,
    ProviderFactory,
    ProviderRegistry,
    RerankProvider,
)
from codeweaver.sources import DataSource, SourceFactory
from codeweaver.testing import (
    MockDataSource,
    MockEmbeddingProvider,
    MockHybridSearchBackend,
    MockRerankProvider,
    MockVectorBackend,
)


logger = logging.getLogger(__name__)


@dataclass
class FactoryValidationResult:
    """Result of factory pattern validation."""

    factory_name: str
    is_valid: bool
    created_instances: int
    failed_creations: int
    validation_errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    test_details: dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        return (
            f"{status} {self.factory_name}: "
            f"{self.created_instances} created, {self.failed_creations} failed"
        )

    def get_detailed_report(self) -> str:
        """Get detailed validation report."""
        lines = [
            f"Factory Validation Report: {self.factory_name}",
            "=" * (30 + len(self.factory_name)),
            f"Status: {'VALID' if self.is_valid else 'INVALID'}",
            f"Successful Creations: {self.created_instances}",
            f"Failed Creations: {self.failed_creations}",
            "",
        ]

        if self.validation_errors:
            lines.extend([
                "Validation Errors:",
                *[f"  - {error}" for error in self.validation_errors],
                "",
            ])

        if self.warnings:
            lines.extend(["Warnings:", *[f"  - {warning}" for warning in self.warnings], ""])

        if self.test_details:
            lines.extend([
                "Test Details:",
                *[f"  - {key}: {value}" for key, value in self.test_details.items()],
                "",
            ])

        return "\n".join(lines)


class FactoryPatternValidator:
    """Comprehensive factory pattern validator."""

    def __init__(self):
        """Initialize factory pattern validator."""
        self.register_mock_factories()

    def register_mock_factories(self) -> None:
        """Register mock implementations with factories for testing."""
        # Register mock backend
        BackendFactory.register_backend("mock", MockVectorBackend, supports_hybrid=False)
        BackendFactory.register_backend(
            "mock_hybrid", MockHybridSearchBackend, supports_hybrid=True
        )

        # Register mock providers
        from codeweaver.cw_types import (
            EmbeddingProviderInfo,
            ProviderCapabilities,
            ProviderCapability,
        )
        from codeweaver.providers.factory import ProviderRegistry

        # Create mock provider info
        mock_provider_info = EmbeddingProviderInfo(
            name="mock_embedding",
            display_name="Mock Embedding Provider",
            description="Test embedding provider for validation",
            supported_capabilities=[ProviderCapability.EMBEDDING],
            capabilities=ProviderCapabilities(
                supports_embedding=True,
                supports_reranking=False,
                supports_batch_processing=True,
                max_batch_size=32,
                max_input_length=8192,
                requires_api_key=False,
                default_embedding_model="mock-model",
                supported_embedding_models=["mock-model"],
                native_dimensions={"mock-model": 1536},
            ),
        )

        mock_rerank_info = EmbeddingProviderInfo(
            name="mock_rerank",
            display_name="Mock Rerank Provider",
            description="Test reranking provider for validation",
            supported_capabilities=[ProviderCapability.RERANKING],
            capabilities=ProviderCapabilities(
                supports_embedding=False,
                supports_reranking=True,
                supports_batch_processing=True,
                max_batch_size=32,
                max_input_length=8192,
                requires_api_key=False,
                default_reranking_model="mock-rerank-model",
                supported_reranking_models=["mock-rerank-model"],
            ),
        )

        ProviderRegistry.register_embedding_provider(
            "mock_embedding", MockEmbeddingProvider, mock_provider_info, check_availability=False
        )
        ProviderRegistry.register_reranking_provider(
            "mock_rerank", MockRerankProvider, mock_rerank_info, check_availability=False
        )

        # Register mock data source
        from codeweaver.sources.base import get_source_registry

        source_registry = get_source_registry()
        source_registry.register("mock", MockDataSource)

    async def validate_all_factories(self) -> dict[str, FactoryValidationResult]:
        """Validate all factory patterns."""
        results = {"backend_factory": await self.validate_backend_factory()}

        # Validate provider factories
        results["embedding_provider_factory"] = await self.validate_embedding_provider_factory()
        results["rerank_provider_factory"] = await self.validate_rerank_provider_factory()

        # Validate data source factory
        results["data_source_factory"] = await self.validate_data_source_factory()

        # Validate configuration integration
        results["configuration_integration"] = await self.validate_configuration_integration()

        return results

    async def validate_backend_factory(self) -> FactoryValidationResult:
        """Validate backend factory pattern."""
        result = FactoryValidationResult(
            factory_name="BackendFactory", is_valid=True, created_instances=0, failed_creations=0
        )

        # Test cases for backend creation
        test_cases = [
            {
                "name": "mock_backend",
                "config": BackendConfig(
                    provider="mock", kind="combined", url="http://localhost:6333"
                ),
                "expected_type": MockVectorBackend,
            },
            {
                "name": "mock_hybrid_backend",
                "config": BackendConfig(
                    provider="mock_hybrid", kind="combined", url="http://localhost:6333"
                ),
                "expected_type": MockHybridSearchBackend,
            },
        ]

        for test_case in test_cases:
            try:
                # Test backend creation
                backend = BackendFactory.create_backend(test_case["config"])

                # Validate instance type
                if not isinstance(backend, test_case["expected_type"]):
                    result.validation_errors.append(
                        f"Expected {test_case['expected_type'].__name__}, got {type(backend).__name__}"
                    )
                    result.failed_creations += 1
                else:
                    result.created_instances += 1
                    result.test_details[test_case["name"]] = "success"

                # Test protocol compliance
                if not isinstance(backend, VectorBackend):
                    result.validation_errors.append(
                        f"Backend {type(backend).__name__} doesn't implement VectorBackend protocol"
                    )

            except Exception as e:
                result.validation_errors.append(f"Failed to create {test_case['name']}: {e}")
                result.failed_creations += 1
                result.test_details[test_case["name"]] = f"error: {e}"

        # Test factory methods
        try:
            # Test list_supported_providers
            providers = BackendFactory.list_supported_providers()
            if not isinstance(providers, dict):
                result.validation_errors.append("list_supported_providers should return dict")
            elif "mock" not in providers:
                result.warnings.append("Mock provider not found in supported providers")

            # Test invalid provider
            with contextlib.suppress(Exception):
                invalid_config = BackendConfig(
                    provider="nonexistent", kind="combined", url="http://localhost:6333"
                )
                BackendFactory.create_backend(invalid_config)
                result.validation_errors.append("Should have failed with invalid provider")
        except Exception as e:
            result.validation_errors.append(f"Factory method test failed: {e}")

        # Update validity
        if result.validation_errors or result.failed_creations > 0:
            result.is_valid = False

        return result

    async def validate_embedding_provider_factory(self) -> FactoryValidationResult:
        """Validate embedding provider factory pattern."""
        result = FactoryValidationResult(
            factory_name="EmbeddingProviderFactory",
            is_valid=True,
            created_instances=0,
            failed_creations=0,
        )

        # Test cases for provider creation
        test_cases = [
            {
                "name": "mock_embedding",
                "config": {"provider": "mock_embedding"},
                "expected_type": MockEmbeddingProvider,
            }
        ]

        for test_case in test_cases:
            try:
                # Test provider creation
                from codeweaver.providers.config import EmbeddingProviderConfig

                config = EmbeddingProviderConfig(provider_type=ProviderType.MOCK_EMBEDDING)
                factory = ProviderFactory()
                provider = factory.create_embedding_provider(config)

                # Validate instance type
                if not isinstance(provider, test_case["expected_type"]):
                    result.validation_errors.append(
                        f"Expected {test_case['expected_type'].__name__}, got {type(provider).__name__}"
                    )
                    result.failed_creations += 1
                else:
                    result.created_instances += 1
                    result.test_details[test_case["name"]] = "success"

                # Test protocol compliance
                if not isinstance(provider, EmbeddingProvider):
                    result.validation_errors.append(
                        f"Provider {type(provider).__name__} doesn't implement EmbeddingProvider protocol"
                    )

                # Test basic functionality
                info = provider.get_provider_info()
                if not info or not info.name:
                    result.validation_errors.append("Provider info is invalid")

            except Exception as e:
                result.validation_errors.append(f"Failed to create {test_case['name']}: {e}")
                result.failed_creations += 1
                result.test_details[test_case["name"]] = f"error: {e}"

        # Test factory methods
        try:
            # Test get_available_embedding_providers
            providers = ProviderRegistry.get_available_embedding_providers()
            if not isinstance(providers, dict):
                result.validation_errors.append(
                    "get_available_embedding_providers should return dict"
                )
            elif "mock_embedding" not in providers:
                result.warnings.append("Mock embedding provider not found in available providers")

        except Exception as e:
            result.validation_errors.append(f"Factory method test failed: {e}")

        # Update validity
        if result.validation_errors or result.failed_creations > 0:
            result.is_valid = False

        return result

    async def validate_rerank_provider_factory(self) -> FactoryValidationResult:
        """Validate rerank provider factory pattern."""
        result = FactoryValidationResult(
            factory_name="RerankProviderFactory",
            is_valid=True,
            created_instances=0,
            failed_creations=0,
        )

        # Test cases for provider creation
        test_cases = [
            {
                "name": "mock_rerank",
                "config": {"provider": "mock_rerank"},
                "expected_type": MockRerankProvider,
            }
        ]

        for test_case in test_cases:
            try:
                # Test provider creation
                from codeweaver.providers.config import RerankingProviderConfig

                config = RerankingProviderConfig(provider_type=ProviderType.MOCK_RERANK)
                factory = ProviderFactory()
                provider = factory.create_reranking_provider(config)

                # Validate instance type
                if not isinstance(provider, test_case["expected_type"]):
                    result.validation_errors.append(
                        f"Expected {test_case['expected_type'].__name__}, got {type(provider).__name__}"
                    )
                    result.failed_creations += 1
                else:
                    result.created_instances += 1
                    result.test_details[test_case["name"]] = "success"

                # Test protocol compliance
                if not isinstance(provider, RerankProvider):
                    result.validation_errors.append(
                        f"Provider {type(provider).__name__} doesn't implement RerankProvider protocol"
                    )

                # Test basic functionality
                info = provider.get_provider_info()
                if not info or not info.name:
                    result.validation_errors.append("Provider info is invalid")

            except Exception as e:
                result.validation_errors.append(f"Failed to create {test_case['name']}: {e}")
                result.failed_creations += 1
                result.test_details[test_case["name"]] = f"error: {e}"

        # Update validity
        if result.validation_errors or result.failed_creations > 0:
            result.is_valid = False

        return result

    async def validate_data_source_factory(self) -> FactoryValidationResult:
        """Validate data source factory pattern."""
        result = FactoryValidationResult(
            factory_name="DataSourceFactory", is_valid=True, created_instances=0, failed_creations=0
        )

        # Test cases for source creation
        test_cases = [
            {
                "name": "mock_source",
                "config": {"type": "mock", "enabled": True, "priority": 1, "config": {}},
                "expected_type": MockDataSource,
            }
        ]

        for test_case in test_cases:
            try:
                # Test source creation
                source = await test_case["config"].source.create_data_source(test_case["config"])

                # Validate instance type
                if not isinstance(source, test_case["expected_type"]):
                    result.validation_errors.append(
                        f"Expected {test_case['expected_type'].__name__}, got {type(source).__name__}"
                    )
                    result.failed_creations += 1
                else:
                    result.created_instances += 1
                    result.test_details[test_case["name"]] = "success"

                # Test protocol compliance
                if not isinstance(source, DataSource):
                    result.validation_errors.append(
                        f"Source {type(source).__name__} doesn't implement DataSource protocol"
                    )

                # Test basic functionality
                capabilities = source.get_capabilities()
                if not capabilities:
                    result.validation_errors.append("Source capabilities are empty")

            except Exception as e:
                result.validation_errors.append(f"Failed to create {test_case['name']}: {e}")
                result.failed_creations += 1
                result.test_details[test_case["name"]] = f"error: {e}"

        # Test factory methods
        try:
            # Test list_available_sources
            sources = SourceFactory.list_available_sources()
            if not isinstance(sources, list):
                result.validation_errors.append("list_available_sources should return list")
            elif "mock" not in sources:
                result.warnings.append("Mock source not found in available sources")

        except Exception as e:
            result.validation_errors.append(f"Factory method test failed: {e}")

        # Update validity
        if result.validation_errors or result.failed_creations > 0:
            result.is_valid = False

        return result

    async def validate_configuration_integration(self) -> FactoryValidationResult:
        """Validate configuration integration with factories."""
        result = FactoryValidationResult(
            factory_name="ConfigurationIntegration",
            is_valid=True,
            created_instances=0,
            failed_creations=0,
        )

        try:
            # Test configuration creation
            config = CodeWeaverConfig()

            # Test backend creation from configuration
            config.backend.provider = "mock"
            config.backend.kind = "combined"
            config.backend.url = "http://localhost:6333"

            backend = BackendFactory.create_backend(config.backend)
            if isinstance(backend, VectorBackend):
                result.created_instances += 1
                result.test_details["backend_from_config"] = "success"
            else:
                result.validation_errors.append("Failed to create backend from config")
                result.failed_creations += 1

            # Test provider creation from configuration
            config.provider.embedding_provider = "mock_embedding"

            provider = await config.provider.create_embedding_provider(config.provider.model_dump())
            if isinstance(provider, EmbeddingProvider):
                result.created_instances += 1
                result.test_details["provider_from_config"] = "success"
            else:
                result.validation_errors.append("Failed to create provider from config")
                result.failed_creations += 1

            # Test data source creation from configuration
            source_config = {"type": "mock", "enabled": True, "priority": 1, "config": {}}

            source = await config.provider.create_data_source(source_config)
            if isinstance(source, DataSource):
                result.created_instances += 1
                result.test_details["source_from_config"] = "success"
            else:
                result.validation_errors.append("Failed to create source from config")
                result.failed_creations += 1

            # Test configuration validation
            if not config.backend.provider:
                result.validation_errors.append("Configuration validation failed")

        except Exception as e:
            result.validation_errors.append(f"Configuration integration failed: {e}")
            result.failed_creations += 1

        # Update validity
        if result.validation_errors or result.failed_creations > 0:
            result.is_valid = False

        return result

    def validate_factory_registration(self) -> FactoryValidationResult:
        """Validate factory registration mechanisms."""
        result = FactoryValidationResult(
            factory_name="FactoryRegistration",
            is_valid=True,
            created_instances=0,
            failed_creations=0,
        )

        try:
            # Test backend registration
            class TestBackend:
                """A mock backend class for testing registration."""

            BackendFactory.register_backend("test", TestBackend, supports_hybrid=True)
            providers = BackendFactory.list_supported_providers()

            if "test" in providers:
                result.created_instances += 1
                result.test_details["backend_registration"] = "success"
            else:
                result.validation_errors.append("Backend registration failed")
                result.failed_creations += 1

            # Test provider registration
            class TestProvider:
                """A mock provider class for testing registration."""

            from codeweaver.cw_types import (
                EmbeddingProviderInfo,
                ProviderCapabilities,
                ProviderCapability,
            )
            from codeweaver.providers.factory import ProviderRegistry

            # Create test provider info
            test_provider_info = EmbeddingProviderInfo(
                name="test_provider",
                display_name="Test Provider",
                description="Test provider for validation",
                supported_capabilities=[ProviderCapability.EMBEDDING],
                capabilities=ProviderCapabilities(
                    supports_embedding=True,
                    supports_reranking=False,
                    supports_batch_processing=True,
                    max_batch_size=32,
                    max_input_length=8192,
                    requires_api_key=False,
                    default_embedding_model="test-model",
                    supported_embedding_models=["test-model"],
                    native_dimensions={"test-model": 1536},
                ),
            )

            ProviderRegistry.register_embedding_provider(
                "test_provider", TestProvider, test_provider_info, check_availability=False
            )
            providers = ProviderRegistry.get_available_embedding_providers()

            if "test_provider" in providers:
                result.created_instances += 1
                result.test_details["provider_registration"] = "success"
            else:
                result.validation_errors.append("Provider registration failed")
                result.failed_creations += 1

            # Test source registration
            class TestSource:
                """A mock data source class for testing registration."""

            from codeweaver.sources.base import get_source_registry

            source_registry = get_source_registry()
            source_registry.register("test_source", TestSource)

            # Check if source was registered
            source_class = source_registry.get_source_class("test_source")
            if source_class is TestSource:
                result.created_instances += 1
                result.test_details["source_registration"] = "success"
            else:
                result.validation_errors.append("Source registration failed")
                result.failed_creations += 1

        except Exception as e:
            result.validation_errors.append(f"Factory registration failed: {e}")
            result.failed_creations += 1

        # Update validity
        if result.validation_errors or result.failed_creations > 0:
            result.is_valid = False

        return result


# Convenience functions


async def validate_all_factory_patterns() -> dict[str, FactoryValidationResult]:
    """Validate all factory patterns in CodeWeaver."""
    validator = FactoryPatternValidator()
    return await validator.validate_all_factories()


async def validate_factory_pattern(factory_name: str) -> FactoryValidationResult:
    """Validate a specific factory pattern."""
    validator = FactoryPatternValidator()

    if factory_name == "backend":
        return await validator.validate_backend_factory()
    if factory_name == "embedding_provider":
        return await validator.validate_embedding_provider_factory()
    if factory_name == "rerank_provider":
        return await validator.validate_rerank_provider_factory()
    if factory_name == "data_source":
        return await validator.validate_data_source_factory()
    if factory_name == "configuration":
        return await validator.validate_configuration_integration()
    if factory_name == "registration":
        return validator.validate_factory_registration()
    result = FactoryValidationResult(
        factory_name=factory_name, is_valid=False, created_instances=0, failed_creations=1
    )
    result.validation_errors.append(f"Unknown factory pattern: {factory_name}")
    return result


def print_factory_validation_results(results: dict[str, FactoryValidationResult]) -> None:
    """Print formatted factory validation results."""
    print("\n" + "=" * 80)
    print("FACTORY PATTERN VALIDATION RESULTS")
    print("=" * 80)

    for result in results.values():
        print(f"\n{result}")

        if result.validation_errors:
            print("  Errors:")
            for error in result.validation_errors:
                print(f"    - {error}")

        if result.warnings:
            print("  Warnings:")
            for warning in result.warnings:
                print(f"    - {warning}")

    # Summary
    total_valid = sum(bool(result.is_valid) for result in results.values())
    total_factories = len(results)

    print(f"\nSUMMARY: {total_valid}/{total_factories} factory patterns valid")

    if total_valid == total_factories:
        print("✅ All factory patterns are working correctly!")
    else:
        print("❌ Some factory patterns need attention.")


def save_factory_validation_results(
    results: dict[str, FactoryValidationResult], filename: str
) -> None:
    """Save factory validation results to JSON file."""
    import json

    serializable_results = {
        factory_name: {
            "factory_name": result.factory_name,
            "is_valid": result.is_valid,
            "created_instances": result.created_instances,
            "failed_creations": result.failed_creations,
            "validation_errors": result.validation_errors,
            "warnings": result.warnings,
            "test_details": result.test_details,
        }
        for factory_name, result in results.items()
    }
    with open(filename, "w") as f:
        json.dump(serializable_results, f, indent=2)

    logger.info("Factory validation results saved to: %s", filename)
