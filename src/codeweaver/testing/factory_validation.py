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

from dataclasses import dataclass, field
from typing import Any

from codeweaver.backends.base import VectorBackend
from codeweaver.backends.factory import BackendConfig, BackendFactory
from codeweaver.config import CodeWeaverConfig
from codeweaver.providers.base import EmbeddingProvider, RerankProvider
from codeweaver.providers.factory import (
    ProviderFactory,
    create_embedding_provider,
    create_rerank_provider,
)
from codeweaver.sources.base import DataSource
from codeweaver.sources.factory import SourceFactory, create_data_source
from codeweaver.testing.mocks import (
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
    validation_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    test_details: dict[str, Any] = field(default_factory=dict)

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
        ProviderFactory.register_provider("mock_embedding", MockEmbeddingProvider)
        ProviderFactory.register_provider("mock_rerank", MockRerankProvider)

        # Register mock data source
        SourceFactory.register_source("mock", MockDataSource)

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
                "config": BackendConfig(provider="mock", url="test://localhost"),
                "expected_type": MockVectorBackend,
            },
            {
                "name": "mock_hybrid_backend",
                "config": BackendConfig(provider="mock_hybrid", url="test://localhost"),
                "expected_type": MockHybridSearchBackend,
            },
        ]

        for test_case in test_cases:
            try:
                # Test backend creation
                backend = await BackendFactory.create_backend(test_case["config"])

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
                invalid_config = BackendConfig(provider="nonexistent", url="test://localhost")
                await BackendFactory.create_backend(invalid_config)
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
                provider = await create_embedding_provider(test_case["config"])

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
            # Test list_available_providers
            providers = ProviderFactory.list_available_providers()
            if not isinstance(providers, list):
                result.validation_errors.append("list_available_providers should return list")
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
                provider = await create_rerank_provider(test_case["config"])

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
                source = await create_data_source(test_case["config"])

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
            config.backend.url = "test://localhost"

            backend = await BackendFactory.create_backend(config.backend)
            if isinstance(backend, VectorBackend):
                result.created_instances += 1
                result.test_details["backend_from_config"] = "success"
            else:
                result.validation_errors.append("Failed to create backend from config")
                result.failed_creations += 1

            # Test provider creation from configuration
            config.provider.embedding_provider = "mock_embedding"

            provider = await create_embedding_provider(config.provider.to_dict())
            if isinstance(provider, EmbeddingProvider):
                result.created_instances += 1
                result.test_details["provider_from_config"] = "success"
            else:
                result.validation_errors.append("Failed to create provider from config")
                result.failed_creations += 1

            # Test data source creation from configuration
            source_config = {"type": "mock", "enabled": True, "priority": 1, "config": {}}

            source = await create_data_source(source_config)
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
                """ A mock backend class for testing registration. """

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
                """ A mock provider class for testing registration. """

            ProviderFactory.register_provider("test_provider", TestProvider)
            providers = ProviderFactory.list_available_providers()

            if "test_provider" in providers:
                result.created_instances += 1
                result.test_details["provider_registration"] = "success"
            else:
                result.validation_errors.append("Provider registration failed")
                result.failed_creations += 1

            # Test source registration
            class TestSource:
                """ A mock data source class for testing registration. """

            SourceFactory.register_source("test_source", TestSource)
            sources = SourceFactory.list_available_sources()

            if "test_source" in sources:
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
    total_valid = sum(bool(result.is_valid)
                  for result in results.values())
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
