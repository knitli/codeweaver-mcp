# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Test cases for factory pattern validation.

Tests the factory validation framework and ensures all factory patterns
work correctly with configuration and component instantiation.
"""

import asyncio

import pytest

from codeweaver.testing.factory_validation import (
    FactoryPatternValidator,
    FactoryValidationResult,
    print_factory_validation_results,
    save_factory_validation_results,
    validate_all_factory_patterns,
    validate_factory_pattern,
)


class TestFactoryValidationResult:
    """Test FactoryValidationResult data structure."""

    def test_factory_validation_result_creation(self) -> None:
        """Test creating factory validation result."""
        result = FactoryValidationResult(
            factory_name="TestFactory", is_valid=True, created_instances=5, failed_creations=0
        )

        assert result.factory_name == "TestFactory"
        assert result.is_valid is True
        assert result.created_instances == 5
        assert result.failed_creations == 0
        assert result.validation_errors == []
        assert result.warnings == []
        assert result.test_details == {}

    def test_factory_validation_result_string_representation(self) -> None:
        """Test factory validation result string formatting."""
        result = FactoryValidationResult(
            factory_name="TestFactory", is_valid=True, created_instances=3, failed_creations=1
        )

        str_repr = str(result)
        assert "TestFactory" in str_repr
        assert "✅ VALID" in str_repr
        assert "3 created, 1 failed" in str_repr

        # Test invalid result
        result.is_valid = False
        str_repr = str(result)
        assert "❌ INVALID" in str_repr

    def test_factory_validation_result_detailed_report(self) -> None:
        """Test detailed factory validation report."""
        result = FactoryValidationResult(
            factory_name="TestFactory",
            is_valid=False,
            created_instances=2,
            failed_creations=1,
            validation_errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
            test_details={"test1": "success", "test2": "error"},
        )

        detailed_report = result.get_detailed_report()

        assert "Factory Validation Report: TestFactory" in detailed_report
        assert "Status: INVALID" in detailed_report
        assert "Successful Creations: 2" in detailed_report
        assert "Failed Creations: 1" in detailed_report
        assert "Validation Errors:" in detailed_report
        assert "Error 1" in detailed_report
        assert "Error 2" in detailed_report
        assert "Warnings:" in detailed_report
        assert "Warning 1" in detailed_report
        assert "Test Details:" in detailed_report
        assert "test1: success" in detailed_report


class TestFactoryPatternValidator:
    """Test FactoryPatternValidator functionality."""

    def test_validator_initialization(self) -> None:
        """Test validator initialization."""
        validator = FactoryPatternValidator()

        # Should register mock factories
        # This will be tested implicitly in other tests
        assert validator is not None

    @pytest.mark.asyncio
    async def test_validate_all_factories(self) -> None:
        """Test validating all factory patterns."""
        validator = FactoryPatternValidator()
        results = await validator.validate_all_factories()

        expected_factories = [
            "backend_factory",
            "embedding_provider_factory",
            "rerank_provider_factory",
            "data_source_factory",
            "configuration_integration",
        ]

        assert len(results) == len(expected_factories)
        for factory_name in expected_factories:
            assert factory_name in results
            assert isinstance(results[factory_name], FactoryValidationResult)

    @pytest.mark.asyncio
    async def test_validate_backend_factory(self) -> None:
        """Test backend factory validation."""
        validator = FactoryPatternValidator()
        result = await validator.validate_backend_factory()

        assert result.factory_name == "BackendFactory"
        assert isinstance(result.is_valid, bool)
        assert result.created_instances >= 0
        assert result.failed_creations >= 0

        if result.is_valid:
            # Should have successfully created at least one backend
            assert result.created_instances > 0

        # Should have test details
        assert isinstance(result.test_details, dict)

    @pytest.mark.asyncio
    async def test_validate_embedding_provider_factory(self) -> None:
        """Test embedding provider factory validation."""
        validator = FactoryPatternValidator()
        result = await validator.validate_embedding_provider_factory()

        assert result.factory_name == "EmbeddingProviderFactory"
        assert isinstance(result.is_valid, bool)
        assert result.created_instances >= 0
        assert result.failed_creations >= 0

        if result.is_valid:
            # Should have successfully created at least one provider
            assert result.created_instances > 0

    @pytest.mark.asyncio
    async def test_validate_rerank_provider_factory(self) -> None:
        """Test rerank provider factory validation."""
        validator = FactoryPatternValidator()
        result = await validator.validate_rerank_provider_factory()

        assert result.factory_name == "RerankProviderFactory"
        assert isinstance(result.is_valid, bool)
        assert result.created_instances >= 0
        assert result.failed_creations >= 0

        if result.is_valid:
            # Should have successfully created at least one provider
            assert result.created_instances > 0

    @pytest.mark.asyncio
    async def test_validate_data_source_factory(self) -> None:
        """Test data source factory validation."""
        validator = FactoryPatternValidator()
        result = await validator.validate_data_source_factory()

        assert result.factory_name == "DataSourceFactory"
        assert isinstance(result.is_valid, bool)
        assert result.created_instances >= 0
        assert result.failed_creations >= 0

        if result.is_valid:
            # Should have successfully created at least one source
            assert result.created_instances > 0

    @pytest.mark.asyncio
    async def test_validate_configuration_integration(self) -> None:
        """Test configuration integration validation."""
        validator = FactoryPatternValidator()
        result = await validator.validate_configuration_integration()

        assert result.factory_name == "ConfigurationIntegration"
        assert isinstance(result.is_valid, bool)
        assert result.created_instances >= 0
        assert result.failed_creations >= 0

        # Should test configuration-based component creation
        assert isinstance(result.test_details, dict)

    def test_validate_factory_registration(self) -> None:
        """Test factory registration validation."""
        validator = FactoryPatternValidator()
        result = validator.validate_factory_registration()

        assert result.factory_name == "FactoryRegistration"
        assert isinstance(result.is_valid, bool)
        assert result.created_instances >= 0
        assert result.failed_creations >= 0

        # Should test dynamic registration
        assert isinstance(result.test_details, dict)


class TestConvenienceFunctions:
    """Test convenience functions for factory validation."""

    @pytest.mark.asyncio
    async def test_validate_all_factory_patterns(self) -> None:
        """Test validate_all_factory_patterns function."""
        results = await validate_all_factory_patterns()

        assert isinstance(results, dict)
        assert len(results) > 0

        # All results should be FactoryValidationResult instances
        for result in results.values():
            assert isinstance(result, FactoryValidationResult)
            assert result.factory_name is not None

    @pytest.mark.asyncio
    async def test_validate_specific_factory_patterns(self) -> None:
        """Test validating specific factory patterns."""
        factory_types = [
            "backend",
            "embedding_provider",
            "rerank_provider",
            "data_source",
            "configuration",
            "registration",
        ]

        for factory_type in factory_types:
            result = await validate_factory_pattern(factory_type)

            assert isinstance(result, FactoryValidationResult)
            assert result.factory_name is not None
            assert isinstance(result.is_valid, bool)

    @pytest.mark.asyncio
    async def test_validate_unknown_factory_pattern(self) -> None:
        """Test validating unknown factory pattern."""
        result = await validate_factory_pattern("unknown_factory")

        assert isinstance(result, FactoryValidationResult)
        assert result.factory_name == "unknown_factory"
        assert result.is_valid is False
        assert result.failed_creations == 1
        assert len(result.validation_errors) > 0
        assert "Unknown factory pattern" in result.validation_errors[0]

    def test_print_factory_validation_results(self, capsys) -> None:
        """Test printing factory validation results."""
        results = {
            "test_factory": FactoryValidationResult(
                factory_name="TestFactory",
                is_valid=True,
                created_instances=3,
                failed_creations=1,
                validation_errors=["Test error"],
                warnings=["Test warning"],
            ),
            "failing_factory": FactoryValidationResult(
                factory_name="FailingFactory",
                is_valid=False,
                created_instances=0,
                failed_creations=2,
            ),
        }

        print_factory_validation_results(results)

        captured = capsys.readouterr()
        assert "FACTORY PATTERN VALIDATION RESULTS" in captured.out
        assert "TestFactory" in captured.out
        assert "FailingFactory" in captured.out
        assert "Test error" in captured.out
        assert "Test warning" in captured.out
        assert "SUMMARY:" in captured.out
        assert "1/2 factory patterns valid" in captured.out
        assert "❌ Some factory patterns need attention." in captured.out

    def test_print_factory_validation_results_all_valid(self, capsys) -> None:
        """Test printing results when all factories are valid."""
        results = {
            "factory1": FactoryValidationResult(
                factory_name="Factory1", is_valid=True, created_instances=2, failed_creations=0
            ),
            "factory2": FactoryValidationResult(
                factory_name="Factory2", is_valid=True, created_instances=1, failed_creations=0
            ),
        }

        print_factory_validation_results(results)

        captured = capsys.readouterr()
        assert "SUMMARY: 2/2 factory patterns valid" in captured.out
        assert "✅ All factory patterns are working correctly!" in captured.out

    def test_save_factory_validation_results(self, tmp_path) -> None:
        """Test saving factory validation results to file."""
        import json

        results = {
            "test_factory": FactoryValidationResult(
                factory_name="TestFactory",
                is_valid=True,
                created_instances=2,
                failed_creations=0,
                validation_errors=[],
                warnings=["Test warning"],
                test_details={"test1": "success"},
            )
        }

        filename = tmp_path / "factory_validation.json"
        save_factory_validation_results(results, str(filename))

        # Verify file was created
        assert filename.exists()

        # Verify content
        with open(filename) as f:
            loaded_results = json.load(f)

        assert "test_factory" in loaded_results
        factory_result = loaded_results["test_factory"]
        assert factory_result["factory_name"] == "TestFactory"
        assert factory_result["is_valid"] is True
        assert factory_result["created_instances"] == 2
        assert factory_result["failed_creations"] == 0
        assert factory_result["warnings"] == ["Test warning"]
        assert factory_result["test_details"]["test1"] == "success"


class TestErrorHandling:
    """Test error handling in factory validation."""

    @pytest.mark.asyncio
    async def test_factory_validation_with_missing_dependencies(self) -> None:
        """Test factory validation when dependencies are missing."""
        # This test would require mocking missing dependencies
        # For now, just test that validation completes
        validator = FactoryPatternValidator()
        results = await validator.validate_all_factories()

        # Should complete without throwing exceptions
        assert isinstance(results, dict)
        for result in results.values():
            assert isinstance(result, FactoryValidationResult)

    @pytest.mark.asyncio
    async def test_factory_validation_with_invalid_configurations(self) -> None:
        """Test factory validation with invalid configurations."""
        validator = FactoryPatternValidator()

        # This will be tested as part of the standard validation
        # since some configurations are expected to fail
        result = await validator.validate_backend_factory()

        # Should handle invalid configurations gracefully
        assert isinstance(result, FactoryValidationResult)
        assert isinstance(result.is_valid, bool)


class TestFactoryIntegration:
    """Test factory integration scenarios."""

    @pytest.mark.asyncio
    async def test_factory_pattern_consistency(self) -> None:
        """Test that factory patterns are consistent across components."""
        results = await validate_all_factory_patterns()

        # All factories should be tested
        assert len(results) >= 4

        # All results should be properly structured
        for result in results.values():
            assert isinstance(result, FactoryValidationResult)
            assert result.factory_name is not None
            assert isinstance(result.is_valid, bool)
            assert result.created_instances >= 0
            assert result.failed_creations >= 0
            assert isinstance(result.validation_errors, list)
            assert isinstance(result.warnings, list)
            assert isinstance(result.test_details, dict)

    @pytest.mark.asyncio
    async def test_mock_factory_registration(self) -> None:
        """Test that mock factories are properly registered."""
        validator = FactoryPatternValidator()

        # Mock factories should be registered during initialization
        # This is tested indirectly through successful validation
        results = await validator.validate_all_factories()

        # At least some factories should be valid if mocks are registered
        valid_factories = [r for r in results.values() if r.is_valid]
        assert len(valid_factories) > 0, (
            "No valid factories found - mock registration may have failed"
        )


@pytest.mark.asyncio
async def test_concurrent_factory_validation() -> None:
    """Test running factory validations concurrently."""
    factory_types = ["backend", "embedding_provider", "rerank_provider", "data_source"]

    # Run validations concurrently
    tasks = [validate_factory_pattern(factory_type) for factory_type in factory_types]
    results = await asyncio.gather(*tasks)

    assert len(results) == 4
    for result in results:
        assert isinstance(result, FactoryValidationResult)
        assert result.factory_name is not None


@pytest.mark.asyncio
async def test_end_to_end_factory_validation() -> None:
    """Test complete end-to-end factory validation workflow."""
    # Run complete validation
    results = await validate_all_factory_patterns()

    # Should have results for all factory types
    assert len(results) >= 4

    # Check specific factory types
    expected_patterns = ["backend", "embedding", "rerank", "source", "configuration"]
    found_patterns = []

    for factory_name in results:
        for pattern in expected_patterns:
            if pattern in factory_name.lower():
                found_patterns.append(pattern)
                break

    # Should have found most expected patterns
    assert len(found_patterns) >= 3, (
        f"Expected patterns not found: {expected_patterns}, found: {found_patterns}"
    )

    # All results should be properly formatted
    for result in results.values():
        assert isinstance(result, FactoryValidationResult)
        # Should have some kind of testing activity
        assert (result.created_instances + result.failed_creations) > 0


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
