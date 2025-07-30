# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Integration tests for CodeWeaver testing framework.

Comprehensive integration tests that validate end-to-end workflows,
component combinations, and configuration scenarios.
"""

import asyncio
import os

import pytest

from codeweaver.testing import (
    IntegrationTestSuite,
    MockDataSource,
    MockEmbeddingProvider,
    MockHybridSearchBackend,
    MockRerankProvider,
    MockVectorBackend,
    TestConfiguration,
    create_test_configuration,
    run_integration_tests,
    run_quick_integration_test,
)


class TestTestConfiguration:
    """Test the TestConfiguration class."""

    def test_default_configuration(self) -> None:
        """Test default test configuration values."""
        config = TestConfiguration()

        assert config.backend_type == "mock"
        assert config.embedding_provider == "mock"
        assert config.rerank_provider == "mock"
        assert config.data_source_type == "mock"
        assert config.run_compliance_tests is True
        assert config.run_performance_tests is True
        assert config.run_workflow_tests is True
        assert config.test_timeout_seconds == 60
        assert config.mock_latency_ms == 10.0
        assert config.mock_error_rate == 0.0
        assert len(config.test_documents) > 0
        assert len(config.test_queries) > 0

    def test_custom_configuration(self) -> None:
        """Test creating configuration with custom values."""
        config = TestConfiguration(
            backend_type="qdrant",
            embedding_provider="voyage-ai",
            rerank_provider=None,
            run_performance_tests=False,
            mock_latency_ms=50.0,
            test_timeout_seconds=120,
        )

        assert config.backend_type == "qdrant"
        assert config.embedding_provider == "voyage-ai"
        assert config.rerank_provider is None
        assert config.run_performance_tests is False
        assert config.mock_latency_ms == 50.0
        assert config.test_timeout_seconds == 120

    def test_create_test_configuration_helper(self) -> None:
        """Test the create_test_configuration helper function."""
        config = create_test_configuration(
            backend_type="pinecone", embedding_provider="openai", run_compliance_tests=False
        )

        assert config.backend_type == "pinecone"
        assert config.embedding_provider == "openai"
        assert config.run_compliance_tests is False


class TestIntegrationTestSuite:
    """Test the IntegrationTestSuite class."""

    @pytest.mark.asyncio
    async def test_suite_initialization(self) -> None:
        """Test integration test suite initialization."""
        config = create_test_configuration()
        suite = IntegrationTestSuite(config)

        assert suite.config == config
        assert suite.validator is not None
        assert suite.backend is None  # Not initialized yet
        assert suite.embedding_provider is None
        assert suite.rerank_provider is None
        assert suite.data_source is None
        assert suite.test_collection == "integration_test_collection"

    @pytest.mark.asyncio
    async def test_setup_test_environment_mock_components(self) -> None:
        """Test setting up test environment with mock components."""
        config = create_test_configuration(
            backend_type="mock",
            embedding_provider="mock",
            rerank_provider="mock",
            data_source_type="mock",
        )
        suite = IntegrationTestSuite(config)

        await suite._setup_test_environment()

        try:
            # Verify components are created
            assert suite.backend is not None
            assert isinstance(suite.backend, MockVectorBackend | MockHybridSearchBackend)
            assert suite.embedding_provider is not None
            assert isinstance(suite.embedding_provider, MockEmbeddingProvider)
            assert suite.rerank_provider is not None
            assert isinstance(suite.rerank_provider, MockRerankProvider)
            assert suite.data_source is not None
            assert isinstance(suite.data_source, MockDataSource)
            assert suite.temp_dir is not None
            assert suite.temp_dir.exists()

        finally:
            await suite._cleanup_test_environment()

    @pytest.mark.asyncio
    async def test_setup_test_environment_no_rerank(self) -> None:
        """Test setting up test environment without rerank provider."""
        config = create_test_configuration(rerank_provider=None)
        suite = IntegrationTestSuite(config)

        await suite._setup_test_environment()

        try:
            assert suite.backend is not None
            assert isinstance(suite.backend, MockVectorBackend)  # Not hybrid without rerank
            assert suite.embedding_provider is not None
            assert suite.rerank_provider is None
            assert suite.data_source is not None

        finally:
            await suite._cleanup_test_environment()

    @pytest.mark.asyncio
    async def test_cleanup_test_environment(self) -> None:
        """Test cleaning up test environment."""
        config = create_test_configuration()
        suite = IntegrationTestSuite(config)

        await suite._setup_test_environment()
        temp_dir = suite.temp_dir

        assert temp_dir is not None
        assert temp_dir.exists()

        await suite._cleanup_test_environment()

        # Temporary directory should be cleaned up
        assert not temp_dir.exists()


class TestWorkflowTests:
    """Test individual workflow testing methods."""

    @pytest.mark.asyncio
    async def test_embedding_workflow(self) -> None:
        """Test embedding provider workflow."""
        config = create_test_configuration()
        suite = IntegrationTestSuite(config)

        await suite._setup_test_environment()

        try:
            success = await suite._test_embedding_workflow()
            assert success is True

        finally:
            await suite._cleanup_test_environment()

    @pytest.mark.asyncio
    async def test_search_workflow(self) -> None:
        """Test vector backend search workflow."""
        config = create_test_configuration()
        suite = IntegrationTestSuite(config)

        await suite._setup_test_environment()

        try:
            success = await suite._test_search_workflow()
            assert success is True

        finally:
            await suite._cleanup_test_environment()

    @pytest.mark.asyncio
    async def test_rerank_workflow(self) -> None:
        """Test rerank provider workflow."""
        config = create_test_configuration(rerank_provider="mock")
        suite = IntegrationTestSuite(config)

        await suite._setup_test_environment()

        try:
            success = await suite._test_rerank_workflow()
            assert success is True

        finally:
            await suite._cleanup_test_environment()

    @pytest.mark.asyncio
    async def test_data_source_workflow(self) -> None:
        """Test data source workflow."""
        config = create_test_configuration()
        suite = IntegrationTestSuite(config)

        await suite._setup_test_environment()

        try:
            success = await suite._test_data_source_workflow()
            assert success is True

        finally:
            await suite._cleanup_test_environment()

    @pytest.mark.asyncio
    async def test_hybrid_search_workflow(self) -> None:
        """Test hybrid search workflow."""
        config = create_test_configuration(rerank_provider="mock")  # Enables hybrid backend
        suite = IntegrationTestSuite(config)

        await suite._setup_test_environment()

        try:
            success = await suite._test_hybrid_search_workflow()
            assert success is True

        finally:
            await suite._cleanup_test_environment()

    @pytest.mark.asyncio
    async def test_complete_pipeline(self) -> None:
        """Test complete indexing and search pipeline."""
        config = create_test_configuration()
        suite = IntegrationTestSuite(config)

        await suite._setup_test_environment()

        try:
            success = await suite._test_complete_pipeline()
            assert success is True

        finally:
            await suite._cleanup_test_environment()


class TestIntegrationTestExecution:
    """Test full integration test execution."""

    @pytest.mark.asyncio
    async def test_run_all_tests_success(self) -> None:
        """Test running all integration tests with successful outcome."""
        config = create_test_configuration(
            run_compliance_tests=True,
            run_performance_tests=False,  # Skip for speed
            run_workflow_tests=True,
            mock_latency_ms=1.0,  # Fast execution
        )
        suite = IntegrationTestSuite(config)

        result = await suite.run_all_tests()

        assert result.success is True
        assert result.duration_seconds > 0
        assert len(result.compliance_results) > 0
        assert len(result.workflow_results) > 0
        assert len(result.errors) == 0

        # Check that all compliance tests passed
        for compliance_result in result.compliance_results.values():
            assert compliance_result.is_compliant

        # Check that all workflow tests passed
        for workflow_passed in result.workflow_results.values():
            assert workflow_passed is True

    @pytest.mark.asyncio
    async def test_run_all_tests_with_errors(self) -> None:
        """Test running integration tests with simulated errors."""
        config = create_test_configuration(
            mock_error_rate=0.5,  # 50% error rate
            run_performance_tests=False,
        )
        suite = IntegrationTestSuite(config)

        result = await suite.run_all_tests()

        # Should complete but may have failures
        assert result.duration_seconds > 0
        assert len(result.compliance_results) > 0

        # Some tests may fail due to high error rate
        # But the test framework should handle this gracefully

    @pytest.mark.asyncio
    async def test_run_compliance_tests_only(self) -> None:
        """Test running only compliance tests."""
        config = create_test_configuration(
            run_compliance_tests=True, run_performance_tests=False, run_workflow_tests=False
        )
        suite = IntegrationTestSuite(config)

        result = await suite.run_all_tests()

        assert result.success is True
        assert len(result.compliance_results) > 0
        assert len(result.workflow_results) == 0  # No workflow tests run

    @pytest.mark.asyncio
    async def test_run_workflow_tests_only(self) -> None:
        """Test running only workflow tests."""
        config = create_test_configuration(
            run_compliance_tests=False, run_performance_tests=False, run_workflow_tests=True
        )
        suite = IntegrationTestSuite(config)

        result = await suite.run_all_tests()

        assert result.success is True
        assert len(result.compliance_results) == 0  # No compliance tests run
        assert len(result.workflow_results) > 0


class TestConvenienceFunctions:
    """Test convenience functions for integration testing."""

    @pytest.mark.asyncio
    async def test_run_integration_tests_default(self) -> None:
        """Test run_integration_tests with default configuration."""
        result = await run_integration_tests()

        assert result.success is True
        assert result.test_name.startswith("Integration Test Suite")
        assert result.duration_seconds > 0
        assert len(result.compliance_results) > 0
        assert len(result.workflow_results) > 0

    @pytest.mark.asyncio
    async def test_run_integration_tests_custom_config(self) -> None:
        """Test run_integration_tests with custom configuration."""
        config = create_test_configuration(
            backend_type="mock",
            embedding_provider="mock",
            run_performance_tests=False,
            mock_latency_ms=1.0,
        )

        result = await run_integration_tests(config)

        assert result.success is True
        assert "mock" in result.test_name.lower()

    @pytest.mark.asyncio
    async def test_run_quick_integration_test(self) -> None:
        """Test quick integration test function."""
        result = await run_quick_integration_test()

        assert result.success is True
        assert result.duration_seconds < 30  # Should be quick
        assert len(result.compliance_results) > 0
        assert len(result.workflow_results) > 0


class TestIntegrationTestResult:
    """Test IntegrationTestResult class and formatting."""

    @pytest.mark.asyncio
    async def test_result_string_representation(self) -> None:
        """Test result string formatting."""
        result = await run_quick_integration_test()

        str_repr = str(result)
        assert result.test_name in str_repr
        assert ("✅ PASS" in str_repr) if result.success else ("❌ FAIL" in str_repr)
        assert f"{result.duration_seconds:.2f}s" in str_repr

    @pytest.mark.asyncio
    async def test_result_detailed_report(self) -> None:
        """Test detailed result report generation."""
        result = await run_quick_integration_test()

        detailed_report = result.get_detailed_report()
        assert "Integration Test Report" in detailed_report
        assert result.test_name in detailed_report
        assert "Status:" in detailed_report
        assert "Duration:" in detailed_report

        if result.compliance_results:
            assert "Protocol Compliance Results:" in detailed_report

        if result.workflow_results:
            assert "Workflow Test Results:" in detailed_report


class TestErrorHandling:
    """Test error handling in integration tests."""

    @pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip timeout test in CI")
    @pytest.mark.asyncio
    async def test_timeout_handling(self) -> None:
        """Test handling of test timeouts."""
        config = create_test_configuration(
            test_timeout_seconds=1,  # Very short timeout
            mock_latency_ms=2000.0,  # High latency
        )

        # This test may timeout, but should handle gracefully
        result = await run_integration_tests(config)

        # Should complete even with timeouts
        assert result.test_name is not None
        assert result.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_component_creation_failure(self) -> None:
        """Test handling of component creation failures."""
        config = create_test_configuration(backend_type="nonexistent_backend")

        result = await run_integration_tests(config)

        # Should handle gracefully
        assert not result.success
        assert len(result.errors) > 0


@pytest.mark.asyncio
async def test_concurrent_integration_tests() -> None:
    """Test running multiple integration tests concurrently."""
    configs = [
        create_test_configuration(run_performance_tests=False, mock_latency_ms=1.0),
        create_test_configuration(run_compliance_tests=False, mock_latency_ms=1.0),
        create_test_configuration(rerank_provider=None, mock_latency_ms=1.0),
    ]

    # Run tests concurrently
    tasks = [run_integration_tests(config) for config in configs]
    results = await asyncio.gather(*tasks)

    assert len(results) == 3

    # All tests should complete
    for result in results:
        assert result.duration_seconds > 0
        assert result.test_name is not None


@pytest.mark.asyncio
async def test_integration_with_different_component_combinations() -> None:
    """Test integration with different component combinations."""
    test_cases = [
        # Basic vector backend
        {"backend_type": "mock", "rerank_provider": None},
        # Hybrid search backend
        {"backend_type": "mock", "rerank_provider": "mock"},
        # Different embedding provider
        {"embedding_provider": "mock", "rerank_provider": None},
    ]

    for case in test_cases:
        config = create_test_configuration(run_performance_tests=False, mock_latency_ms=1.0, **case)

        result = await run_integration_tests(config)

        # Each combination should work
        assert result.success is True, f"Failed for case: {case}"
        assert len(result.compliance_results) > 0
        assert len(result.workflow_results) > 0


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
