# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Test cases for protocol compliance validation.

Comprehensive tests that ensure all protocol implementations conform to their
interfaces and behave correctly under various conditions.
"""

import asyncio

import pytest

from codeweaver.testing.mocks import (
    MockDataSource,
    MockEmbeddingProvider,
    MockHybridSearchBackend,
    MockRerankProvider,
    MockVectorBackend,
)
from codeweaver.testing.protocol_compliance import (
    ProtocolComplianceValidator,
    validate_data_source_protocol,
    validate_embedding_provider_protocol,
    validate_hybrid_search_backend_protocol,
    validate_rerank_provider_protocol,
    validate_vector_backend_protocol,
)


class TestProtocolComplianceValidator:
    """Test the protocol compliance validator itself."""

    def test_validator_initialization(self) -> None:
        """Test validator initialization with different configurations."""
        # Default configuration
        validator = ProtocolComplianceValidator()
        assert validator.strict_mode is True
        assert validator.performance_benchmarks is True

        # Custom configuration
        validator = ProtocolComplianceValidator(strict_mode=False, performance_benchmarks=False)
        assert validator.strict_mode is False
        assert validator.performance_benchmarks is False

    @pytest.mark.asyncio
    async def test_validate_all_protocols(self) -> None:
        """Test validating multiple protocols at once."""
        validator = ProtocolComplianceValidator()

        implementations = {
            "vector_backend": MockVectorBackend(),
            "embedding_provider": MockEmbeddingProvider(),
            "rerank_provider": MockRerankProvider(),
            "data_source": MockDataSource(),
        }

        results = await validator.validate_all_protocols(implementations)

        assert len(results) == 4
        assert "vector_backend" in results
        assert "embedding_provider" in results
        assert "rerank_provider" in results
        assert "data_source" in results

        # All mock implementations should be compliant
        for protocol_name, result in results.items():
            assert result.is_compliant, f"{protocol_name} should be compliant"
            assert result.passed_tests > 0, f"{protocol_name} should have passed tests"

    @pytest.mark.asyncio
    async def test_validate_unknown_protocol(self) -> None:
        """Test handling of unknown protocol types."""
        validator = ProtocolComplianceValidator()

        implementations = {"unknown_protocol": MockVectorBackend()}

        results = await validator.validate_all_protocols(implementations)

        assert len(results) == 1
        assert "unknown_protocol" in results
        assert not results["unknown_protocol"].is_compliant
        assert "Unknown protocol" in results["unknown_protocol"].validation_errors[0]


class TestVectorBackendCompliance:
    """Test VectorBackend protocol compliance."""

    @pytest.mark.asyncio
    async def test_mock_vector_backend_compliance(self) -> None:
        """Test that MockVectorBackend is compliant."""
        backend = MockVectorBackend()
        result = await validate_vector_backend_protocol(backend)

        assert result.is_compliant
        assert result.passed_tests > 0
        assert len(result.validation_errors) == 0
        assert result.protocol_name == "VectorBackend"
        assert result.implementation_name == "MockVectorBackend"

    @pytest.mark.asyncio
    async def test_vector_backend_with_errors(self) -> None:
        """Test vector backend with simulated errors."""
        backend = MockVectorBackend(error_rate=1.0)  # Always fail
        result = await validate_vector_backend_protocol(backend)

        # Should still be protocol compliant (has all methods)
        # but functionality tests may fail
        assert result.protocol_name == "VectorBackend"
        assert result.implementation_name == "MockVectorBackend"

    @pytest.mark.asyncio
    async def test_vector_backend_performance_metrics(self) -> None:
        """Test that performance metrics are collected."""
        backend = MockVectorBackend(latency_ms=50.0)
        validator = ProtocolComplianceValidator(performance_benchmarks=True)
        result = await validator.validate_vector_backend(backend)

        assert len(result.performance_metrics) > 0
        # Check that some operations were timed
        assert any("create_collection" in metric for metric in result.performance_metrics)


class TestHybridSearchBackendCompliance:
    """Test HybridSearchBackend protocol compliance."""

    @pytest.mark.asyncio
    async def test_mock_hybrid_search_backend_compliance(self) -> None:
        """Test that MockHybridSearchBackend is compliant."""
        backend = MockHybridSearchBackend()
        result = await validate_hybrid_search_backend_protocol(backend)

        assert result.is_compliant
        assert result.passed_tests > 0
        assert len(result.validation_errors) == 0
        assert result.protocol_name == "HybridSearchBackend"
        assert result.implementation_name == "MockHybridSearchBackend"

    @pytest.mark.asyncio
    async def test_hybrid_search_extends_vector_backend(self) -> None:
        """Test that hybrid search backend extends vector backend capabilities."""
        backend = MockHybridSearchBackend()
        result = await validate_hybrid_search_backend_protocol(backend)

        # Should include both VectorBackend and HybridSearchBackend tests
        assert result.total_tests > 10  # More tests than just VectorBackend
        assert result.is_compliant


class TestEmbeddingProviderCompliance:
    """Test EmbeddingProvider protocol compliance."""

    @pytest.mark.asyncio
    async def test_mock_embedding_provider_compliance(self) -> None:
        """Test that MockEmbeddingProvider is compliant."""
        provider = MockEmbeddingProvider()
        result = await validate_embedding_provider_protocol(provider)

        assert result.is_compliant
        assert result.passed_tests > 0
        assert len(result.validation_errors) == 0
        assert result.protocol_name == "EmbeddingProvider"
        assert result.implementation_name == "MockEmbeddingProvider"

    @pytest.mark.asyncio
    async def test_embedding_provider_properties(self) -> None:
        """Test embedding provider property validation."""
        provider = MockEmbeddingProvider(
            provider_name="test_provider", model_name="test_model", dimension=256
        )
        result = await validate_embedding_provider_protocol(provider)

        assert result.is_compliant
        assert provider.provider_name == "test_provider"
        assert provider.model_name == "test_model"
        assert provider.dimension == 256

    @pytest.mark.asyncio
    async def test_embedding_provider_functionality(self) -> None:
        """Test embedding provider actual functionality."""
        provider = MockEmbeddingProvider(dimension=128)

        # Test document embeddings
        texts = ["Hello world", "Test document"]
        embeddings = await provider.embed_documents(texts)

        assert len(embeddings) == len(texts)
        assert all(len(emb) == 128 for emb in embeddings)

        # Test query embedding
        query_embedding = await provider.embed_query("test query")
        assert len(query_embedding) == 128

        # Test provider info
        info = provider.get_provider_info()
        assert info.name == provider.provider_name
        assert 128 in info.native_dimensions.values()


class TestRerankProviderCompliance:
    """Test RerankProvider protocol compliance."""

    @pytest.mark.asyncio
    async def test_mock_rerank_provider_compliance(self) -> None:
        """Test that MockRerankProvider is compliant."""
        provider = MockRerankProvider()
        result = await validate_rerank_provider_protocol(provider)

        assert result.is_compliant
        assert result.passed_tests > 0
        assert len(result.validation_errors) == 0
        assert result.protocol_name == "RerankProvider"
        assert result.implementation_name == "MockRerankProvider"

    @pytest.mark.asyncio
    async def test_rerank_provider_functionality(self) -> None:
        """Test rerank provider actual functionality."""
        provider = MockRerankProvider()

        query = "machine learning"
        documents = [
            "Introduction to machine learning algorithms",
            "Python programming tutorial",
            "Deep learning with neural networks",
        ]

        results = await provider.rerank(query, documents, top_k=2)

        assert len(results) <= 2
        assert all(0 <= r.relevance_score <= 1 for r in results)
        assert all(0 <= r.index < len(documents) for r in results)

        # Check that results are sorted by relevance
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)


class TestDataSourceCompliance:
    """Test DataSource protocol compliance."""

    @pytest.mark.asyncio
    async def test_mock_data_source_compliance(self) -> None:
        """Test that MockDataSource is compliant."""
        source = MockDataSource()
        result = await validate_data_source_protocol(source)

        assert result.is_compliant
        assert result.passed_tests > 0
        assert len(result.validation_errors) == 0
        assert result.protocol_name == "DataSource"
        assert result.implementation_name == "MockDataSource"

    @pytest.mark.asyncio
    async def test_data_source_capabilities(self) -> None:
        """Test data source capabilities."""
        source = MockDataSource()
        capabilities = source.get_capabilities()

        assert len(capabilities) > 0
        # Check that all capabilities are valid SourceCapability enum values
        from codeweaver.types import SourceCapability

        assert all(isinstance(cap, SourceCapability) for cap in capabilities)

    @pytest.mark.asyncio
    async def test_data_source_functionality(self) -> None:
        """Test data source actual functionality."""
        source = MockDataSource()

        test_config = {"enabled": True, "priority": 1, "source_id": "test_source"}

        # Test source validation
        is_valid = await source.validate_source(test_config)
        assert is_valid is True

        # Test content discovery
        content_items = await source.discover_content(test_config)
        assert len(content_items) > 0

        # Test content reading
        if content_items:
            content = await source.read_content(content_items[0])
            assert isinstance(content, str)
            assert len(content) > 0

            # Test metadata extraction
            metadata = await source.get_content_metadata(content_items[0])
            assert isinstance(metadata, dict)
            assert "source_type" in metadata


class TestComplianceResultFormatting:
    """Test compliance result formatting and reporting."""

    @pytest.mark.asyncio
    async def test_compliance_result_string_representation(self) -> None:
        """Test compliance result string formatting."""
        backend = MockVectorBackend()
        result = await validate_vector_backend_protocol(backend)

        str_repr = str(result)
        assert "VectorBackend" in str_repr
        assert "MockVectorBackend" in str_repr
        assert ("✅ PASS" in str_repr) if result.is_compliant else ("❌ FAIL" in str_repr)
        assert f"{result.passed_tests}/{result.total_tests}" in str_repr

    @pytest.mark.asyncio
    async def test_compliance_result_detailed_report(self) -> None:
        """Test detailed compliance report generation."""
        backend = MockVectorBackend()
        result = await validate_vector_backend_protocol(backend)

        detailed_report = result.get_detailed_report()
        assert "Protocol Compliance Report" in detailed_report
        assert "VectorBackend" in detailed_report
        assert "MockVectorBackend" in detailed_report
        assert "Tests Passed:" in detailed_report

        if result.performance_metrics:
            assert "Performance Metrics:" in detailed_report


class TestErrorHandling:
    """Test error handling in protocol compliance validation."""

    @pytest.mark.asyncio
    async def test_timeout_handling(self) -> None:
        """Test handling of operation timeouts."""
        # Create backend with very high latency
        backend = MockVectorBackend(latency_ms=1000.0)
        validator = ProtocolComplianceValidator()

        # This test might timeout on some operations, but should handle gracefully
        result = await validator.validate_vector_backend(backend)

        # Should still complete validation
        assert result.protocol_name == "VectorBackend"
        assert result.implementation_name == "MockVectorBackend"

    @pytest.mark.asyncio
    async def test_high_error_rate_handling(self) -> None:
        """Test handling of implementations with high error rates."""
        # Create backend that always fails operations
        backend = MockVectorBackend(error_rate=1.0)
        result = await validate_vector_backend_protocol(backend)

        # Protocol interface should still be compliant
        # but functionality tests may fail
        assert result.protocol_name == "VectorBackend"
        # Some tests may fail due to errors, but interface validation should pass


@pytest.mark.asyncio
async def test_concurrent_compliance_validation() -> None:
    """Test running multiple compliance validations concurrently."""
    # Create multiple implementations
    implementations = [
        MockVectorBackend(),
        MockEmbeddingProvider(),
        MockRerankProvider(),
        MockDataSource(),
    ]

    # Run validations concurrently
    tasks = [
        validate_vector_backend_protocol(implementations[0]),
        validate_embedding_provider_protocol(implementations[1]),
        validate_rerank_provider_protocol(implementations[2]),
        validate_data_source_protocol(implementations[3]),
    ]

    results = await asyncio.gather(*tasks)

    assert len(results) == 4
    assert all(result.is_compliant for result in results)
    assert all(result.passed_tests > 0 for result in results)


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
