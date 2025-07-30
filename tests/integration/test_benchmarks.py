# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Test cases for performance benchmarking framework.

Tests the benchmarking tools and validates performance measurement
functionality across all protocol implementations.
"""

import asyncio

import pytest

from codeweaver.testing import (
    BenchmarkResult,
    BenchmarkSuite,
    MockDataSource,
    MockEmbeddingProvider,
    MockRerankProvider,
    MockVectorBackend,
    print_benchmark_results,
    run_performance_benchmarks,
    save_benchmark_results,
)


@pytest.mark.benchmark
@pytest.mark.performance
@pytest.mark.integration
@pytest.mark.mock_only
class TestBenchmarkResult:
    """Test BenchmarkResult data structure."""

    def test_benchmark_result_creation(self) -> None:
        """Test creating benchmark result."""
        result = BenchmarkResult(
            benchmark_name="test_benchmark",
            operation="test_operation",
            implementation_name="TestImplementation",
            total_duration_ms=1000.0,
            average_duration_ms=100.0,
            median_duration_ms=95.0,
            min_duration_ms=80.0,
            max_duration_ms=120.0,
            std_deviation_ms=15.0,
            operations_per_second=10.0,
            iterations=10,
            batch_size=1,
            test_data_size=1,
        )

        assert result.benchmark_name == "test_benchmark"
        assert result.operation == "test_operation"
        assert result.implementation_name == "TestImplementation"
        assert result.total_duration_ms == 1000.0
        assert result.operations_per_second == 10.0

    def test_benchmark_result_string_representation(self) -> None:
        """Test benchmark result string formatting."""
        result = BenchmarkResult(
            benchmark_name="test_benchmark",
            operation="test_operation",
            implementation_name="TestImplementation",
            total_duration_ms=1000.0,
            average_duration_ms=100.0,
            median_duration_ms=95.0,
            min_duration_ms=80.0,
            max_duration_ms=120.0,
            std_deviation_ms=15.0,
            operations_per_second=10.0,
            iterations=10,
            batch_size=1,
            test_data_size=1,
        )

        str_repr = str(result)
        assert "test_benchmark" in str_repr
        assert "10.00 ops/sec" in str_repr
        assert "100.00ms" in str_repr

    def test_benchmark_result_to_dict(self) -> None:
        """Test benchmark result serialization."""
        result = BenchmarkResult(
            benchmark_name="test_benchmark",
            operation="test_operation",
            implementation_name="TestImplementation",
            total_duration_ms=1000.0,
            average_duration_ms=100.0,
            median_duration_ms=95.0,
            min_duration_ms=80.0,
            max_duration_ms=120.0,
            std_deviation_ms=15.0,
            operations_per_second=10.0,
            iterations=10,
            batch_size=1,
            test_data_size=1,
            metadata={"test": "value"},
        )

        result_data = result.to_dict()

        assert isinstance(result_data, dict)
        assert result_data["benchmark_name"] == "test_benchmark"
        assert result_data["operations_per_second"] == 10.0
        assert result_data["metadata"]["test"] == "value"


@pytest.mark.benchmark
@pytest.mark.performance
@pytest.mark.integration
class TestBenchmarkSuite:
    """Test BenchmarkSuite functionality."""

    def test_benchmark_suite_initialization(self) -> None:
        """Test benchmark suite initialization."""
        # Default configuration
        suite = BenchmarkSuite()
        self._warmup_bench(suite, 3, 10, 60)
        assert suite.measure_resources is False

        # Custom configuration
        suite = BenchmarkSuite(
            warmup_iterations=1, benchmark_iterations=5, timeout_seconds=30, measure_resources=True
        )
        self._warmup_bench(suite, 1, 5, 30)

    # TODO Rename this here and in `test_benchmark_suite_initialization`
    def _warmup_bench(self, suite, arg1, arg2, arg3):
        assert suite.warmup_iterations == arg1
        assert suite.benchmark_iterations == arg2
        assert suite.timeout_seconds == arg3
        # measure_resources may be False if psutil not available

    @pytest.mark.asyncio
    async def test_benchmark_vector_backend(self) -> None:
        """Test benchmarking vector backend."""
        suite = BenchmarkSuite(warmup_iterations=1, benchmark_iterations=3, timeout_seconds=30)

        backend = MockVectorBackend(latency_ms=1.0)
        results = await suite.benchmark_vector_backend(backend)

        assert len(results) > 0

        # Check that we have expected benchmark types
        benchmark_names = [r.benchmark_name for r in results]
        assert any("upsert" in name for name in benchmark_names)
        assert any("search" in name for name in benchmark_names)

        # Check result validity
        for result in results:
            assert result.implementation_name == "MockVectorBackend"
            assert result.operations_per_second >= 0
            assert result.average_duration_ms >= 0
            assert result.iterations > 0

    @pytest.mark.asyncio
    async def test_benchmark_embedding_provider(self) -> None:
        """Test benchmarking embedding provider."""
        suite = BenchmarkSuite(warmup_iterations=1, benchmark_iterations=3, timeout_seconds=30)

        provider = MockEmbeddingProvider(latency_ms=1.0)
        results = await suite.benchmark_embedding_provider(provider)

        assert len(results) > 0

        # Check that we have expected benchmark types
        benchmark_names = [r.benchmark_name for r in results]
        assert any("embed" in name for name in benchmark_names)

        # Check result validity
        for result in results:
            assert result.implementation_name == "MockEmbeddingProvider"
            assert result.operations_per_second >= 0
            assert result.average_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_benchmark_rerank_provider(self) -> None:
        """Test benchmarking rerank provider."""
        suite = BenchmarkSuite(warmup_iterations=1, benchmark_iterations=3, timeout_seconds=30)

        provider = MockRerankProvider(latency_ms=1.0)
        results = await suite.benchmark_rerank_provider(provider)

        assert len(results) > 0

        # Check that we have expected benchmark types
        benchmark_names = [r.benchmark_name for r in results]
        assert any("rerank" in name for name in benchmark_names)

        # Check result validity
        for result in results:
            assert result.implementation_name == "MockRerankProvider"
            assert result.operations_per_second >= 0
            assert result.average_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_benchmark_data_source(self) -> None:
        """Test benchmarking data source."""
        suite = BenchmarkSuite(warmup_iterations=1, benchmark_iterations=3, timeout_seconds=30)

        source = MockDataSource(latency_ms=1.0)
        results = await suite.benchmark_data_source(source)

        assert len(results) > 0

        # Check that we have expected benchmark types
        benchmark_names = [r.benchmark_name for r in results]
        assert any("discover" in name for name in benchmark_names)

        # Check result validity
        for result in results:
            assert result.implementation_name == "MockDataSource"
            assert result.operations_per_second >= 0
            assert result.average_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_benchmark_with_errors(self) -> None:
        """Test benchmarking with error-prone implementations."""
        suite = BenchmarkSuite(warmup_iterations=1, benchmark_iterations=3, timeout_seconds=30)

        # Create backend with high error rate
        backend = MockVectorBackend(error_rate=0.5)
        results = await suite.benchmark_vector_backend(backend)

        # Should still produce results
        assert len(results) > 0

        # Some results may have low success rate
        for result in results:
            assert 0.0 <= result.success_rate <= 1.0
            assert result.error_count >= 0

    @pytest.mark.asyncio
    async def test_benchmark_with_high_latency(self) -> None:
        """Test benchmarking with high latency implementations."""
        suite = BenchmarkSuite(
            warmup_iterations=1,
            benchmark_iterations=2,  # Fewer iterations for speed
            timeout_seconds=30,
        )

        # Create provider with high latency
        provider = MockEmbeddingProvider(latency_ms=100.0)
        results = await suite.benchmark_embedding_provider(provider)

        # Should still produce results
        assert len(results) > 0

        # Results should reflect higher latency
        for result in results:
            # Should have measurable duration
            assert result.average_duration_ms > 50.0  # At least some latency

    def test_generate_test_vectors(self) -> None:
        """Test test vector generation."""
        suite = BenchmarkSuite()

        vectors = suite._generate_test_vectors(count=10, dimension=128)

        assert len(vectors) == 10
        for vector in vectors:
            assert len(vector.vector) == 128
            assert vector.id is not None
            assert vector.payload is not None

            # Check that vectors are normalized
            length = sum(x * x for x in vector.vector) ** 0.5
            assert abs(length - 1.0) < 0.001  # Should be unit length

    def test_generate_test_texts(self) -> None:
        """Test test text generation."""
        suite = BenchmarkSuite()

        texts = suite._generate_test_texts(count=5, length=100)

        assert len(texts) == 5
        for text in texts:
            assert isinstance(text, str)
            assert len(text) > 50  # Should be approximately the right length
            assert len(text.split()) > 1  # Should have multiple words


@pytest.mark.benchmark
@pytest.mark.performance
@pytest.mark.integration
@pytest.mark.slow
class TestBenchmarkScenarios:
    """Test different benchmark scenarios."""

    @pytest.mark.asyncio
    async def test_custom_vector_backend_scenarios(self) -> None:
        """Test vector backend with custom scenarios."""
        suite = BenchmarkSuite(warmup_iterations=1, benchmark_iterations=2, timeout_seconds=30)

        backend = MockVectorBackend(latency_ms=1.0)

        custom_scenarios = [
            {
                "name": "tiny_scale",
                "vector_count": 10,
                "dimension": 64,
                "operations": ["upsert", "search"],
            }
        ]

        results = await suite.benchmark_vector_backend(backend, custom_scenarios)

        assert len(results) > 0
        assert any("tiny_scale" in r.benchmark_name for r in results)

    @pytest.mark.asyncio
    async def test_custom_embedding_scenarios(self) -> None:
        """Test embedding provider with custom scenarios."""
        suite = BenchmarkSuite(warmup_iterations=1, benchmark_iterations=2, timeout_seconds=30)

        provider = MockEmbeddingProvider(latency_ms=1.0)

        custom_scenarios = [
            {
                "name": "small_batch",
                "text_count": 5,
                "text_length": 50,
                "operations": ["documents"],
                "batch_sizes": [1, 5],
            }
        ]

        results = await suite.benchmark_embedding_provider(provider, custom_scenarios)

        assert len(results) > 0
        assert any("small_batch" in r.benchmark_name for r in results)


@pytest.mark.benchmark
@pytest.mark.performance
@pytest.mark.integration
class TestConvenienceFunctions:
    """Test convenience functions for benchmarking."""

    @pytest.mark.asyncio
    async def test_run_performance_benchmarks(self) -> None:
        """Test run_performance_benchmarks function."""
        components = {
            "vector_backend": MockVectorBackend(latency_ms=1.0),
            "embedding_provider": MockEmbeddingProvider(latency_ms=1.0),
            "rerank_provider": MockRerankProvider(latency_ms=1.0),
            "data_source": MockDataSource(latency_ms=1.0),
        }

        benchmark_config = {
            "warmup_iterations": 1,
            "benchmark_iterations": 2,
            "timeout_seconds": 30,
        }

        results = await run_performance_benchmarks(components, benchmark_config)

        assert len(results) == 4
        assert "vector_backend" in results
        assert "embedding_provider" in results
        assert "rerank_provider" in results
        assert "data_source" in results

        # Each component should have benchmark results
        for component_results in results.values():
            assert len(component_results) > 0
            for result in component_results:
                assert isinstance(result, BenchmarkResult)

    @pytest.mark.asyncio
    async def test_run_performance_benchmarks_unknown_component(self) -> None:
        """Test run_performance_benchmarks with unknown component type."""
        components = {"unknown_component": MockVectorBackend()}

        results = await run_performance_benchmarks(components)

        # Should handle unknown component gracefully
        assert len(results) == 0

    def test_print_benchmark_results(self, capsys) -> None:
        """Test printing benchmark results."""
        # Create mock results
        results = {
            "test_component": [
                BenchmarkResult(
                    benchmark_name="test_benchmark",
                    operation="test_operation",
                    implementation_name="TestImplementation",
                    total_duration_ms=1000.0,
                    average_duration_ms=100.0,
                    median_duration_ms=95.0,
                    min_duration_ms=80.0,
                    max_duration_ms=120.0,
                    std_deviation_ms=15.0,
                    operations_per_second=10.0,
                    iterations=10,
                    batch_size=1,
                    test_data_size=1,
                    success_rate=1.0,
                    error_count=0,
                )
            ]
        }

        print_benchmark_results(results)

        captured = capsys.readouterr()
        assert "PERFORMANCE BENCHMARK RESULTS" in captured.out
        assert "test_benchmark" in captured.out
        assert "10.00 ops/sec" in captured.out
        assert "Success Rate: 100.0%" in captured.out

    def test_save_benchmark_results(self, tmp_path) -> None:
        """Test saving benchmark results to file."""
        import json

        results = {
            "test_component": [
                BenchmarkResult(
                    benchmark_name="test_benchmark",
                    operation="test_operation",
                    implementation_name="TestImplementation",
                    total_duration_ms=1000.0,
                    average_duration_ms=100.0,
                    median_duration_ms=95.0,
                    min_duration_ms=80.0,
                    max_duration_ms=120.0,
                    std_deviation_ms=15.0,
                    operations_per_second=10.0,
                    iterations=10,
                    batch_size=1,
                    test_data_size=1,
                )
            ]
        }

        filename = tmp_path / "benchmark_results.json"
        save_benchmark_results(results, str(filename))

        # Verify file was created
        assert filename.exists()

        # Verify content
        with open(filename) as f:
            loaded_results = json.load(f)

        assert "test_component" in loaded_results
        assert len(loaded_results["test_component"]) == 1
        assert loaded_results["test_component"][0]["benchmark_name"] == "test_benchmark"


@pytest.mark.benchmark
@pytest.mark.performance
@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceRegression:
    """Test performance regression detection."""

    @pytest.mark.asyncio
    async def test_consistent_performance_measurements(self) -> None:
        """Test that performance measurements are consistent."""
        suite = BenchmarkSuite(warmup_iterations=1, benchmark_iterations=5, timeout_seconds=30)

        # Use very low latency for consistent results
        provider = MockEmbeddingProvider(latency_ms=0.1)

        # Run benchmark twice
        results1 = await suite.benchmark_embedding_provider(provider)
        results2 = await suite.benchmark_embedding_provider(provider)

        # Results should be similar (within reasonable variance)
        assert len(results1) == len(results2)

        for r1, r2 in zip(results1, results2, strict=False):
            # Allow for some variance (factor of 10)
            ratio = r1.operations_per_second / r2.operations_per_second
            assert 0.1 < ratio < 10.0, f"Performance too different: {ratio}"

    @pytest.mark.asyncio
    async def test_performance_scales_with_latency(self) -> None:
        """Test that performance measurements scale correctly with latency."""
        suite = BenchmarkSuite(warmup_iterations=1, benchmark_iterations=3, timeout_seconds=30)

        # Test with different latencies
        fast_provider = MockEmbeddingProvider(latency_ms=1.0)
        slow_provider = MockEmbeddingProvider(latency_ms=50.0)

        fast_results = await suite.benchmark_embedding_provider(fast_provider)
        slow_results = await suite.benchmark_embedding_provider(slow_provider)

        # Fast provider should have higher ops/sec
        fast_ops = fast_results[0].operations_per_second
        slow_ops = slow_results[0].operations_per_second

        assert fast_ops > slow_ops, "Fast provider should have higher ops/sec"


@pytest.mark.asyncio
async def test_concurrent_benchmarking() -> None:
    """Test running benchmarks concurrently."""
    suite = BenchmarkSuite(warmup_iterations=1, benchmark_iterations=2, timeout_seconds=30)

    # Create multiple components
    components = [
        MockVectorBackend(latency_ms=1.0),
        MockEmbeddingProvider(latency_ms=1.0),
        MockRerankProvider(latency_ms=1.0),
    ]

    # Run benchmarks concurrently
    tasks = [
        suite.benchmark_vector_backend(components[0]),
        suite.benchmark_embedding_provider(components[1]),
        suite.benchmark_rerank_provider(components[2]),
    ]

    results = await asyncio.gather(*tasks)

    assert len(results) == 3
    for result_list in results:
        assert len(result_list) > 0
        for result in result_list:
            assert isinstance(result, BenchmarkResult)


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
