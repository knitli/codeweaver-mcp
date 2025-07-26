# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Performance benchmarking tools for CodeWeaver protocols.

Comprehensive benchmarking framework that measures performance characteristics,
identifies bottlenecks, and validates performance requirements across all
protocol implementations.
"""

import asyncio
import contextlib
import logging
import statistics
import time

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from codeweaver.backends.base import DistanceMetric, VectorBackend, VectorPoint
from codeweaver.providers.base import EmbeddingProvider, RerankProvider
from codeweaver.sources.base import DataSource, SourceConfig


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""

    benchmark_name: str
    operation: str
    implementation_name: str

    # Performance metrics
    total_duration_ms: float
    average_duration_ms: float
    median_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    std_deviation_ms: float

    # Throughput metrics
    operations_per_second: float
    items_per_second: float | None = None

    # Test configuration
    iterations: int = 0
    batch_size: int = 1
    test_data_size: int = 0

    # Resource metrics
    memory_usage_mb: float | None = None
    cpu_usage_percent: float | None = None

    # Quality metrics
    success_rate: float = 1.0
    error_count: int = 0

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.benchmark_name}: {self.operations_per_second:.2f} ops/sec "
            f"(avg: {self.average_duration_ms:.2f}ms)"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert benchmark result to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "operation": self.operation,
            "implementation_name": self.implementation_name,
            "total_duration_ms": self.total_duration_ms,
            "average_duration_ms": self.average_duration_ms,
            "median_duration_ms": self.median_duration_ms,
            "min_duration_ms": self.min_duration_ms,
            "max_duration_ms": self.max_duration_ms,
            "std_deviation_ms": self.std_deviation_ms,
            "operations_per_second": self.operations_per_second,
            "items_per_second": self.items_per_second,
            "iterations": self.iterations,
            "batch_size": self.batch_size,
            "test_data_size": self.test_data_size,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "success_rate": self.success_rate,
            "error_count": self.error_count,
            "metadata": self.metadata,
        }


class BenchmarkSuite:
    """Comprehensive benchmarking suite for CodeWeaver protocols."""

    def __init__(
        self,
        warmup_iterations: int = 3,
        benchmark_iterations: int = 10,
        timeout_seconds: int = 60,
        measure_resources: bool = False,
    ):
        """Initialize benchmark suite.

        Args:
            warmup_iterations: Number of warmup iterations before benchmarking
            benchmark_iterations: Number of benchmark iterations to run
            timeout_seconds: Timeout for individual benchmark operations
            measure_resources: Whether to measure resource usage (requires psutil)
        """
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.timeout_seconds = timeout_seconds
        self.measure_resources = measure_resources

        # Resource monitoring
        self._process = None
        if measure_resources:
            try:
                import psutil

                self._process = psutil.Process()
            except ImportError:
                logger.warning("psutil not available, resource monitoring disabled")
                self.measure_resources = False

    async def benchmark_vector_backend(
        self, backend: VectorBackend, test_scenarios: list[dict[str, Any]] | None = None
    ) -> list[BenchmarkResult]:
        """Benchmark vector backend performance."""
        if test_scenarios is None:
            test_scenarios = self._get_default_vector_backend_scenarios()

        results = []

        for i, scenario in enumerate(test_scenarios):
            collection_name = f"benchmark_collection_{i}"
            dimension = scenario.get("dimension", 128)

            try:
                # Setup test collection with scenario-specific dimension
                await backend.create_collection(
                    name=collection_name, dimension=dimension, distance_metric=DistanceMetric.COSINE
                )

                scenario_results = await self._run_vector_backend_scenario(
                    backend, collection_name, scenario
                )
                results.extend(scenario_results)

            finally:
                # Cleanup scenario collection
                with contextlib.suppress(Exception):
                    await backend.delete_collection(collection_name)

        return results

    async def benchmark_embedding_provider(
        self, provider: EmbeddingProvider, test_scenarios: list[dict[str, Any]] | None = None
    ) -> list[BenchmarkResult]:
        """Benchmark embedding provider performance."""
        if test_scenarios is None:
            test_scenarios = self._get_default_embedding_scenarios()

        results = []

        for scenario in test_scenarios:
            scenario_results = await self._run_embedding_scenario(provider, scenario)
            results.extend(scenario_results)

        return results

    async def benchmark_rerank_provider(
        self, provider: RerankProvider, test_scenarios: list[dict[str, Any]] | None = None
    ) -> list[BenchmarkResult]:
        """Benchmark rerank provider performance."""
        if test_scenarios is None:
            test_scenarios = self._get_default_rerank_scenarios()

        results = []

        for scenario in test_scenarios:
            scenario_results = await self._run_rerank_scenario(provider, scenario)
            results.extend(scenario_results)

        return results

    async def benchmark_data_source(
        self, source: DataSource, test_scenarios: list[dict[str, Any]] | None = None
    ) -> list[BenchmarkResult]:
        """Benchmark data source performance."""
        if test_scenarios is None:
            test_scenarios = self._get_default_data_source_scenarios()

        results = []

        for scenario in test_scenarios:
            scenario_results = await self._run_data_source_scenario(source, scenario)
            results.extend(scenario_results)

        return results

    async def _run_vector_backend_scenario(
        self, backend: VectorBackend, collection_name: str, scenario: dict[str, Any]
    ) -> list[BenchmarkResult]:
        """Run a specific vector backend benchmark scenario."""
        results = []
        scenario_name = scenario["name"]
        operations = scenario["operations"]

        # Generate test data
        test_vectors = self._generate_test_vectors(
            count=scenario.get("vector_count", 100), dimension=scenario.get("dimension", 128)
        )

        for operation in operations:
            if operation == "upsert":
                result = await self._benchmark_operation(
                    f"{scenario_name}_upsert",
                    "upsert_vectors",
                    backend,
                    lambda b: b.upsert_vectors(collection_name, test_vectors),
                    batch_size=len(test_vectors),
                )
                results.append(result)

            elif operation == "search":
                # First upsert some data for search
                try:
                    await backend.upsert_vectors(collection_name, test_vectors[:50])

                    query_vector = test_vectors[0].vector
                    result = await self._benchmark_operation(
                        f"{scenario_name}_search",
                        "search_vectors",
                        backend,
                        lambda b, qv=query_vector: b.search_vectors(collection_name, qv, limit=10),
                    )
                    results.append(result)
                except Exception as e:
                    logger.debug("Setup failed for search benchmark: %s", e)
                    # Skip search benchmark if setup fails

            elif operation == "batch_search":
                # Benchmark multiple searches
                try:
                    await backend.upsert_vectors(collection_name, test_vectors[:50])

                    query_vectors = [v.vector for v in test_vectors[:10]]

                    async def batch_search_func(
                        b: VectorBackend, vectors=query_vectors
                    ) -> list[VectorPoint]:
                        tasks = [b.search_vectors(collection_name, qv, limit=5) for qv in vectors]
                        return await asyncio.gather(*tasks)

                    result = await self._benchmark_operation(
                        f"{scenario_name}_batch_search",
                        "batch_search_vectors",
                        backend,
                        batch_search_func,
                        batch_size=len(query_vectors),
                    )
                    results.append(result)
                except Exception as e:
                    logger.debug("Setup failed for batch_search benchmark: %s", e)
                    # Skip batch_search benchmark if setup fails

        return results

    async def _run_embedding_scenario(
        self, provider: EmbeddingProvider, scenario: dict[str, Any]
    ) -> list[BenchmarkResult]:
        """Run a specific embedding provider benchmark scenario."""
        results = []
        scenario_name = scenario["name"]

        # Generate test texts
        test_texts = self._generate_test_texts(
            count=scenario.get("text_count", 50), length=scenario.get("text_length", 100)
        )

        # Benchmark document embedding
        if "documents" in scenario.get("operations", []):
            batch_sizes = scenario.get("batch_sizes", [1, 5, 10, 20])

            for batch_size in batch_sizes:
                if batch_size <= len(test_texts):
                    batch = test_texts[:batch_size]
                    result = await self._benchmark_operation(
                        f"{scenario_name}_embed_documents_batch_{batch_size}",
                        "embed_documents",
                        provider,
                        lambda p, batch_data=batch: p.embed_documents(batch_data),
                        batch_size=batch_size,
                    )
                    results.append(result)

        # Benchmark query embedding
        if "queries" in scenario.get("operations", []):
            result = await self._benchmark_operation(
                f"{scenario_name}_embed_query",
                "embed_query",
                provider,
                lambda p: p.embed_query(test_texts[0]),
            )
            results.append(result)

        return results

    async def _run_rerank_scenario(
        self, provider: RerankProvider, scenario: dict[str, Any]
    ) -> list[BenchmarkResult]:
        """Run a specific rerank provider benchmark scenario."""
        results = []
        scenario_name = scenario["name"]

        # Generate test data
        query = "test query for reranking performance"
        documents = self._generate_test_texts(
            count=scenario.get("document_count", 20), length=scenario.get("text_length", 200)
        )

        # Benchmark reranking with different document counts
        doc_counts = scenario.get("document_counts", [5, 10, 20])

        for doc_count in doc_counts:
            if doc_count <= len(documents):
                docs_subset = documents[:doc_count]
                result = await self._benchmark_operation(
                    f"{scenario_name}_rerank_{doc_count}_docs",
                    "rerank",
                    provider,
                    lambda p, docs=docs_subset, count=doc_count: p.rerank(
                        query, docs, top_k=count // 2
                    ),
                    batch_size=doc_count,
                )
                results.append(result)

        return results

    async def _run_data_source_scenario(
        self, source: DataSource, scenario: dict[str, Any]
    ) -> list[BenchmarkResult]:
        """Run a specific data source benchmark scenario."""
        results = []
        scenario_name = scenario["name"]

        # Test configuration
        test_config: SourceConfig = {
            "enabled": True,
            "priority": 1,
            "source_id": "benchmark_source",
            **scenario.get("config", {}),
        }

        # Benchmark content discovery
        if "discovery" in scenario.get("operations", []):
            result = await self._benchmark_operation(
                f"{scenario_name}_discover_content",
                "discover_content",
                source,
                lambda s: s.discover_content(test_config),
            )
            results.append(result)

            # Get content items for reading benchmark
            content_items = await source.discover_content(test_config)

            # Benchmark content reading
            if "reading" in scenario.get("operations", []) and content_items:
                # Test reading first few items
                items_to_read = content_items[: min(5, len(content_items))]

                async def read_multiple_func(s: DataSource) -> list[Any]:
                    tasks = [s.read_content(item) for item in items_to_read]
                    return await asyncio.gather(*tasks, return_exceptions=True)

                result = await self._benchmark_operation(
                    f"{scenario_name}_read_content",
                    "read_content",
                    source,
                    read_multiple_func,
                    batch_size=len(items_to_read),
                )
                results.append(result)

        return results

    async def _benchmark_operation(
        self,
        benchmark_name: str,
        operation: str,
        implementation: Any,
        operation_func: Callable,
        batch_size: int = 1,
    ) -> BenchmarkResult:  # sourcery skip: low-code-quality
        """Benchmark a specific operation."""
        implementation_name = type(implementation).__name__
        durations = []
        errors = 0

        # Warmup iterations
        for _ in range(self.warmup_iterations):
            with contextlib.suppress(Exception):
                await asyncio.wait_for(operation_func(implementation), timeout=self.timeout_seconds)
        # Start resource monitoring
        memory_before = self._get_memory_usage() if self.measure_resources else None

        # Benchmark iterations
        for i in range(self.benchmark_iterations):
            try:
                start_time = time.perf_counter()
                await asyncio.wait_for(operation_func(implementation), timeout=self.timeout_seconds)
                end_time = time.perf_counter()

                duration_ms = (end_time - start_time) * 1000
                durations.append(duration_ms)

            except Exception as e:
                errors += 1
                logger.debug("Benchmark iteration %d failed: %s", i, e)

        # End resource monitoring
        memory_after = self._get_memory_usage() if self.measure_resources else None

        # Calculate statistics
        if durations:
            total_duration = sum(durations)
            avg_duration = statistics.mean(durations)
            median_duration = statistics.median(durations)
            min_duration = min(durations)
            max_duration = max(durations)
            std_dev = statistics.stdev(durations) if len(durations) > 1 else 0.0

            # Calculate throughput
            successful_ops = len(durations)
            ops_per_second = successful_ops / (total_duration / 1000) if total_duration > 0 else 0
            items_per_second = (
                (successful_ops * batch_size) / (total_duration / 1000)
                if total_duration > 0
                else None
            )

        else:
            # All operations failed
            total_duration = avg_duration = median_duration = 0.0
            min_duration = max_duration = std_dev = 0.0
            ops_per_second = 0.0
            items_per_second = None

        # Calculate success rate
        total_attempts = self.benchmark_iterations
        success_rate = (total_attempts - errors) / total_attempts if total_attempts > 0 else 0.0

        # Memory usage
        memory_usage_mb = None
        if memory_before is not None and memory_after is not None:
            memory_usage_mb = memory_after - memory_before

        return BenchmarkResult(
            benchmark_name=benchmark_name,
            operation=operation,
            implementation_name=implementation_name,
            total_duration_ms=total_duration,
            average_duration_ms=avg_duration,
            median_duration_ms=median_duration,
            min_duration_ms=min_duration,
            max_duration_ms=max_duration,
            std_deviation_ms=std_dev,
            operations_per_second=ops_per_second,
            items_per_second=items_per_second,
            iterations=len(durations),
            batch_size=batch_size,
            test_data_size=batch_size,
            memory_usage_mb=memory_usage_mb,
            success_rate=success_rate,
            error_count=errors,
            metadata={
                "warmup_iterations": self.warmup_iterations,
                "benchmark_iterations": self.benchmark_iterations,
                "timeout_seconds": self.timeout_seconds,
            },
        )

    def _get_memory_usage(self) -> float | None:
        """Get current memory usage in MB."""
        if self._process:
            try:
                return self._process.memory_info().rss / 1024 / 1024
            except Exception:
                return None
        return None

    def _generate_test_vectors(self, count: int, dimension: int) -> list[VectorPoint]:
        """Generate test vectors for benchmarking."""
        import random

        vectors = []
        for i in range(count):
            # Generate random normalized vector
            vector = [random.gauss(0, 1) for _ in range(dimension)]
            length = sum(x * x for x in vector) ** 0.5
            if length > 0:
                vector = [x / length for x in vector]

            vectors.append(
                VectorPoint(
                    id=f"test_vector_{i}",
                    vector=vector,
                    payload={"content": f"Test content {i}", "index": i},
                )
            )

        return vectors

    def _generate_test_texts(self, count: int, length: int) -> list[str]:
        """Generate test texts for benchmarking."""
        import random

        words = [
            "machine",
            "learning",
            "artificial",
            "intelligence",
            "algorithm",
            "data",
            "science",
            "computer",
            "programming",
            "python",
            "javascript",
            "code",
            "development",
            "software",
            "system",
            "application",
            "framework",
            "library",
            "database",
            "search",
            "vector",
            "embedding",
            "similarity",
            "neural",
            "network",
            "model",
            "training",
            "optimization",
            "performance",
            "benchmark",
        ]

        texts = []
        for _i in range(count):
            # Generate text of approximately specified length
            text_words = []
            current_length = 0

            while current_length < length:
                word = random.choice(words)
                text_words.append(word)
                current_length += len(word) + 1  # +1 for space

            text = " ".join(text_words)
            texts.append(text)

        return texts

    def _get_default_vector_backend_scenarios(self) -> list[dict[str, Any]]:
        """Get default vector backend benchmark scenarios."""
        return [
            {
                "name": "small_scale",
                "vector_count": 100,
                "dimension": 128,
                "operations": ["upsert", "search", "batch_search"],
            },
            {
                "name": "medium_scale",
                "vector_count": 1000,
                "dimension": 256,
                "operations": ["upsert", "search"],
            },
            {
                "name": "high_dimension",
                "vector_count": 100,
                "dimension": 1024,
                "operations": ["upsert", "search"],
            },
        ]

    def _get_default_embedding_scenarios(self) -> list[dict[str, Any]]:
        """Get default embedding provider benchmark scenarios."""
        return [
            {
                "name": "single_documents",
                "text_count": 20,
                "text_length": 100,
                "operations": ["documents"],
                "batch_sizes": [1, 5, 10, 20],
            },
            {
                "name": "long_documents",
                "text_count": 10,
                "text_length": 1000,
                "operations": ["documents"],
                "batch_sizes": [1, 5, 10],
            },
            {
                "name": "query_embedding",
                "text_count": 10,
                "text_length": 50,
                "operations": ["queries"],
            },
        ]

    def _get_default_rerank_scenarios(self) -> list[dict[str, Any]]:
        """Get default rerank provider benchmark scenarios."""
        return [
            {
                "name": "small_rerank",
                "document_count": 20,
                "text_length": 200,
                "document_counts": [5, 10, 20],
            },
            {
                "name": "large_rerank",
                "document_count": 50,
                "text_length": 500,
                "document_counts": [10, 25, 50],
            },
        ]

    def _get_default_data_source_scenarios(self) -> list[dict[str, Any]]:
        """Get default data source benchmark scenarios."""
        return [{"name": "content_discovery", "operations": ["discovery", "reading"], "config": {}}]


# Convenience functions


async def run_performance_benchmarks(
    components: dict[str, Any], benchmark_config: dict[str, Any] | None = None
) -> dict[str, list[BenchmarkResult]]:
    """Run performance benchmarks for multiple components.

    Args:
        components: Dict mapping component types to implementations
        benchmark_config: Configuration for benchmark suite

    Returns:
        Dict mapping component types to benchmark results
    """
    config = benchmark_config or {}
    suite = BenchmarkSuite(**config)

    results = {}

    for component_type, implementation in components.items():
        if component_type == "vector_backend":
            results[component_type] = await suite.benchmark_vector_backend(implementation)
        elif component_type == "embedding_provider":
            results[component_type] = await suite.benchmark_embedding_provider(implementation)
        elif component_type == "rerank_provider":
            results[component_type] = await suite.benchmark_rerank_provider(implementation)
        elif component_type == "data_source":
            results[component_type] = await suite.benchmark_data_source(implementation)
        else:
            logger.warning("Unknown component type for benchmarking: %s", component_type)

    return results


def print_benchmark_results(results: dict[str, list[BenchmarkResult]]) -> None:
    """Print formatted benchmark results."""
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK RESULTS")
    print("=" * 80)

    for component_type, component_results in results.items():
        print(f"\n{component_type.upper()} BENCHMARKS:")
        print("-" * 40)

        for result in component_results:
            print(f"  {result}")
            print(f"    Success Rate: {result.success_rate:.1%}")
            print(f"    Min/Max: {result.min_duration_ms:.2f}ms / {result.max_duration_ms:.2f}ms")
            if result.items_per_second:
                print(f"    Items/sec: {result.items_per_second:.2f}")
            print()


def save_benchmark_results(results: dict[str, list[BenchmarkResult]], filename: str) -> None:
    """Save benchmark results to JSON file."""
    import json

    serializable_results = {
        component_type: [result.to_dict() for result in component_results]
        for component_type, component_results in results.items()
    }
    with open(filename, "w") as f:
        json.dump(serializable_results, f, indent=2)

    logger.info("Benchmark results saved to: %s", filename)
