#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
CodeWeaver Testing Framework Demonstration.

This script demonstrates all aspects of the CodeWeaver testing framework:
- Protocol compliance validation
- Mock implementations
- Integration testing
- Performance benchmarking
- Factory pattern validation

Run with: python examples/testing_framework_demo.py
"""

import asyncio
import logging
import sys

from pathlib import Path


# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def demo_protocol_compliance() -> None:
    """Demonstrate protocol compliance validation."""
    print("\n" + "=" * 80)
    print("PROTOCOL COMPLIANCE VALIDATION DEMO")
    print("=" * 80)

    from codeweaver.testing.mocks import (
        MockDataSource,
        MockEmbeddingProvider,
        MockHybridSearchBackend,
        MockRerankProvider,
        MockVectorBackend,
    )
    from codeweaver.testing.protocol_compliance import ProtocolComplianceValidator

    # Create validator
    validator = ProtocolComplianceValidator(strict_mode=True, performance_benchmarks=True)

    # Test implementations
    implementations = {
        "vector_backend": MockVectorBackend(latency_ms=5.0),
        "hybrid_search_backend": MockHybridSearchBackend(latency_ms=10.0),
        "embedding_provider": MockEmbeddingProvider(dimension=128, latency_ms=20.0),
        "rerank_provider": MockRerankProvider(latency_ms=15.0),
        "data_source": MockDataSource(latency_ms=8.0),
    }

    print("Testing protocol compliance for all implementations...")

    results = await validator.validate_all_protocols(implementations)

    # Display results
    for result in results.values():
        print(f"\n{result}")

        if result.performance_metrics:
            print("  Performance Metrics:")
            for metric, value in result.performance_metrics.items():
                print(f"    - {metric}: {value:.2f}ms")

        if result.validation_errors:
            print("  Validation Errors:")
            for error in result.validation_errors[:3]:  # Show first 3
                print(f"    - {error}")

        if result.warnings:
            print("  Warnings:")
            for warning in result.warnings[:3]:  # Show first 3
                print(f"    - {warning}")

    # Summary
    compliant_count = sum(bool(result.is_compliant)
                      for result in results.values())
    total_count = len(results)

    print(f"\n📊 COMPLIANCE SUMMARY: {compliant_count}/{total_count} protocols compliant")

    if compliant_count == total_count:
        print("✅ All implementations are protocol compliant!")
    else:
        print("❌ Some implementations need attention.")


async def demo_mock_implementations() -> None:
    # sourcery skip: avoid-global-variables, no-long-functions
    """Demonstrate mock implementations."""
    print("\n" + "=" * 80)
    print("MOCK IMPLEMENTATIONS DEMO")
    print("=" * 80)

    from codeweaver.testing.mocks import (
        MockDataSource,
        MockEmbeddingProvider,
        MockRerankProvider,
        MockVectorBackend,
    )

    # Demonstrate vector backend
    print("\n🔍 Vector Backend Mock:")
    backend = MockVectorBackend(latency_ms=10.0, error_rate=0.1)

    # Create collection
    await backend.create_collection("demo_collection", dimension=128)
    print("  ✅ Created collection")

    # Upsert vectors
    from codeweaver.backends.base import VectorPoint

    test_vectors = [
        VectorPoint(
            id=f"vector_{i}",
            vector=[float(j + i * 0.1) for j in range(128)],
            payload={"content": f"Test content {i}"},
        )
        for i in range(5)
    ]

    await backend.upsert_vectors("demo_collection", test_vectors)
    print(f"  ✅ Upserted {len(test_vectors)} vectors")

    # Search vectors
    query_vector = test_vectors[0].vector
    results = await backend.search_vectors("demo_collection", query_vector, limit=3)
    print(f"  ✅ Search returned {len(results)} results")

    # Demonstrate embedding provider
    print("\n🧠 Embedding Provider Mock:")
    provider = MockEmbeddingProvider(dimension=256, latency_ms=25.0)

    test_texts = ["Hello world", "Test document", "Sample text"]
    embeddings = await provider.embed_documents(test_texts)
    print(f"  ✅ Generated embeddings: {len(embeddings)} vectors of dimension {len(embeddings[0])}")

    query_embedding = await provider.embed_query("search query")
    print(f"  ✅ Generated query embedding: dimension {len(query_embedding)}")

    # Demonstrate rerank provider
    print("\n🔄 Rerank Provider Mock:")
    reranker = MockRerankProvider(latency_ms=20.0)

    query = "machine learning"
    documents = [
        "Introduction to machine learning algorithms",
        "Python programming tutorial",
        "Deep learning with neural networks",
    ]

    rerank_results = await reranker.rerank(query, documents, top_k=3)
    print(f"  ✅ Reranked {len(documents)} documents:")
    for i, result in enumerate(rerank_results):
        print(f"    {i + 1}. Score: {result.relevance_score:.3f} - Index: {result.index}")

    # Demonstrate data source
    print("\n📁 Data Source Mock:")
    source = MockDataSource(latency_ms=15.0)

    test_config = {"enabled": True, "priority": 1, "source_id": "demo_source"}

    # Discover content
    content_items = await source.discover_content(test_config)
    print(f"  ✅ Discovered {len(content_items)} content items")

    # Read content
    if content_items:
        content = await source.read_content(content_items[0])
        print(f"  ✅ Read content: {len(content)} characters")

        # Get metadata
        metadata = await source.get_content_metadata(content_items[0])
        print(f"  ✅ Retrieved metadata: {len(metadata)} fields")

    print("\n🎯 All mock implementations working correctly!")


async def demo_integration_testing() -> None:
    """Demonstrate integration testing."""
    print("\n" + "=" * 80)
    print("INTEGRATION TESTING DEMO")
    print("=" * 80)

    from codeweaver.testing.integration import (
        create_test_configuration,
        run_integration_tests,
        run_quick_integration_test,
    )

    # Quick integration test
    print("\n🚀 Running quick integration test...")
    result = await run_quick_integration_test()

    print(f"Result: {result}")
    print(f"Duration: {result.duration_seconds:.2f} seconds")

    if result.compliance_results:
        print("\nCompliance Results:")
        for protocol, compliance in result.compliance_results.items():
            status = "✅" if compliance.is_compliant else "❌"
            print(
                f"  {status} {protocol}: {compliance.passed_tests}/{compliance.total_tests} tests passed"
            )

    if result.workflow_results:
        print("\nWorkflow Results:")
        for workflow, success in result.workflow_results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {workflow}")

    # Custom configuration test
    print("\n⚙️ Running integration test with custom configuration...")

    config = create_test_configuration(
        backend_type="mock",
        embedding_provider="mock",
        rerank_provider="mock",
        run_performance_tests=False,  # Skip for demo speed
        mock_latency_ms=1.0,
        test_documents=[
            "Custom document about artificial intelligence",
            "Machine learning in practice",
            "Deep learning fundamentals",
        ],
        test_queries=["AI", "ML", "neural networks"],
    )

    result = await run_integration_tests(config)

    print(f"Custom test result: {result}")

    if result.errors:
        print("Errors encountered:")
        for error in result.errors[:3]:  # Show first 3
            print(f"  - {error}")

    if result.warnings:
        print("Warnings:")
        for warning in result.warnings[:3]:  # Show first 3
            print(f"  - {warning}")


async def demo_performance_benchmarking() -> None:
    # sourcery skip: no-long-functions
    """Demonstrate performance benchmarking."""
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARKING DEMO")
    print("=" * 80)

    from codeweaver.testing.benchmarks import (
        BenchmarkSuite,
        print_benchmark_results,
        run_performance_benchmarks,
    )
    from codeweaver.testing.mocks import (
        MockDataSource,
        MockEmbeddingProvider,
        MockRerankProvider,
        MockVectorBackend,
    )

    # Create benchmark suite
    suite = BenchmarkSuite(
        warmup_iterations=2,
        benchmark_iterations=5,
        timeout_seconds=30,
        measure_resources=False,  # Disabled for demo
    )

    print("\n📊 Benchmarking individual components...")

    # Benchmark vector backend
    print("\n🔍 Vector Backend Benchmark:")
    backend = MockVectorBackend(latency_ms=2.0)
    backend_results = await suite.benchmark_vector_backend(backend)

    for result in backend_results:
        print(f"  📈 {result.benchmark_name}:")
        print(f"    - Operations/sec: {result.operations_per_second:.2f}")
        print(f"    - Average duration: {result.average_duration_ms:.2f}ms")
        print(f"    - Success rate: {result.success_rate:.1%}")

    # Benchmark embedding provider
    print("\n🧠 Embedding Provider Benchmark:")
    provider = MockEmbeddingProvider(latency_ms=10.0)
    provider_results = await suite.benchmark_embedding_provider(provider)

    for result in provider_results:
        print(f"  📈 {result.benchmark_name}:")
        print(f"    - Operations/sec: {result.operations_per_second:.2f}")
        print(f"    - Average duration: {result.average_duration_ms:.2f}ms")
        if result.items_per_second:
            print(f"    - Items/sec: {result.items_per_second:.2f}")

    # Comprehensive benchmark
    print("\n🎯 Comprehensive Multi-Component Benchmark:")

    components = {
        "vector_backend": MockVectorBackend(latency_ms=1.0),
        "embedding_provider": MockEmbeddingProvider(latency_ms=5.0),
        "rerank_provider": MockRerankProvider(latency_ms=8.0),
        "data_source": MockDataSource(latency_ms=3.0),
    }

    benchmark_config = {"warmup_iterations": 1, "benchmark_iterations": 3, "timeout_seconds": 30}

    all_results = await run_performance_benchmarks(components, benchmark_config)

    # Print formatted results
    print_benchmark_results(all_results)

    # Performance analysis
    print("\n📊 Performance Analysis:")
    total_operations = 0
    total_duration = 0.0

    for component_type, component_results in all_results.items():
        component_ops = sum(r.operations_per_second for r in component_results)
        component_duration = sum(r.average_duration_ms for r in component_results)

        print(f"  {component_type}:")
        print(f"    - Total ops/sec: {component_ops:.2f}")
        print(f"    - Average duration: {component_duration / len(component_results):.2f}ms")

        total_operations += component_ops
        total_duration += component_duration

    print(f"\n🎯 Overall Performance: {total_operations:.2f} total ops/sec across all components")


async def demo_factory_validation() -> None:
    """Demonstrate factory pattern validation."""
    print("\n" + "=" * 80)
    print("FACTORY PATTERN VALIDATION DEMO")
    print("=" * 80)

    from codeweaver.testing.factory_validation import (
        print_factory_validation_results,
        validate_all_factory_patterns,
        validate_factory_pattern,
    )

    print("\n🏭 Validating all factory patterns...")

    # Validate all factories
    results = await validate_all_factory_patterns()

    # Print results
    print_factory_validation_results(results)

    # Detailed analysis
    print("\n🔍 Detailed Factory Analysis:")

    for factory_name, result in results.items():
        print(f"\n📋 {factory_name}:")
        print(f"  Status: {'✅ Valid' if result.is_valid else '❌ Invalid'}")
        print(f"  Created instances: {result.created_instances}")
        print(f"  Failed creations: {result.failed_creations}")

        if result.test_details:
            print("  Test details:")
            for test_name, test_result in result.test_details.items():
                status = "✅" if test_result == "success" else "❌"
                print(f"    {status} {test_name}: {test_result}")

        if result.validation_errors:
            print("  Errors:")
            for error in result.validation_errors[:2]:  # Show first 2
                print(f"    - {error}")

    # Test specific factory
    print("\n🔧 Testing specific factory pattern...")

    backend_result = await validate_factory_pattern("backend")
    print("\nBackend Factory Validation:")
    print(backend_result.get_detailed_report())


async def demo_comprehensive_testing() -> None:
    # sourcery skip: no-long-functions
    """Demonstrate comprehensive testing scenario."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TESTING SCENARIO")
    print("=" * 80)

    print("\n🎯 Running comprehensive testing workflow...")

    # This would be a typical testing workflow for a new implementation

    # 1. Protocol Compliance
    print("\n1️⃣ Step 1: Protocol Compliance Validation")
    from codeweaver.testing import validate_vector_backend_protocol
    from codeweaver.testing.mocks import MockVectorBackend

    backend = MockVectorBackend(latency_ms=5.0)
    compliance_result = await validate_vector_backend_protocol(backend)

    print(f"   Protocol compliance: {'✅ PASS' if compliance_result.is_compliant else '❌ FAIL'}")
    print(f"   Tests passed: {compliance_result.passed_tests}/{compliance_result.total_tests}")

    # 2. Performance Benchmarking
    print("\n2️⃣ Step 2: Performance Benchmarking")
    from codeweaver.testing.benchmarks import BenchmarkSuite

    suite = BenchmarkSuite(warmup_iterations=1, benchmark_iterations=3)
    bench_results = await suite.benchmark_vector_backend(backend)

    avg_ops_per_sec = sum(r.operations_per_second for r in bench_results) / len(bench_results)
    print(f"   Average performance: {avg_ops_per_sec:.2f} ops/sec")

    # Performance threshold check
    performance_threshold = 50.0  # ops/sec
    perf_status = "✅ PASS" if avg_ops_per_sec >= performance_threshold else "❌ FAIL"
    print(f"   Performance threshold ({performance_threshold} ops/sec): {perf_status}")

    # 3. Integration Testing
    print("\n3️⃣ Step 3: Integration Testing")
    from codeweaver.testing.integration import create_test_configuration, run_integration_tests

    config = create_test_configuration(
        run_performance_tests=False,  # Already done above
        mock_latency_ms=1.0,
    )

    integration_result = await run_integration_tests(config)

    print(f"   Integration test: {'✅ PASS' if integration_result.success else '❌ FAIL'}")
    print(f"   Duration: {integration_result.duration_seconds:.2f}s")

    workflow_pass_count = sum(bool(success)
                          for success in integration_result.workflow_results.values())
    workflow_total = len(integration_result.workflow_results)
    print(f"   Workflow tests: {workflow_pass_count}/{workflow_total} passed")

    # 4. Factory Validation
    print("\n4️⃣ Step 4: Factory Pattern Validation")
    from codeweaver.testing.factory_validation import validate_factory_pattern

    factory_result = await validate_factory_pattern("backend")

    print(f"   Factory validation: {'✅ PASS' if factory_result.is_valid else '❌ FAIL'}")
    print(f"   Instances created: {factory_result.created_instances}")

    # Overall Assessment
    print("\n📊 OVERALL ASSESSMENT:")

    all_tests_passed = all([
        compliance_result.is_compliant,
        avg_ops_per_sec >= performance_threshold,
        integration_result.success,
        factory_result.is_valid,
    ])

    overall_status = "✅ ALL TESTS PASSED" if all_tests_passed else "❌ SOME TESTS FAILED"
    print(f"   {overall_status}")

    if all_tests_passed:
        print("   🎉 Implementation is ready for production!")
    else:
        print("   🔧 Implementation needs attention before deployment.")

    # Test Summary
    print("\n📋 Test Summary:")
    print(f"   - Protocol Compliance: {'✅' if compliance_result.is_compliant else '❌'}")
    print(
        f"   - Performance Benchmark: {'✅' if avg_ops_per_sec >= performance_threshold else '❌'}"
    )
    print(f"   - Integration Testing: {'✅' if integration_result.success else '❌'}")
    print(f"   - Factory Validation: {'✅' if factory_result.is_valid else '❌'}")


async def main() -> int:
    """Run all testing framework demonstrations."""
    print("🚀 CodeWeaver Testing Framework Demonstration")
    print("=" * 80)
    print("This demo showcases all aspects of the CodeWeaver testing framework.")
    print("Each section demonstrates different testing capabilities.")
    print("=" * 80)

    try:
        # Run all demonstrations
        await demo_protocol_compliance()
        await demo_mock_implementations()
        await demo_integration_testing()
        await demo_performance_benchmarking()
        await demo_factory_validation()
        await demo_comprehensive_testing()

        print("\n" + "=" * 80)
        print("🎉 TESTING FRAMEWORK DEMONSTRATION COMPLETE")
        print("=" * 80)
        print("\n✅ All demonstrations completed successfully!")
        print("\n📚 Next steps:")
        print("   1. Run the actual test suite: pytest tests/ -v")
        print("   2. Explore individual testing components")
        print("   3. Create custom tests for your implementations")
        print("   4. Integrate testing into your CI/CD pipeline")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        logger.exception("Demonstration failed")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
