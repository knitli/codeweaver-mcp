# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
CodeWeaver Testing Framework.

Comprehensive testing framework for protocol compliance validation,
mock implementations, and integration testing of CodeWeaver's extensible architecture.
"""

from codeweaver.testing.benchmarks import (
    BenchmarkResult,
    BenchmarkSuite,
    print_benchmark_results,
    run_performance_benchmarks,
    save_benchmark_results,
)
from codeweaver.testing.integration import (
    IntegrationTestSuite,
    TestConfiguration,
    create_test_configuration,
    run_integration_tests,
    run_quick_integration_test,
)
from codeweaver.testing.mocks import (
    MockDataSource,
    MockEmbeddingProvider,
    MockHybridSearchBackend,
    MockRerankProvider,
    MockVectorBackend,
)
from codeweaver.testing.protocol_compliance import (
    ComplianceResult,
    ProtocolComplianceValidator,
    validate_data_source_protocol,
    validate_embedding_provider_protocol,
    validate_hybrid_search_backend_protocol,
    validate_rerank_provider_protocol,
    validate_vector_backend_protocol,
)


__all__ = [
    "BenchmarkResult",
    "BenchmarkSuite",
    "ComplianceResult",
    "IntegrationTestSuite",
    "MockDataSource",
    "MockEmbeddingProvider",
    "MockHybridSearchBackend",
    "MockRerankProvider",
    "MockVectorBackend",
    "ProtocolComplianceValidator",
    "TestConfiguration",
    "create_test_configuration",
    "print_benchmark_results",
    "run_integration_tests",
    "run_performance_benchmarks",
    "run_quick_integration_test",
    "save_benchmark_results",
    "validate_data_source_protocol",
    "validate_embedding_provider_protocol",
    "validate_hybrid_search_backend_protocol",
    "validate_rerank_provider_protocol",
    "validate_vector_backend_protocol",
]
