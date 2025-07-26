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
    run_performance_benchmarks,
)
from codeweaver.testing.integration import (
    IntegrationTestSuite,
    TestConfiguration,
    create_test_configuration,
    run_integration_tests,
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
    validate_embedding_provider_protocol,
    validate_data_source_protocol,
    validate_hybrid_search_backend_protocol,
    validate_rerank_provider_protocol,
    validate_vector_backend_protocol,
)


__all__ = [
    "BenchmarkResult",
    # Performance Benchmarks
    "BenchmarkSuite",
    "ComplianceResult",
    # Integration Testing
    "IntegrationTestSuite",
    "MockDataSource",
    "MockEmbeddingProvider",
    "MockHybridSearchBackend",
    "MockRerankProvider",
    # Mock Implementations
    "MockVectorBackend",
    # Protocol Compliance
    "ProtocolComplianceValidator",
    "TestConfiguration",
    "create_test_configuration",
    "run_integration_tests",
    "run_performance_benchmarks",
    "validate_embedding_provider_protocol",
    "validate_data_source_protocol",
    "validate_hybrid_search_backend_protocol",
    "validate_rerank_provider_protocol",
    "validate_vector_backend_protocol",
]
