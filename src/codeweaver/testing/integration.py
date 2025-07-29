# sourcery skip: lambdas-should-be-short
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Integration testing framework for CodeWeaver.

Comprehensive integration testing that validates end-to-end workflows,
backend + provider + source combinations, and configuration scenarios.
"""

import contextlib
import logging
import tempfile
import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codeweaver.backends import BackendFactory, HybridSearchBackend, VectorBackend
from codeweaver.config import CodeWeaverConfig
from codeweaver.providers import EmbeddingProvider, RerankProvider, get_provider_factory
from codeweaver.sources import DataSource, SourceFactory
from codeweaver.testing.mocks import (
    MockDataSource,
    MockEmbeddingProvider,
    MockHybridSearchBackend,
    MockRerankProvider,
    MockVectorBackend,
)
from codeweaver.testing.protocol_compliance import ComplianceResult, ProtocolComplianceValidator


logger = logging.getLogger(__name__)


@dataclass
class TestConfiguration:
    """Configuration for integration tests."""

    # Component selection
    backend_type: str = "mock"
    embedding_provider: str = "mock"
    rerank_provider: str | None = "mock"
    data_source_type: str = "mock"

    # Test settings
    run_compliance_tests: bool = True
    run_performance_tests: bool = True
    run_workflow_tests: bool = True
    test_timeout_seconds: int = 60

    # Mock settings
    mock_latency_ms: float = 10.0
    mock_error_rate: float = 0.0

    # Test data
    test_documents: list[str] = field(
        default_factory=lambda: [
            "This is a test document about machine learning algorithms.",
            "Python is a popular programming language for data science.",
            "Vector databases enable efficient similarity search.",
            "Embedding models convert text into numerical representations.",
            "Retrieval augmented generation improves LLM performance.",
        ]
    )
    test_queries: list[str] = field(
        default_factory=lambda: ["machine learning", "python programming", "vector search"]
    )

    # Configuration overrides
    config_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationTestResult:
    """Result of integration test execution."""

    test_name: str
    success: bool
    duration_seconds: float
    compliance_results: dict[str, ComplianceResult] = field(default_factory=dict)
    performance_metrics: dict[str, float] = field(default_factory=dict)
    workflow_results: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "✅ PASS" if self.success else "❌ FAIL"
        return f"{status} {self.test_name} ({self.duration_seconds:.2f}s)"

    def get_detailed_report(self) -> str:
        """Get detailed test report."""
        lines = [
            f"Integration Test Report: {self.test_name}",
            "=" * (25 + len(self.test_name)),
            f"Status: {'SUCCESS' if self.success else 'FAILED'}",
            f"Duration: {self.duration_seconds:.2f} seconds",
            "",
        ]

        if self.compliance_results:
            lines.extend([
                "Protocol Compliance Results:",
                *[f"  - {result}" for result in self.compliance_results.values()],
                "",
            ])

        if self.performance_metrics:
            lines.extend([
                "Performance Metrics:",
                *[
                    f"  - {metric}: {value:.4f}ms"
                    for metric, value in self.performance_metrics.items()
                ],
                "",
            ])

        if self.workflow_results:
            lines.extend([
                "Workflow Test Results:",
                *[
                    f"  - {workflow}: {'PASS' if passed else 'FAIL'}"
                    for workflow, passed in self.workflow_results.items()
                ],
                "",
            ])

        if self.errors:
            lines.extend(["Errors:", *[f"  - {error}" for error in self.errors], ""])

        if self.warnings:
            lines.extend(["Warnings:", *[f"  - {warning}" for warning in self.warnings], ""])

        return "\n".join(lines)


class IntegrationTestSuite:
    """Comprehensive integration test suite for CodeWeaver."""

    def __init__(self, config: TestConfiguration):
        """Initialize integration test suite.

        Args:
            config: Test configuration
        """
        self.config = config
        self.validator = ProtocolComplianceValidator(
            strict_mode=True, performance_benchmarks=config.run_performance_tests
        )

        # Component instances
        self.backend: VectorBackend | None = None
        self.embedding_provider: EmbeddingProvider | None = None
        self.rerank_provider: RerankProvider | None = None
        self.data_source: DataSource | None = None

        # Test state
        self.test_collection = "integration_test_collection"
        self.temp_dir: Path | None = None

    async def run_all_tests(self) -> IntegrationTestResult:
        """Run all integration tests."""
        start_time = time.time()
        result = IntegrationTestResult(
            test_name=f"Integration Test Suite ({self.config.backend_type})",
            success=True,
            duration_seconds=0.0,
        )

        try:
            # Setup test environment
            await self._setup_test_environment()

            # Run component tests
            if self.config.run_compliance_tests:
                await self._run_compliance_tests(result)

            # Run workflow tests
            if self.config.run_workflow_tests:
                await self._run_workflow_tests(result)

                # Run end-to-end integration (part of workflow tests)
                await self._run_end_to_end_tests(result)

        except Exception as e:
            result.success = False
            result.errors.append(f"Test suite failed: {e}")
            logger.exception("Integration test suite failed")

        finally:
            # Cleanup
            await self._cleanup_test_environment()
            result.duration_seconds = time.time() - start_time

        return result

    async def _setup_test_environment(self) -> None:
        """Setup test environment with components."""
        # Create temporary directory for file-based tests
        self.temp_dir = Path(tempfile.mkdtemp(prefix="codeweaver_test_"))

        # Create mock components
        if self.config.backend_type == "mock":
            if self.config.rerank_provider:
                self.backend = MockHybridSearchBackend(
                    latency_ms=self.config.mock_latency_ms, error_rate=self.config.mock_error_rate
                )
            else:
                self.backend = MockVectorBackend(
                    latency_ms=self.config.mock_latency_ms, error_rate=self.config.mock_error_rate
                )
        else:
            # Create real backend from factory
            config = CodeWeaverConfig()
            self.backend = await BackendFactory.create_backend(config.backend)

        # Create embedding provider
        if self.config.embedding_provider == "mock":
            self.embedding_provider = MockEmbeddingProvider(
                latency_ms=self.config.mock_latency_ms, error_rate=self.config.mock_error_rate
            )
        else:
            # Create real provider from factory
            config = CodeWeaverConfig()
            factory = get_provider_factory()
            self.embedding_provider = factory.create_embedding_provider(config.providers.embedding)

        # Create rerank provider if specified
        if self.config.rerank_provider:
            if self.config.rerank_provider == "mock":
                self.rerank_provider = MockRerankProvider(
                    latency_ms=self.config.mock_latency_ms, error_rate=self.config.mock_error_rate
                )
            else:
                # Create real rerank provider from factory
                config = CodeWeaverConfig()
                self.rerank_provider = factory.get_default_reranking_provider(
                    embedding_provider_name=config.providers.embedding.provider,
                    api_key=config.providers.embedding.api_key,
                )

        # Create data source
        if self.config.data_source_type == "mock":
            self.data_source = MockDataSource(
                latency_ms=self.config.mock_latency_ms, error_rate=self.config.mock_error_rate
            )
        else:
            # Create real data source from factory
            source_config = {
                "type": self.config.data_source_type,
                "enabled": True,
                "priority": 1,
                "config": self.config.config_overrides.get("data_source_config", {}),
            }
            from codeweaver.types import SourceProvider

            source_factory = SourceFactory()
            # Map string type to SourceProvider enum
            source_type = SourceProvider(self.config.data_source_type)
            self.data_source = source_factory.create_source(source_type, source_config)

        logger.info("Test environment setup complete")

    async def _cleanup_test_environment(self) -> None:
        """Cleanup test environment."""
        try:
            # Cleanup backend
            if self.backend:
                with contextlib.suppress(Exception):
                    await self.backend.delete_collection(self.test_collection)
            # Cleanup data source
            if self.data_source and hasattr(self.data_source, "cleanup"):
                await self.data_source.cleanup()

            # Cleanup temporary directory
            if self.temp_dir and self.temp_dir.exists():
                import shutil

                shutil.rmtree(self.temp_dir)

            logger.info("Test environment cleanup complete")

        except Exception:
            logger.exception("Error during test cleanup")

    async def _run_compliance_tests(self, result: IntegrationTestResult) -> None:
        """Run protocol compliance tests."""
        logger.info("Running protocol compliance tests")

        try:
            # Test backend compliance
            if self.backend:
                if isinstance(self.backend, HybridSearchBackend):
                    compliance_result = await self.validator.validate_hybrid_search_backend(
                        self.backend
                    )
                    result.compliance_results["hybrid_search_backend"] = compliance_result
                else:
                    compliance_result = await self.validator.validate_vector_backend(self.backend)
                    result.compliance_results["vector_backend"] = compliance_result

                if not compliance_result.is_compliant:
                    result.success = False
                    result.errors.extend(compliance_result.validation_errors)

            # Test embedding provider compliance
            if self.embedding_provider:
                compliance_result = await self.validator.validate_embedding_provider(
                    self.embedding_provider
                )
                result.compliance_results["embedding_provider"] = compliance_result

                if not compliance_result.is_compliant:
                    result.success = False
                    result.errors.extend(compliance_result.validation_errors)

            # Test rerank provider compliance
            if self.rerank_provider:
                compliance_result = await self.validator.validate_rerank_provider(
                    self.rerank_provider
                )
                result.compliance_results["rerank_provider"] = compliance_result

                if not compliance_result.is_compliant:
                    result.success = False
                    result.errors.extend(compliance_result.validation_errors)

            # Test data source compliance
            if self.data_source:
                compliance_result = await self.validator.validate_data_source(self.data_source)
                result.compliance_results["data_source"] = compliance_result

                if not compliance_result.is_compliant:
                    result.success = False
                    result.errors.extend(compliance_result.validation_errors)

        except Exception as e:
            result.success = False
            result.errors.append(f"Compliance testing failed: {e}")
            logger.exception("Compliance testing failed")

    async def _run_workflow_tests(self, result: IntegrationTestResult) -> None:
        """Run individual workflow tests."""
        logger.info("Running workflow tests")

        workflows = [
            ("embedding_workflow", self._test_embedding_workflow),
            ("search_workflow", self._test_search_workflow),
            ("data_source_workflow", self._test_data_source_workflow),
        ]

        if self.rerank_provider:
            workflows.append(("rerank_workflow", self._test_rerank_workflow))

        if isinstance(self.backend, HybridSearchBackend):
            workflows.append(("hybrid_search_workflow", self._test_hybrid_search_workflow))

        for workflow_name, workflow_func in workflows:
            try:
                start_time = time.time()
                success = await workflow_func()
                duration = (time.time() - start_time) * 1000

                result.workflow_results[workflow_name] = success
                result.performance_metrics[f"{workflow_name}_duration"] = duration

                if not success:
                    result.success = False
                    result.errors.append(f"Workflow {workflow_name} failed")

            except Exception as e:
                result.workflow_results[workflow_name] = False
                result.success = False
                result.errors.append(f"Workflow {workflow_name} error: {e}")
                logger.exception("Workflow %s failed")

    async def _run_end_to_end_tests(self, result: IntegrationTestResult) -> None:
        """Run comprehensive end-to-end integration tests."""
        logger.info("Running end-to-end integration tests")

        try:
            # Test complete indexing and search pipeline
            success = await self._test_complete_pipeline()
            result.workflow_results["complete_pipeline"] = success

            if not success:
                result.success = False
                result.errors.append("Complete pipeline test failed")

            # Test configuration integration
            success = await self._test_configuration_integration()
            result.workflow_results["configuration_integration"] = success

            if not success:
                result.success = False
                result.errors.append("Configuration integration test failed")

        except Exception as e:
            result.success = False
            result.errors.append(f"End-to-end testing failed: {e}")
            logger.exception("End-to-end testing failed")

    async def _test_embedding_workflow(self) -> bool:
        """Test embedding provider workflow."""
        if not self.embedding_provider:
            return False

        try:
            # Test document embeddings
            embeddings = await self.embedding_provider.embed_documents(self.config.test_documents)
            if len(embeddings) != len(self.config.test_documents):
                return False

            # Test query embedding
            query_embedding = await self.embedding_provider.embed_query(self.config.test_queries[0])
            if len(query_embedding) != self.embedding_provider.dimension:
                return False

            # Test provider info
            self.embedding_provider.get_provider_info()

        except Exception:
            logger.exception("Embedding workflow test failed")
            return False

        else:
            return True

    async def _test_search_workflow(self) -> bool:
        """Test vector backend search workflow."""
        if not self.backend or not self.embedding_provider:
            return False

        try:
            # Create collection
            dimension = self.embedding_provider.dimension
            await self.backend.create_collection(name=self.test_collection, dimension=dimension)

            # Generate embeddings and index documents
            embeddings = await self.embedding_provider.embed_documents(self.config.test_documents)

            from codeweaver.backends.base import VectorPoint

            vectors = [
                VectorPoint(id=i, vector=embedding, payload={"content": doc, "index": i})
                for i, (doc, embedding) in enumerate(
                    zip(self.config.test_documents, embeddings, strict=False)
                )
            ]

            await self.backend.upsert_vectors(self.test_collection, vectors)

            # Test search
            query_embedding = await self.embedding_provider.embed_query(self.config.test_queries[0])
            results = await self.backend.search_vectors(
                collection_name=self.test_collection, query_vector=query_embedding, limit=3
            )

            return len(results) > 0 and all(r.score >= 0 for r in results)

        except Exception:
            logger.exception("Search workflow test failed")
            return False

    async def _test_rerank_workflow(self) -> bool:
        """Test rerank provider workflow."""
        if not self.rerank_provider:
            return False

        try:
            # Test reranking
            results = await self.rerank_provider.rerank(
                query=self.config.test_queries[0], documents=self.config.test_documents, top_k=3
            )

            # Verify results
            if len(results) > len(self.config.test_documents):
                return False

            # Check that results are sorted by relevance
            scores = [r.relevance_score for r in results]
            return scores == sorted(scores, reverse=True)

        except Exception:
            logger.exception("Rerank workflow test failed")
            return False

    async def _test_data_source_workflow(self) -> bool:
        """Test data source workflow."""
        if not self.data_source:
            return False

        try:
            # Test source validation
            test_config = {"enabled": True, "priority": 1, "source_id": "test_source"}

            if not await self.data_source.validate_source(test_config):
                return False

            # Test content discovery
            content_items = await self.data_source.discover_content(test_config)
            if not content_items:
                return False

            # Test content reading
            content = await self.data_source.read_content(content_items[0])
            if not isinstance(content, str):
                return False

            # Test metadata extraction
            metadata = await self.data_source.get_content_metadata(content_items[0])
            return isinstance(metadata, dict)

        except Exception:
            logger.exception("Data source workflow test failed")
            return False

    async def _test_hybrid_search_workflow(self) -> bool:
        """Test hybrid search workflow."""
        if not isinstance(self.backend, HybridSearchBackend) or not self.embedding_provider:
            return False

        try:
            # Setup collection with sparse index
            dimension = self.embedding_provider.dimension
            try:
                await self.backend.create_collection(name=self.test_collection, dimension=dimension)
            except ValueError as e:
                if "already exists" not in str(e):
                    raise

            await self.backend.create_sparse_index(
                collection_name=self.test_collection, fields=["content"], index_type="bm25"
            )

            # Index documents with sparse vectors
            embeddings = await self.embedding_provider.embed_documents(self.config.test_documents)

            from codeweaver.backends.base import VectorPoint

            vectors = [
                VectorPoint(
                    id=i,
                    vector=embedding,
                    payload={"content": doc, "index": i},
                    sparse_vector={j: float(j) for j in range(5)},  # Mock sparse vector
                )
                for i, (doc, embedding) in enumerate(
                    zip(self.config.test_documents, embeddings, strict=False)
                )
            ]

            await self.backend.upsert_vectors(self.test_collection, vectors)

            # Test hybrid search
            query_embedding = await self.embedding_provider.embed_query(self.config.test_queries[0])
            results = await self.backend.hybrid_search(
                collection_name=self.test_collection,
                dense_vector=query_embedding,
                sparse_query="machine learning",
                limit=3,
            )

            return len(results) > 0 and all(r.score >= 0 for r in results)

        except Exception:
            logger.exception("Hybrid search workflow test failed")
            return False

    async def _test_complete_pipeline(self) -> bool:
        # sourcery skip: low-code-quality
        """Test complete indexing and search pipeline."""
        if not all([self.backend, self.embedding_provider, self.data_source]):
            return False

        try:
            # Create collection
            dimension = self.embedding_provider.dimension
            try:
                await self.backend.create_collection(name=self.test_collection, dimension=dimension)
            except ValueError as e:
                if "already exists" not in str(e):
                    raise

            # Discover content from data source
            test_config = {"enabled": True, "priority": 1}
            content_items = await self.data_source.discover_content(test_config)

            if not content_items:
                return False

            # Read and embed content
            documents = []
            for item in content_items[:5]:  # Limit to first 5 items
                content = await self.data_source.read_content(item)
                documents.append(content)

            embeddings = await self.embedding_provider.embed_documents(documents)

            # Index in vector backend
            from codeweaver.backends.base import VectorPoint

            vectors = [
                VectorPoint(id=item.path, vector=embedding, payload=item.model_dump())
                for item, embedding in zip(content_items[:5], embeddings, strict=False)
            ]

            await self.backend.upsert_vectors(self.test_collection, vectors)

            # Search
            query_embedding = await self.embedding_provider.embed_query("test query")
            results = await self.backend.search_vectors(
                collection_name=self.test_collection, query_vector=query_embedding, limit=3
            )

            # Optional reranking
            if self.rerank_provider and results:
                # Get documents for reranking
                result_docs = []
                for result in results:
                    if result.payload and "path" in result.payload:
                        # Find corresponding content item
                        for item in content_items:
                            if item.path == result.id:
                                content = await self.data_source.read_content(item)
                                result_docs.append(content)
                                break

                if result_docs:
                    rerank_results = await self.rerank_provider.rerank(
                        query="test query", documents=result_docs, top_k=3
                    )

                    if not rerank_results:
                        return False

            return len(results) > 0

        except Exception:
            logger.exception("Complete pipeline test failed")
            return False

    async def _test_configuration_integration(self) -> bool:
        """Test configuration integration."""
        try:
            # Test creating components from configuration
            config = CodeWeaverConfig()

            # Validate configuration structure (not content)
            if not hasattr(config, "backend"):
                return False

            if not hasattr(config, "providers"):
                return False

            # Test configuration serialization/deserialization
            config_obj = config.model_dump()
            return isinstance(config_obj, dict)

        except Exception:
            logger.exception("Configuration integration test failed")
            return False


# Convenience functions


def create_test_configuration(
    backend_type: str = "mock", embedding_provider: str = "mock", **kwargs: Any
) -> TestConfiguration:
    """Create a test configuration with sensible defaults."""
    return TestConfiguration(
        backend_type=backend_type, embedding_provider=embedding_provider, **kwargs
    )


async def run_integration_tests(config: TestConfiguration | None = None) -> IntegrationTestResult:
    """Run integration tests with given configuration."""
    if config is None:
        config = create_test_configuration()

    suite = IntegrationTestSuite(config)
    return await suite.run_all_tests()


async def run_quick_integration_test() -> IntegrationTestResult:
    """Run a quick integration test with mock components."""
    config = create_test_configuration(run_performance_tests=False, mock_latency_ms=1.0)
    return await run_integration_tests(config)
