# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Protocol compliance validation framework for CodeWeaver.

Comprehensive validation framework that ensures all protocol implementations
conform to their respective interfaces with proper typing, error handling,
and behavior validation.
"""

import contextlib
import inspect
import logging
import time

from typing import Any

from pydantic import Field
from pydantic.dataclasses import dataclass

from codeweaver.backends import HybridSearchBackend, VectorBackend
from codeweaver.cw_types import (
    CollectionInfo,
    ContentItem,
    DistanceMetric,
    HybridStrategy,
    ProviderInfo,
    RerankResult,
    SearchResult,
    VectorPoint,
)
from codeweaver.providers import EmbeddingProvider, RerankProvider
from codeweaver.sources import DataSource, SourceCapability, SourceConfig


logger = logging.getLogger(__name__)


@dataclass
class ComplianceResult:
    """Result of protocol compliance validation."""

    protocol_name: str
    implementation_name: str
    is_compliant: bool
    passed_tests: int
    total_tests: int
    validation_errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    performance_metrics: dict[str, float] = Field(default_factory=dict)
    test_details: dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        status = "✅ PASS" if self.is_compliant else "❌ FAIL"
        return (
            f"{status} {self.protocol_name} compliance for {self.implementation_name}: "
            f"{self.passed_tests}/{self.total_tests} tests passed"
        )

    def get_detailed_report(self) -> str:
        """Get detailed compliance report."""
        lines = [
            "Protocol Compliance Report",
            "=" * 50,
            f"Protocol: {self.protocol_name}",
            f"Implementation: {self.implementation_name}",
            f"Status: {'COMPLIANT' if self.is_compliant else 'NON-COMPLIANT'}",
            f"Tests Passed: {self.passed_tests}/{self.total_tests}",
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

        if self.performance_metrics:
            lines.extend([
                "Performance Metrics:",
                *[
                    f"  - {metric}: {value:.4f}ms"
                    for metric, value in self.performance_metrics.items()
                ],
                "",
            ])

        return "\n".join(lines)


class ProtocolComplianceValidator:
    """Comprehensive protocol compliance validator."""

    def __init__(self, *, strict_mode: bool = True, performance_benchmarks: bool = True):
        """Initialize the compliance validator.

        Args:
            strict_mode: Whether to enforce strict type checking and behavior
            performance_benchmarks: Whether to run performance benchmarks
        """
        self.strict_mode = strict_mode
        self.performance_benchmarks = performance_benchmarks
        self.test_data_cache: dict[str, Any] = {}

    async def validate_all_protocols(
        self, implementations: dict[str, Any]
    ) -> dict[str, ComplianceResult]:
        """Validate multiple protocol implementations.

        Args:
            implementations: Dict mapping protocol names to implementations

        Returns:
            Dict mapping protocol names to compliance results
        """
        results = {}

        for protocol_name, implementation in implementations.items():
            if protocol_name == "vector_backend":
                result = await self.validate_vector_backend(implementation)
            elif protocol_name == "hybrid_search_backend":
                result = await self.validate_hybrid_search_backend(implementation)
            elif protocol_name == "embedding_provider":
                result = await self.validate_embedding_provider(implementation)
            elif protocol_name == "rerank_provider":
                result = await self.validate_rerank_provider(implementation)
            elif protocol_name == "data_source":
                result = await self.validate_data_source(implementation)
            else:
                result = ComplianceResult(
                    protocol_name=protocol_name,
                    implementation_name=type(implementation).__name__,
                    is_compliant=False,
                    passed_tests=0,
                    total_tests=1,
                    validation_errors=[f"Unknown protocol: {protocol_name}"],
                )

            results[protocol_name] = result

        return results

    async def validate_vector_backend(self, backend: VectorBackend) -> ComplianceResult:
        """Validate VectorBackend protocol compliance."""
        result = ComplianceResult(
            protocol_name="VectorBackend",
            implementation_name=type(backend).__name__,
            is_compliant=True,
            passed_tests=0,
            total_tests=0,
        )

        # Test method existence and signatures
        await self._validate_protocol_interface(backend, VectorBackend, result)

        # Test basic functionality
        if hasattr(backend, "create_collection"):
            await self._test_vector_backend_functionality(backend, result)

        # Performance benchmarks
        if self.performance_benchmarks and result.is_compliant:
            await self._benchmark_vector_backend(backend, result)

        return result

    async def validate_hybrid_search_backend(
        self, backend: HybridSearchBackend
    ) -> ComplianceResult:
        """Validate HybridSearchBackend protocol compliance."""
        result = ComplianceResult(
            protocol_name="HybridSearchBackend",
            implementation_name=type(backend).__name__,
            is_compliant=True,
            passed_tests=0,
            total_tests=0,
        )

        # First validate as VectorBackend
        base_result = await self.validate_vector_backend(backend)
        result.passed_tests += base_result.passed_tests
        result.total_tests += base_result.total_tests
        result.validation_errors.extend(base_result.validation_errors)
        result.warnings.extend(base_result.warnings)

        if not base_result.is_compliant:
            result.is_compliant = False

        # Test hybrid search specific functionality
        await self._validate_protocol_interface(backend, HybridSearchBackend, result)

        if hasattr(backend, "create_sparse_index"):
            await self._test_hybrid_search_functionality(backend, result)

        # Performance benchmarks
        if self.performance_benchmarks and result.is_compliant:
            await self._benchmark_hybrid_search_backend(backend, result)

        return result

    async def validate_embedding_provider(self, provider: EmbeddingProvider) -> ComplianceResult:
        """Validate EmbeddingProvider protocol compliance."""
        result = ComplianceResult(
            protocol_name="EmbeddingProvider",
            implementation_name=type(provider).__name__,
            is_compliant=True,
            passed_tests=0,
            total_tests=0,
        )

        # Test protocol interface
        await self._validate_protocol_interface(provider, EmbeddingProvider, result)

        # Test embedding functionality
        if hasattr(provider, "embed_documents"):
            await self._test_embedding_provider_functionality(provider, result)

        # Performance benchmarks
        if self.performance_benchmarks and result.is_compliant:
            await self._benchmark_embedding_provider(provider, result)

        return result

    async def validate_rerank_provider(self, provider: RerankProvider) -> ComplianceResult:
        """Validate RerankProvider protocol compliance."""
        result = ComplianceResult(
            protocol_name="RerankProvider",
            implementation_name=type(provider).__name__,
            is_compliant=True,
            passed_tests=0,
            total_tests=0,
        )

        # Test protocol interface
        await self._validate_protocol_interface(provider, RerankProvider, result)

        # Test reranking functionality
        if hasattr(provider, "rerank"):
            await self._test_rerank_provider_functionality(provider, result)

        # Performance benchmarks
        if self.performance_benchmarks and result.is_compliant:
            await self._benchmark_rerank_provider(provider, result)

        return result

    async def validate_data_source(self, source: DataSource) -> ComplianceResult:
        """Validate DataSource protocol compliance."""
        result = ComplianceResult(
            protocol_name="DataSource",
            implementation_name=type(source).__name__,
            is_compliant=True,
            passed_tests=0,
            total_tests=0,
        )

        # Test protocol interface
        await self._validate_protocol_interface(source, DataSource, result)

        # Test data source functionality
        if hasattr(source, "discover_content"):
            await self._test_data_source_functionality(source, result)

        # Performance benchmarks
        if self.performance_benchmarks and result.is_compliant:
            await self._benchmark_data_source(source, result)

        return result

    async def _validate_protocol_interface(
        self, implementation: Any, protocol: type, result: ComplianceResult
    ) -> None:
        """Validate that implementation conforms to protocol interface."""
        protocol_methods = self._get_protocol_methods(protocol)

        for method_name, method_info in protocol_methods.items():
            result.total_tests += 1

            # Check method exists
            if not hasattr(implementation, method_name):
                result.validation_errors.append(f"Missing method: {method_name}")
                continue

            method = getattr(implementation, method_name)

            # Check callable
            if not callable(method):
                result.validation_errors.append(f"Attribute {method_name} is not callable")
                continue

            # Check signature compatibility (basic check)
            try:
                sig = inspect.signature(method)
                expected_params = method_info.get("parameters", [])

                if self.strict_mode:
                    self._validate_method_signature(method_name, sig, expected_params, result)

                result.passed_tests += 1

            except Exception as e:
                result.validation_errors.append(f"Error inspecting {method_name}: {e}")

        # Update compliance status
        if result.validation_errors:
            result.is_compliant = False

    def _get_protocol_methods(self, protocol: type) -> dict[str, dict[str, Any]]:
        """Extract method information from protocol."""
        methods = {}

        for name, method in inspect.getmembers(protocol, inspect.isfunction):
            if not name.startswith("_"):
                try:
                    sig = inspect.signature(method)
                    methods[name] = {
                        "signature": sig,
                        "parameters": list(sig.parameters.keys()),
                        "return_annotation": sig.return_annotation,
                    }
                except Exception:
                    # Skip if can't get signature
                    pass

        # Also get abstract methods if any
        for name in getattr(protocol, "__abstractmethods__", set()):
            if name not in methods:
                methods[name] = {"parameters": [], "signature": None}

        return methods

    def _validate_method_signature(
        self,
        method_name: str,
        signature: inspect.Signature,
        expected_params: list[str],
        result: ComplianceResult,
    ) -> None:
        """Validate method signature in strict mode."""
        actual_params = list(signature.parameters.keys())

        # Allow for 'self' parameter
        if actual_params and actual_params[0] == "self":
            actual_params = actual_params[1:]

        # Basic parameter count check
        if len(actual_params) < len(expected_params) - 1:  # Allow for **kwargs
            result.warnings.append(
                f"Method {method_name} may have insufficient parameters: "
                f"got {actual_params}, expected similar to {expected_params}"
            )

    async def _test_vector_backend_functionality(
        self, backend: VectorBackend, result: ComplianceResult
    ) -> None:
        """Test VectorBackend functionality with real operations."""
        test_collection = "test_compliance_collection"
        test_vectors = self._get_test_vectors()

        try:
            # Test collection creation
            result.total_tests += 1
            start_time = time.time()
            await backend.create_collection(
                name=test_collection,
                dimension=len(test_vectors[0].vector),
                distance_metric=DistanceMetric.COSINE,
            )
            result.performance_metrics["create_collection"] = (time.time() - start_time) * 1000
            result.passed_tests += 1

            # Test vector upsert
            result.total_tests += 1
            start_time = time.time()
            await backend.upsert_vectors(test_collection, test_vectors)
            result.performance_metrics["upsert_vectors"] = (time.time() - start_time) * 1000
            result.passed_tests += 1

            # Test vector search
            result.total_tests += 1
            start_time = time.time()
            search_results = await backend.search_vectors(
                collection_name=test_collection, query_vector=test_vectors[0].vector, limit=5
            )
            result.performance_metrics["search_vectors"] = (time.time() - start_time) * 1000

            if not isinstance(search_results, list):
                result.validation_errors.append("search_vectors must return a list")
            elif search_results and not isinstance(search_results[0], SearchResult):
                result.validation_errors.append("search_vectors must return list of SearchResult")
            else:
                result.passed_tests += 1

            # Test collection info
            result.total_tests += 1
            collection_info = await backend.get_collection_info(test_collection)
            if isinstance(collection_info, CollectionInfo):
                result.passed_tests += 1
            else:
                result.validation_errors.append("get_collection_info must return CollectionInfo")

            # Test list collections
            result.total_tests += 1
            collections = await backend.list_collections()
            if isinstance(collections, list) and test_collection in collections:
                result.passed_tests += 1
            else:
                result.validation_errors.append(
                    "list_collections must return list containing created collection"
                )

            # Test vector deletion
            result.total_tests += 1
            await backend.delete_vectors(test_collection, [test_vectors[0].id])
            result.passed_tests += 1

            # Cleanup
            await backend.delete_collection(test_collection)

        except Exception as e:
            result.validation_errors.append(f"Vector backend functionality test failed: {e}")
            result.is_compliant = False

            # Try cleanup anyway
            with contextlib.suppress(Exception):
                await backend.delete_collection(test_collection)

    async def _test_hybrid_search_functionality(
        self, backend: HybridSearchBackend, result: ComplianceResult
    ) -> None:
        """Test HybridSearchBackend functionality."""
        test_collection = "test_hybrid_compliance_collection"
        test_vectors = self._get_test_vectors_with_sparse()

        try:
            # Create collection first
            await backend.create_collection(
                name=test_collection,
                dimension=len(test_vectors[0].vector),
                distance_metric=DistanceMetric.COSINE,
            )

            # Test sparse index creation
            result.total_tests += 1
            start_time = time.time()
            await backend.create_sparse_index(
                collection_name=test_collection, fields=["content", "title"], index_type="bm25"
            )
            result.performance_metrics["create_sparse_index"] = (time.time() - start_time) * 1000
            result.passed_tests += 1

            # Upsert vectors with sparse data
            await backend.upsert_vectors(test_collection, test_vectors)

            # Test hybrid search
            result.total_tests += 1
            start_time = time.time()
            hybrid_results = await backend.hybrid_search(
                collection_name=test_collection,
                dense_vector=test_vectors[0].vector,
                sparse_query="test query",
                limit=5,
                hybrid_strategy=HybridStrategy.RRF,
                alpha=0.5,
            )
            result.performance_metrics["hybrid_search"] = (time.time() - start_time) * 1000

            if not isinstance(hybrid_results, list):
                result.validation_errors.append("hybrid_search must return a list")
            elif hybrid_results and not isinstance(hybrid_results[0], SearchResult):
                result.validation_errors.append("hybrid_search must return list of SearchResult")
            else:
                result.passed_tests += 1

            # Test sparse vector updates
            result.total_tests += 1
            await backend.update_sparse_vectors(test_collection, test_vectors[:1])
            result.passed_tests += 1

            # Cleanup
            await backend.delete_collection(test_collection)

        except Exception as e:
            result.validation_errors.append(f"Hybrid search functionality test failed: {e}")
            result.is_compliant = False

            # Try cleanup anyway
            with contextlib.suppress(Exception):
                await backend.delete_collection(test_collection)

    async def _test_embedding_provider_functionality(
        self, provider: EmbeddingProvider, result: ComplianceResult
    ) -> None:
        """Test EmbeddingProvider functionality."""
        test_texts = ["Hello world", "Test document", "Code example"]
        test_query = "search query"

        try:
            # Test properties
            result.total_tests += 4

            provider_name = provider.provider_name
            if isinstance(provider_name, str):
                result.passed_tests += 1
            else:
                result.validation_errors.append("provider_name must return string")

            model_name = provider.model_name
            if isinstance(model_name, str):
                result.passed_tests += 1
            else:
                result.validation_errors.append("model_name must return string")

            dimension = provider.dimension
            if isinstance(dimension, int) and dimension > 0:
                result.passed_tests += 1
            else:
                result.validation_errors.append("dimension must return positive integer")

            # Test provider info
            provider_info = provider.get_provider_info()
            if isinstance(provider_info, ProviderInfo):
                result.passed_tests += 1
            else:
                result.validation_errors.append("get_provider_info must return ProviderInfo")

            # Test document embeddings
            result.total_tests += 1
            start_time = time.time()
            doc_embeddings = await provider.embed_documents(test_texts)
            result.performance_metrics["embed_documents"] = (time.time() - start_time) * 1000

            if not isinstance(doc_embeddings, list):
                result.validation_errors.append("embed_documents must return list")
            elif len(doc_embeddings) != len(test_texts):
                result.validation_errors.append(
                    "embed_documents must return one embedding per text"
                )
            elif not all(isinstance(emb, list) and len(emb) == dimension for emb in doc_embeddings):
                result.validation_errors.append(
                    "embed_documents must return embeddings with correct dimension"
                )
            else:
                result.passed_tests += 1

            # Test query embedding
            result.total_tests += 1
            start_time = time.time()
            query_embedding = await provider.embed_query(test_query)
            result.performance_metrics["embed_query"] = (time.time() - start_time) * 1000

            if not isinstance(query_embedding, list):
                result.validation_errors.append("embed_query must return list")
            elif len(query_embedding) != dimension:
                result.validation_errors.append(
                    "embed_query must return embedding with correct dimension"
                )
            else:
                result.passed_tests += 1

        except Exception as e:
            result.validation_errors.append(f"Embedding provider functionality test failed: {e}")
            result.is_compliant = False

    async def _test_rerank_provider_functionality(
        self, provider: RerankProvider, result: ComplianceResult
    ) -> None:
        """Test RerankProvider functionality."""
        test_query = "test search query"
        test_documents = [
            "This is a relevant document about testing",
            "Another document with different content",
            "A third document for reranking tests",
        ]

        try:
            # Test properties
            result.total_tests += 3

            provider_name = provider.provider_name
            if isinstance(provider_name, str):
                result.passed_tests += 1
            else:
                result.validation_errors.append("provider_name must return string")

            model_name = provider.model_name
            if isinstance(model_name, str):
                result.passed_tests += 1
            else:
                result.validation_errors.append("model_name must return string")

            # Test provider info
            provider_info = provider.get_provider_info()
            if isinstance(provider_info, ProviderInfo):
                result.passed_tests += 1
            else:
                result.validation_errors.append("get_provider_info must return ProviderInfo")

            # Test reranking
            result.total_tests += 1
            start_time = time.time()
            rerank_results = await provider.rerank(
                query=test_query, documents=test_documents, top_k=3
            )
            result.performance_metrics["rerank"] = (time.time() - start_time) * 1000

            if not isinstance(rerank_results, list):
                result.validation_errors.append("rerank must return list")
            elif not all(isinstance(r, RerankResult) for r in rerank_results):
                result.validation_errors.append("rerank must return list of RerankResult")
            elif len(rerank_results) > len(test_documents):
                result.validation_errors.append(
                    "rerank must not return more results than input documents"
                )
            else:
                result.passed_tests += 1

        except Exception as e:
            result.validation_errors.append(f"Rerank provider functionality test failed: {e}")
            result.is_compliant = False

    async def _test_data_source_functionality(
        self, source: DataSource, result: ComplianceResult
    ) -> None:
        """Test DataSource functionality."""
        test_config: SourceConfig = {
            "enabled": True,
            "priority": 1,
            "source_id": "test_source",
            "include_patterns": [],
            "exclude_patterns": [],
            "max_file_size_mb": 10,
            "batch_size": 10,
            "max_concurrent_requests": 5,
            "request_timeout_seconds": 30,
            "enable_change_watching": False,
            "change_check_interval_seconds": 60,
            "enable_content_deduplication": False,
            "enable_metadata_extraction": True,
            "supported_languages": ["python", "javascript"],
        }

        try:
            # Test capabilities
            result.total_tests += 1
            capabilities = source.get_capabilities()
            if isinstance(capabilities, set) and all(
                isinstance(cap, SourceCapability) for cap in capabilities
            ):
                result.passed_tests += 1
            else:
                result.validation_errors.append(
                    "get_capabilities must return set of SourceCapability"
                )

            # Test source validation
            result.total_tests += 1
            start_time = time.time()
            is_valid = await source.validate_source(test_config)
            result.performance_metrics["validate_source"] = (time.time() - start_time) * 1000

            if isinstance(is_valid, bool):
                result.passed_tests += 1
            else:
                result.validation_errors.append("validate_source must return boolean")

            # Test content discovery (if source is valid)
            if is_valid:
                result.total_tests += 1
                start_time = time.time()
                content_items = await source.discover_content(test_config)
                result.performance_metrics["discover_content"] = (time.time() - start_time) * 1000

                if not isinstance(content_items, list):
                    result.validation_errors.append("discover_content must return list")
                elif content_items and not all(
                    isinstance(item, ContentItem) for item in content_items
                ):
                    result.validation_errors.append(
                        "discover_content must return list of ContentItem"
                    )
                else:
                    result.passed_tests += 1

                    # Test content reading if we have items
                    if content_items:
                        result.total_tests += 1
                        try:
                            start_time = time.time()
                            content = await source.read_content(content_items[0])
                            result.performance_metrics["read_content"] = (
                                time.time() - start_time
                            ) * 1000

                            if isinstance(content, str):
                                result.passed_tests += 1
                            else:
                                result.validation_errors.append("read_content must return string")
                        except Exception as e:
                            result.validation_errors.append(f"read_content failed: {e}")

                        # Test metadata extraction
                        result.total_tests += 1
                        try:
                            metadata = await source.get_content_metadata(content_items[0])
                            if isinstance(metadata, dict):
                                result.passed_tests += 1
                            else:
                                result.validation_errors.append(
                                    "get_content_metadata must return dict"
                                )
                        except Exception as e:
                            result.validation_errors.append(f"get_content_metadata failed: {e}")

        except Exception as e:
            result.validation_errors.append(f"Data source functionality test failed: {e}")
            result.is_compliant = False

    async def _benchmark_vector_backend(
        self, backend: VectorBackend, result: ComplianceResult
    ) -> None:
        """Run performance benchmarks for vector backend."""
        # This would run more extensive performance tests

    async def _benchmark_hybrid_search_backend(
        self, backend: HybridSearchBackend, result: ComplianceResult
    ) -> None:
        """Run performance benchmarks for hybrid search backend."""

    async def _benchmark_embedding_provider(
        self, provider: EmbeddingProvider, result: ComplianceResult
    ) -> None:
        """Run performance benchmarks for embedding provider."""

    async def _benchmark_rerank_provider(
        self, provider: RerankProvider, result: ComplianceResult
    ) -> None:
        """Run performance benchmarks for rerank provider."""

    async def _benchmark_data_source(self, source: DataSource, result: ComplianceResult) -> None:
        """Run performance benchmarks for data source."""

    def _get_test_vectors(self) -> list[VectorPoint]:
        """Get test vectors for backend testing."""
        if "test_vectors" not in self.test_data_cache:
            vectors = []
            for i in range(5):
                # Create simple test vectors
                vector = [float(j + i * 0.1) for j in range(128)]
                # Normalize to unit length for cosine similarity
                length = sum(x * x for x in vector) ** 0.5
                vector = [x / length for x in vector]

                vectors.append(
                    VectorPoint(
                        id=f"test_vector_{i}",
                        vector=vector,
                        payload={"content": f"Test content {i}", "category": "test"},
                    )
                )

            self.test_data_cache["test_vectors"] = vectors

        return self.test_data_cache["test_vectors"]

    def _get_test_vectors_with_sparse(self) -> list[VectorPoint]:
        """Get test vectors with sparse data for hybrid search testing."""
        vectors = self._get_test_vectors()

        # Add sparse vector data
        for i, vector in enumerate(vectors):
            vector.sparse_vector = {j: float(j + i) for j in range(0, 10, 2)}

        return vectors


# Convenience functions for specific protocol validation


async def validate_vector_backend_protocol(backend: VectorBackend) -> ComplianceResult:
    """Validate a VectorBackend implementation."""
    validator = ProtocolComplianceValidator()
    return await validator.validate_vector_backend(backend)


async def validate_hybrid_search_backend_protocol(backend: HybridSearchBackend) -> ComplianceResult:
    """Validate a HybridSearchBackend implementation."""
    validator = ProtocolComplianceValidator()
    return await validator.validate_hybrid_search_backend(backend)


async def validate_embedding_provider_protocol(provider: EmbeddingProvider) -> ComplianceResult:
    """Validate an EmbeddingProvider implementation."""
    validator = ProtocolComplianceValidator()
    return await validator.validate_embedding_provider(provider)


async def validate_rerank_provider_protocol(provider: RerankProvider) -> ComplianceResult:
    """Validate a RerankProvider implementation."""
    validator = ProtocolComplianceValidator()
    return await validator.validate_rerank_provider(provider)


async def validate_data_source_protocol(source: DataSource) -> ComplianceResult:
    """Validate a DataSource implementation."""
    validator = ProtocolComplianceValidator()
    return await validator.validate_data_source(source)
